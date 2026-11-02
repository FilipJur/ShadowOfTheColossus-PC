// SPDX-License-Identifier: GPL-2.0-or-later
// sotc: AMD FSR 3.1 temporal upscaler frontend, see vk_fsr3up.h. Captures the game's inputs
// each frame (colour in Prepare, depth/motion vectors/jitter/camera via Dlssg::GetFrameInputs
// and FsrFg::DepthImage) and runs the active Upscaler::Backend (vk_upscaler.h: built-in FSR 3.1,
// FFX API DLL via OptiScaler, or FSR 4). The backend owns the FFX bridge lifecycle; this file
// owns the HUD split (upscale_hud.comp), the reactive mask (fsr3_reactive.comp) and the stats.

#include "video_core/renderer_vulkan/vk_fsr3up.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <vk_mem_alloc.h>

#include "video_core/renderer_vulkan/vk_dlssg.h"
#include "video_core/renderer_vulkan/vk_fsrfg.h"
#include "video_core/renderer_vulkan/vk_upscaler.h"

namespace Vulkan::Upscaler {
void NoteFsr3Scheduler(Scheduler& scheduler);
} // namespace Vulkan::Upscaler
#include "video_core/host_shaders/fsr3_reactive_comp.h"
#include "video_core/host_shaders/upscale_hud_comp.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan::Fsr3Up {

namespace {

constexpr u32 kMaxFrames = 16; // presentation frames, as vk_dlssg.cpp
constexpr vk::Format kColorFormat = vk::Format::eR16G16B16A16Sfloat;
constexpr vk::ImageUsageFlags kUsage =
    vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
    vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
using Clock = std::chrono::steady_clock;

struct OwnedImage {
    vk::Image image;
    VmaAllocation alloc{};
    vk::ImageView view;
    vk::ImageView swapped_view; ///< red and blue exchanged (outputs only)
    u32 width{}, height{};
};

const Instance* g_instance{};
bool failed = false;
int log_level = 0; // SOTC_FSR3UP_LOG

// The game image of each presentation frame (command processor thread writes, present thread reads).
std::array<OwnedImage, kMaxFrames> colors;
std::array<bool, kMaxFrames> color_ready{};
std::array<bool, kMaxFrames> color_swapped{}; ///< the game image holds red and blue exchanged
std::array<OwnedImage, kMaxFrames> outputs;
Clock::time_point last_time{};
float interval_ms = 16.6f;

// sotc: a frame the upscaler skips is shown as the bilinear stretch of the game image (blurry,
// blocky at 720p), with nothing in the log. Counted per reason and reported every 600 frames
// when any frame was skipped (always with SOTC_FSR3UP_LOG=1).
enum Skip { SkipColour, SkipInputs, SkipSize, SkipError, SkipCount };
std::array<u32, SkipCount> skipped{};
u32 upscaled = 0;

void CountFrame(int skip) {
    if (skip < 0) {
        ++upscaled;
    } else {
        ++skipped[skip];
    }
    u32 total = upscaled;
    u32 missed = 0;
    for (const u32 n : skipped) {
        total += n;
        missed += n;
    }
    if (total < 600) {
        return;
    }
    if (missed || log_level >= 1) {
        std::fprintf(stderr,
                     "SOTC FSR3 upscaler: last %u frames %u upscaled, %u bilinear (no colour %u, no "
                     "depth/motion %u, size %u, error %u)\n",
                     total, upscaled, missed, skipped[SkipColour], skipped[SkipInputs],
                     skipped[SkipSize], skipped[SkipError]);
    }
    upscaled = 0;
    skipped = {};
}

bool CreateImage(OwnedImage& img, vk::Format format, u32 width, u32 height) {
    const VkImageCreateInfo info = static_cast<VkImageCreateInfo>(vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = kUsage,
        .initialLayout = vk::ImageLayout::eUndefined,
    });
    const VmaAllocationCreateInfo alloc_info = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    VkImage raw{};
    if (vmaCreateImage(g_instance->GetAllocator(), &info, &alloc_info, &raw, &img.alloc, nullptr) !=
        VK_SUCCESS) {
        img = {};
        return false;
    }
    img.image = vk::Image{raw};
    auto [result, view] = g_instance->GetDevice().createImageView(vk::ImageViewCreateInfo{
        .image = img.image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    });
    if (result != vk::Result::eSuccess) {
        vmaDestroyImage(g_instance->GetAllocator(), img.image, img.alloc);
        img = {};
        return false;
    }
    img.view = view;
    // The game's frame is viewed as B8G8R8A8 while its image is R8G8B8A8 (or the other way round):
    // the copy keeps the stored order, so the result is read back through exchanged channels.
    auto [swap_result, swapped] = g_instance->GetDevice().createImageView(vk::ImageViewCreateInfo{
        .image = img.image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .components = {vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eR,
                       vk::ComponentSwizzle::eA},
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    });
    if (swap_result == vk::Result::eSuccess) {
        img.swapped_view = swapped;
    }
    img.width = width;
    img.height = height;
    return true;
}

void DestroyImage(OwnedImage& img) {
    if (!g_instance) {
        img = {};
        return;
    }
    if (img.view) {
        g_instance->GetDevice().destroyImageView(img.view);
    }
    if (img.swapped_view) {
        g_instance->GetDevice().destroyImageView(img.swapped_view);
    }
    if (img.image) {
        vmaDestroyImage(g_instance->GetAllocator(), img.image, img.alloc);
    }
    img = {};
}

vk::ImageMemoryBarrier2 Barrier(vk::Image image, vk::PipelineStageFlags2 src_stage,
                                vk::AccessFlags2 src_access, vk::ImageLayout old_layout,
                                vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access,
                                vk::ImageLayout new_layout) {
    return vk::ImageMemoryBarrier2{
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .image = image,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
}

// sotc: the game frame without its UI layer (Rasterizer::TakeHudless) is what gets upscaled;
// upscale_hud.comp then puts the UI back from the game frame. With the UI in the upscaler's input,
// text and bars trailed behind when the camera turned. SOTC_UPSCALER_HUD=0: the game frame, UI and
// all, as before.
std::array<OwnedImage, kMaxFrames> hudless_colors;
std::array<bool, kMaxFrames> hudless_ready{};
vk::ShaderModule hud_shader;
vk::UniqueDescriptorSetLayout hud_set_layout;
vk::UniquePipelineLayout hud_layout;
vk::UniquePipeline hud_pipeline;
vk::UniqueSampler hud_sampler;
bool hud_failed = false;

bool HudSplitAsked() {
    static const bool on = [] {
        const char* v = std::getenv("SOTC_UPSCALER_HUD");
        return !(v && v[0] == '0');
    }();
    return on;
}

bool EnsureHudPipeline() {
    if (hud_pipeline || hud_failed) {
        return bool(hud_pipeline);
    }
    hud_failed = true; // until every part below exists
    const vk::Device device = g_instance->GetDevice();
    hud_shader = CompileSPV(UPSCALE_HUD_COMP, device);
    const std::array bindings{
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 2,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
    };
    auto [set_result, set_layout] = device.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    });
    const vk::PushConstantRange push_range = {
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = 2 * sizeof(u32),
    };
    if (!hud_shader || set_result != vk::Result::eSuccess) {
        std::fprintf(stderr, "SOTC upscaler: UI pass unavailable, the UI is upscaled too\n");
        return false;
    }
    const vk::DescriptorSetLayout layouts[] = {*set_layout};
    auto [layout_result, layout] = device.createPipelineLayoutUnique({
        .setLayoutCount = 1,
        .pSetLayouts = layouts,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    });
    auto [sampler_result, sampler] = device.createSamplerUnique({
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
    });
    if (layout_result != vk::Result::eSuccess || sampler_result != vk::Result::eSuccess) {
        std::fprintf(stderr, "SOTC upscaler: UI pass unavailable, the UI is upscaled too\n");
        return false;
    }
    auto [pipeline_result, pipeline] = device.createComputePipelineUnique(
        {}, vk::ComputePipelineCreateInfo{
                .stage{
                    .stage = vk::ShaderStageFlagBits::eCompute,
                    .module = hud_shader,
                    .pName = "main",
                },
                .layout = *layout,
            });
    if (pipeline_result != vk::Result::eSuccess) {
        std::fprintf(stderr, "SOTC upscaler: UI pass unavailable, the UI is upscaled too\n");
        return false;
    }
    hud_set_layout = std::move(set_layout);
    hud_layout = std::move(layout);
    hud_sampler = std::move(sampler);
    hud_pipeline = std::move(pipeline);
    hud_failed = false;
    std::fprintf(stderr, "SOTC upscaler: the UI layer is put back after upscaling\n");
    return true;
}

/// The upscaled frame without the UI (`out`, ShaderReadOnlyOptimal) gets the UI back from the
/// game frame (in place: the backend's output image).
void PutUiBack(vk::CommandBuffer cmdbuf, u32 frame_id, const OwnedImage& out, u32 out_width,
               u32 out_height) {
    const auto to_general = Barrier(out.image, vk::PipelineStageFlagBits2::eAllCommands,
                                    vk::AccessFlagBits2::eMemoryRead,
                                    vk::ImageLayout::eShaderReadOnlyOptimal,
                                    vk::PipelineStageFlagBits2::eComputeShader,
                                    vk::AccessFlagBits2::eShaderRead |
                                        vk::AccessFlagBits2::eShaderWrite,
                                    vk::ImageLayout::eGeneral);
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                               .pImageMemoryBarriers = &to_general});
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *hud_pipeline);
    const vk::DescriptorImageInfo with_ui = {
        .sampler = *hud_sampler,
        .imageView = colors[frame_id].view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };
    const vk::DescriptorImageInfo without_ui = {
        .sampler = *hud_sampler,
        .imageView = hudless_colors[frame_id].view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };
    const vk::DescriptorImageInfo target = {
        .imageView = out.view,
        .imageLayout = vk::ImageLayout::eGeneral,
    };
    const std::array writes{
        vk::WriteDescriptorSet{
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &with_ui,
        },
        vk::WriteDescriptorSet{
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &without_ui,
        },
        vk::WriteDescriptorSet{
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .pImageInfo = &target,
        },
    };
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *hud_layout, 0, writes);
    const u32 params[2] = {out_width, out_height};
    cmdbuf.pushConstants(*hud_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(params), params);
    cmdbuf.dispatch((out_width + 7) / 8, (out_height + 7) / 8, 1);
    const auto to_sample = Barrier(out.image, vk::PipelineStageFlagBits2::eComputeShader,
                                   vk::AccessFlagBits2::eShaderWrite, vk::ImageLayout::eGeneral,
                                   vk::PipelineStageFlagBits2::eAllCommands,
                                   vk::AccessFlagBits2::eShaderRead,
                                   vk::ImageLayout::eShaderReadOnlyOptimal);
    cmdbuf.pipelineBarrier2(
        vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_sample});
}

// sotc: the reactive mask (fsr3_reactive.comp): pixels the motion vectors do not explain, so the
// upscaler drops its history there (particles, smoke, blended effects left trails).
// SOTC_FSR3_REACTIVE=0 turns it off; a number 0..1 sets the strongest value (default 0.6).
OwnedImage reactive_img;
std::array<OwnedImage, 2> history; // the game frame of the previous dispatch, ping-pong
u32 history_index = 0;
bool history_valid = false;
vk::ShaderModule reactive_shader;
vk::UniqueDescriptorSetLayout reactive_set_layout;
vk::UniquePipelineLayout reactive_layout;
vk::UniquePipeline reactive_pipeline;
vk::UniqueSampler reactive_sampler;
bool reactive_failed = false;

float ReactiveStrength() {
    static const float strength = [] {
        const char* v = std::getenv("SOTC_FSR3_REACTIVE");
        return v ? std::clamp(static_cast<float>(std::atof(v)), 0.0f, 1.0f) : 0.6f;
    }();
    return strength;
}

bool EnsureReactive(u32 width, u32 height) {
    if (reactive_failed || ReactiveStrength() <= 0.0f) {
        return false;
    }
    if (!reactive_pipeline) {
        reactive_failed = true; // until every part below exists
        const vk::Device device = g_instance->GetDevice();
        if (!(g_instance->GetPhysicalDevice().getFormatProperties(vk::Format::eR8Unorm)
                  .optimalTilingFeatures & vk::FormatFeatureFlagBits::eStorageImage)) {
            std::fprintf(stderr, "SOTC upscaler: no R8 storage images, no reactive mask\n");
            return false;
        }
        reactive_shader = CompileSPV(FSR3_REACTIVE_COMP, device);
        std::array<vk::DescriptorSetLayoutBinding, 5> bindings{};
        for (u32 i = 0; i < bindings.size(); ++i) {
            bindings[i] = vk::DescriptorSetLayoutBinding{
                .binding = i,
                .descriptorType = i < 3 ? vk::DescriptorType::eCombinedImageSampler
                                        : vk::DescriptorType::eStorageImage,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eCompute,
            };
        }
        auto [set_result, set_layout] = device.createDescriptorSetLayoutUnique({
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        });
        if (!reactive_shader || set_result != vk::Result::eSuccess) {
            std::fprintf(stderr, "SOTC upscaler: reactive mask unavailable\n");
            return false;
        }
        const vk::PushConstantRange push_range = {
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset = 0,
            .size = 4 * sizeof(u32),
        };
        const vk::DescriptorSetLayout layouts[] = {*set_layout};
        auto [layout_result, layout] = device.createPipelineLayoutUnique({
            .setLayoutCount = 1,
            .pSetLayouts = layouts,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_range,
        });
        auto [sampler_result, sampler] = device.createSamplerUnique({
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .mipmapMode = vk::SamplerMipmapMode::eNearest,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
            .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        });
        if (layout_result != vk::Result::eSuccess || sampler_result != vk::Result::eSuccess) {
            std::fprintf(stderr, "SOTC upscaler: reactive mask unavailable\n");
            return false;
        }
        auto [pipeline_result, pipeline] = device.createComputePipelineUnique(
            {}, vk::ComputePipelineCreateInfo{
                    .stage{
                        .stage = vk::ShaderStageFlagBits::eCompute,
                        .module = reactive_shader,
                        .pName = "main",
                    },
                    .layout = *layout,
                });
        if (pipeline_result != vk::Result::eSuccess) {
            std::fprintf(stderr, "SOTC upscaler: reactive mask unavailable\n");
            return false;
        }
        reactive_set_layout = std::move(set_layout);
        reactive_layout = std::move(layout);
        reactive_sampler = std::move(sampler);
        reactive_pipeline = std::move(pipeline);
        reactive_failed = false;
        std::fprintf(stderr, "SOTC upscaler: reactive mask on (strength %.2f)\n",
                     ReactiveStrength());
    }
    if (reactive_img.width != width || reactive_img.height != height) {
        // A new size: wait for the old images' last readers.
        (void)g_instance->GetDevice().waitIdle();
        DestroyImage(reactive_img);
        for (auto& h : history) {
            DestroyImage(h);
        }
        if (!CreateImage(reactive_img, vk::Format::eR8Unorm, width, height) ||
            !CreateImage(history[0], kColorFormat, width, height) ||
            !CreateImage(history[1], kColorFormat, width, height)) {
            std::fprintf(stderr, "SOTC upscaler: cannot create the reactive mask images\n");
            reactive_failed = true;
            return false;
        }
        history_valid = false;
    }
    return true;
}

/// Writes reactive_img (ShaderReadOnlyOptimal after, for the backend to read) from `color`
/// (ShaderReadOnlyOptimal) and the motion vectors, and copies `color` to the history.
void RecordReactive(vk::CommandBuffer cmdbuf, const OwnedImage& color, vk::ImageView mvec_view,
                    bool reset) {
    OwnedImage& prev = history[history_index ^ 1];
    OwnedImage& next = history[history_index];
    const bool valid = history_valid && !reset;
    const std::array pre{
        Barrier(reactive_img.image, vk::PipelineStageFlagBits2::eAllCommands,
                vk::AccessFlagBits2::eNone, vk::ImageLayout::eUndefined,
                vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
                vk::ImageLayout::eGeneral),
        Barrier(next.image, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eNone,
                vk::ImageLayout::eUndefined, vk::PipelineStageFlagBits2::eComputeShader,
                vk::AccessFlagBits2::eShaderWrite, vk::ImageLayout::eGeneral),
        Barrier(prev.image, vk::PipelineStageFlagBits2::eComputeShader,
                vk::AccessFlagBits2::eShaderWrite,
                history_valid ? vk::ImageLayout::eGeneral : vk::ImageLayout::eUndefined,
                vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead,
                vk::ImageLayout::eGeneral),
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = static_cast<u32>(pre.size()),
                                               .pImageMemoryBarriers = pre.data()});
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *reactive_pipeline);
    const std::array infos{
        vk::DescriptorImageInfo{*reactive_sampler, color.view, vk::ImageLayout::eShaderReadOnlyOptimal},
        vk::DescriptorImageInfo{*reactive_sampler, prev.view, vk::ImageLayout::eGeneral},
        vk::DescriptorImageInfo{*reactive_sampler, mvec_view, vk::ImageLayout::eShaderReadOnlyOptimal},
        vk::DescriptorImageInfo{{}, reactive_img.view, vk::ImageLayout::eGeneral},
        vk::DescriptorImageInfo{{}, next.view, vk::ImageLayout::eGeneral},
    };
    std::array<vk::WriteDescriptorSet, 5> writes{};
    for (u32 i = 0; i < writes.size(); ++i) {
        writes[i] = vk::WriteDescriptorSet{
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = i < 3 ? vk::DescriptorType::eCombinedImageSampler
                                    : vk::DescriptorType::eStorageImage,
            .pImageInfo = &infos[i],
        };
    }
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *reactive_layout, 0, writes);
    const float strength = ReactiveStrength();
    u32 params[4] = {color.width, color.height, valid ? 1u : 0u, 0};
    std::memcpy(&params[3], &strength, sizeof(strength));
    cmdbuf.pushConstants(*reactive_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(params),
                         params);
    cmdbuf.dispatch((color.width + 7) / 8, (color.height + 7) / 8, 1);
    const auto post = Barrier(reactive_img.image, vk::PipelineStageFlagBits2::eComputeShader,
                              vk::AccessFlagBits2::eShaderWrite, vk::ImageLayout::eGeneral,
                              vk::PipelineStageFlagBits2::eAllCommands,
                              vk::AccessFlagBits2::eShaderRead,
                              vk::ImageLayout::eShaderReadOnlyOptimal);
    cmdbuf.pipelineBarrier2(
        vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &post});
    history_index ^= 1;
    history_valid = true;
}

} // namespace

void Init(const Instance& instance) {
    g_instance = &instance;
    if (const char* l = std::getenv("SOTC_FSR3UP_LOG")) {
        log_level = std::atoi(l);
    }
    if (AskedAtStart()) {
        // sotc: the backend logs its own line (or the fallback); here only whether the frontend
        // captures inputs at all.
        std::fprintf(stderr, "SOTC upscaler frontend: inputs %s\n",
                     Upscaler::Active() ? "captured" : "not captured (backend unavailable)");
    }
}

bool Wanted() {
    return AskedAtStart() && Upscaler::Active() != nullptr && !failed;
}

bool AskedAtStart() {
    static const bool asked = [] {
        const char* v = std::getenv("SOTC_UPSCALER");
        if (!v || !v[0]) {
            return false;
        }
        // sotc: any non-empty SOTC_UPSCALER in {fsr3, ffxapi, fsr4} (Upscaler::Requested); the
        // backend factory falls back to built-in FSR 3.1 when the asked one is unavailable.
        return Upscaler::Requested() != Upscaler::Kind::None;
    }();
    return asked;
}

float MipBias() {
    static const float bias = [] {
        if (const char* v = std::getenv("SOTC_MIP_BIAS")) {
            return static_cast<float>(std::atof(v));
        }
        if (!AskedAtStart()) {
            return 0.0f;
        }
        // The game's render size (1080p unless SOTC_RENDER_RES) against the output size.
        unsigned render_w = 1920, render_h = 1080, out_w = 0, out_h = 0;
        if (const char* v = std::getenv("SOTC_RENDER_RES")) {
            std::sscanf(v, "%ux%u", &render_w, &render_h);
        }
        if (const char* v = std::getenv("SOTC_OUTPUT_RES")) {
            std::sscanf(v, "%ux%u", &out_w, &out_h);
        }
        if (render_w == 0 || out_w == 0) {
            return 0.0f;
        }
        // sotc: at most -1. Finer mips fill the game's texture page pool: DLSS Performance at 4K
        // (-2) left it 0-1% free and the far ground black and the grass as pale cards; -1 kept the
        // same scene normal (RTX 5060 Ti 8 GB, user test 29 Oct).
        const float value = std::max(
            std::log2(static_cast<float>(render_w) / static_cast<float>(out_w)) - 1.0f, -1.0f);
        if (log_level > 0) {
            std::fprintf(stderr, "SOTC upscaler: texture LOD bias %.2f (render %ux%u, output %ux%u)\n",
                         value, render_w, render_h, out_w, out_h);
        }
        return value;
    }();
    return bias;
}

void Prepare(vk::CommandBuffer cmdbuf, u32 frame_id, vk::Image game_image, u32 width, u32 height,
             bool swap_red_blue, vk::Image hudless_image) {
    if (!Wanted() || frame_id >= kMaxFrames) {
        return;
    }
    color_ready[frame_id] = false;
    hudless_ready[frame_id] = false;
    color_swapped[frame_id] = swap_red_blue;
    OwnedImage& color = colors[frame_id];
    if (color.width != width || color.height != height) {
        // Its last reader is done: the frame was handed out again after its present.
        if (color.image) {
            DestroyImage(color);
        }
        if (!CreateImage(color, kColorFormat, width, height)) {
            return;
        }
    }
    const vk::MemoryBarrier2 earlier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
    };
    const auto to_dst = Barrier(color.image, vk::PipelineStageFlagBits2::eAllCommands,
                                vk::AccessFlagBits2::eNone, vk::ImageLayout::eUndefined,
                                vk::PipelineStageFlagBits2::eBlit,
                                vk::AccessFlagBits2::eTransferWrite,
                                vk::ImageLayout::eTransferDstOptimal);
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1,
                                               .pMemoryBarriers = &earlier,
                                               .imageMemoryBarrierCount = 1,
                                               .pImageMemoryBarriers = &to_dst});
    const vk::ImageBlit blit{
        .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .srcOffsets = std::array{vk::Offset3D{0, 0, 0},
                                 vk::Offset3D{static_cast<s32>(width), static_cast<s32>(height), 1}},
        .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .dstOffsets = std::array{vk::Offset3D{0, 0, 0},
                                 vk::Offset3D{static_cast<s32>(width), static_cast<s32>(height), 1}},
    };
    cmdbuf.blitImage(game_image, vk::ImageLayout::eTransferSrcOptimal, color.image,
                     vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eNearest);
    const auto to_read = Barrier(color.image, vk::PipelineStageFlagBits2::eBlit,
                                 vk::AccessFlagBits2::eTransferWrite,
                                 vk::ImageLayout::eTransferDstOptimal,
                                 vk::PipelineStageFlagBits2::eAllCommands,
                                 vk::AccessFlagBits2::eShaderRead,
                                 vk::ImageLayout::eShaderReadOnlyOptimal);
    cmdbuf.pipelineBarrier2(
        vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_read});
    color_ready[frame_id] = true;

    // The same frame without its UI layer (ShaderReadOnlyOptimal, left so for frame generation).
    OwnedImage& hudless = hudless_colors[frame_id];
    if (!hudless_image || !HudSplitAsked()) {
        return;
    }
    if (hudless.width != width || hudless.height != height) {
        if (hudless.image) {
            DestroyImage(hudless);
        }
        if (!CreateImage(hudless, kColorFormat, width, height)) {
            return;
        }
    }
    const std::array pre{
        Barrier(hudless_image, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eNone,
                vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits2::eBlit,
                vk::AccessFlagBits2::eTransferRead, vk::ImageLayout::eTransferSrcOptimal),
        Barrier(hudless.image, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eNone,
                vk::ImageLayout::eUndefined, vk::PipelineStageFlagBits2::eBlit,
                vk::AccessFlagBits2::eTransferWrite, vk::ImageLayout::eTransferDstOptimal),
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = static_cast<u32>(pre.size()),
                                               .pImageMemoryBarriers = pre.data()});
    cmdbuf.blitImage(hudless_image, vk::ImageLayout::eTransferSrcOptimal, hudless.image,
                     vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eNearest);
    const std::array post{
        Barrier(hudless_image, vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eNone,
                vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eAllCommands,
                vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eShaderReadOnlyOptimal),
        Barrier(hudless.image, vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferWrite,
                vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits2::eAllCommands,
                vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eShaderReadOnlyOptimal),
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = static_cast<u32>(post.size()),
                                               .pImageMemoryBarriers = post.data()});
    hudless_ready[frame_id] = true;
}

bool SplitsUi() {
    return Wanted() && HudSplitAsked();
}

vk::ImageView Dispatch(vk::CommandBuffer cmdbuf, Scheduler& scheduler, u32 frame_id, u32 out_width,
                       u32 out_height) {
    if (!Wanted()) {
        return {};
    }
    Upscaler::Backend* backend = Upscaler::Active();
    if (!backend) {
        return {};
    }
    if (frame_id >= kMaxFrames || !color_ready[frame_id]) {
        CountFrame(SkipColour);
        return {};
    }
    // The frame without its UI when Prepare had it: PutUiBack adds the UI after upscaling.
    const bool split_ui = hudless_ready[frame_id] && EnsureHudPipeline();
    const OwnedImage& color = split_ui ? hudless_colors[frame_id] : colors[frame_id];
    const u32 width = color.width, height = color.height;
    Dlssg::FrameInputs in{};
    vk::Image depth;
    u32 depth_w = 0, depth_h = 0;
    if (!Dlssg::GetFrameInputs(frame_id, in) || !FsrFg::DepthImage(frame_id, depth, depth_w, depth_h)) {
        CountFrame(SkipInputs);
        return {};
    }
    if (in.width != width || in.height != height || depth_w != width || depth_h != height) {
        CountFrame(SkipSize);
        return {};
    }
    OwnedImage& out = outputs[frame_id];
    if (out.width != out_width || out.height != out_height) {
        DestroyImage(out);
        if (!CreateImage(out, kColorFormat, out_width, out_height)) {
            std::fprintf(stderr, "SOTC upscaler: cannot create the %ux%u output\n", out_width,
                         out_height);
            DestroyImage(out);
            failed = true;
            CountFrame(SkipError);
            return {};
        }
    }
    const auto now = Clock::now();
    if (last_time.time_since_epoch().count() != 0) {
        const float ms = std::chrono::duration<float, std::milli>(now - last_time).count();
        if (ms > 2.0f && ms < 100.0f) {
            interval_ms = interval_ms * 0.85f + ms * 0.15f;
        }
    }
    last_time = now;
    const bool reset = in.reset;
    // sotc: the backend leaves its output ShaderReadOnlyOptimal (as its contract promises); the
    // old pass assumed General there, so the output is fully rewritten here first.
    const auto to_general = Barrier(out.image, vk::PipelineStageFlagBits2::eAllCommands,
                                    vk::AccessFlagBits2::eNone, vk::ImageLayout::eUndefined,
                                    vk::PipelineStageFlagBits2::eAllCommands,
                                    vk::AccessFlagBits2::eMemoryWrite, vk::ImageLayout::eGeneral);
    cmdbuf.pipelineBarrier2(
        vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_general});

    // The reactive mask from this colour and the motion vectors (General for the backend here,
    // ShaderReadOnlyOptimal once recorded).
    Upscaler::Resource reactive{};
    bool show_reactive = false;
    if (in.mvec_view && EnsureReactive(width, height)) {
        RecordReactive(cmdbuf, color, in.mvec_view, reset);
        reactive = {
            .image = reactive_img.image,
            .view = reactive_img.view,
            .format = vk::Format::eR8Unorm,
            .size = vk::Extent2D{width, height},
            .layout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };
    }
    // SOTC_FSR3_REACTIVE_VIEW=1 (diagnostic): the reactive mask is shown instead of the frame.
    static const bool show_reactive_env = [] {
        const char* v = std::getenv("SOTC_FSR3_REACTIVE_VIEW");
        return v && v[0] == '1';
    }();
    show_reactive = show_reactive_env && reactive.image != vk::Image{};

    Upscaler::Inputs inputs{};
    inputs.color = {
        .image = color.image,
        .view = color.view,
        .format = kColorFormat,
        .size = vk::Extent2D{width, height},
        .layout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };
    inputs.depth = {
        .image = depth,
        .view = {},
        .format = vk::Format::eR32Sfloat,
        .size = vk::Extent2D{depth_w, depth_h},
        .layout = vk::ImageLayout::eGeneral,
    };
    inputs.motion_vectors = {
        .image = in.mvec,
        .view = in.mvec_view,
        .format = vk::Format::eR16G16Sfloat,
        .size = vk::Extent2D{in.width, in.height},
        .layout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };
    inputs.reactive = reactive;
    inputs.render_size = vk::Extent2D{width, height};
    inputs.output_size = vk::Extent2D{out_width, out_height};
    // sotc: SOTC_FSR3_JITTER=x|y|xy negates the jitter given to the upscaler (diagnostic: the DLSS
    // convention is used as is; a wrong sign makes FSR 3.1 reject its history, edges stay stepped).
    static const std::pair<float, float> jitter_sign = [] {
        const char* v = std::getenv("SOTC_FSR3_JITTER");
        const bool x = v && std::strchr(v, 'x'), y = v && std::strchr(v, 'y');
        return std::pair{x ? -1.0f : 1.0f, y ? -1.0f : 1.0f};
    }();
    inputs.jitter_x = in.jitter[0] * jitter_sign.first;
    inputs.jitter_y = in.jitter[1] * jitter_sign.second;
    // The game's motion vectors are curUV - prevUV; FSR wants prev - cur, in render pixels.
    inputs.mv_scale_x = -static_cast<float>(width);
    inputs.mv_scale_y = -static_cast<float>(height);
    inputs.frame_time_ms = interval_ms;
    inputs.pre_exposure = 1.0f;
    inputs.sharpness = -1.0f; // disabled: the presenter sharpens over the output (RCAS setting)
    inputs.reset = reset;
    // The colour input is the game's final, tonemapped frame (8 bits per channel), not HDR.
    inputs.hdr = false;
    inputs.camera_near = in.z_near;
    inputs.camera_far = in.z_far;
    inputs.camera_fov_y = in.fov_y;
    inputs.depth_inverted = in.depth_inverted;
    inputs.depth_infinite = false;

    Upscaler::NoteFsr3Scheduler(scheduler);
    Upscaler::Resource out_res{
        .image = out.image,
        .view = out.view,
        .format = kColorFormat,
        .size = vk::Extent2D{out_width, out_height},
        .layout = vk::ImageLayout::eGeneral,
    };
    if (log_level >= 1) {
        static u64 logged = 0;
        if (++logged % 600 == 1 || logged < 4) {
            std::fprintf(stderr,
                         "SOTC upscaler: frame %llu %ux%u -> %ux%u jitter %.3f %.3f px near %.3f "
                         "far %.1f fov %.3f reset %d frame time %.1f ms\n",
                         static_cast<unsigned long long>(logged), width, height, out_width,
                         out_height, in.jitter[0], in.jitter[1], in.z_near, in.z_far, in.fov_y,
                         reset ? 1 : 0, interval_ms);
        }
    }
    if (!backend->Dispatch(cmdbuf, inputs, out_res)) {
        static u32 errors = 0;
        if (errors++ < 5) {
            std::fprintf(stderr, "SOTC upscaler: %s dispatch failed\n", backend->Name());
        }
        CountFrame(SkipError);
        return {};
    }
    if (show_reactive) {
        const std::array pre_view{
            Barrier(reactive_img.image, vk::PipelineStageFlagBits2::eAllCommands,
                    vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferRead,
                    vk::ImageLayout::eGeneral),
            Barrier(out.image, vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderWrite, vk::ImageLayout::eGeneral,
                    vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferRead,
                    vk::ImageLayout::eGeneral),
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 2,
                                                   .pImageMemoryBarriers = pre_view.data()});
        const vk::ImageBlit view_blit{
            .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .srcOffsets = std::array{vk::Offset3D{0, 0, 0},
                                     vk::Offset3D{static_cast<s32>(width), static_cast<s32>(height), 1}},
            .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .dstOffsets = std::array{vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<s32>(out_width),
                                                                         static_cast<s32>(out_height), 1}},
        };
        cmdbuf.blitImage(reactive_img.image, vk::ImageLayout::eGeneral, out.image,
                         vk::ImageLayout::eGeneral, view_blit, vk::Filter::eNearest);
        const auto post_view =
            Barrier(out.image, vk::PipelineStageFlagBits2::eBlit,
                    vk::AccessFlagBits2::eTransferWrite, vk::ImageLayout::eGeneral,
                    vk::PipelineStageFlagBits2::eFragmentShader |
                        vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eShaderReadOnlyOptimal);
        cmdbuf.pipelineBarrier2(
            vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &post_view});
        CountFrame(-1);
        return color_swapped[frame_id] && out.swapped_view ? out.swapped_view : out.view;
    }
    if (split_ui) {
        PutUiBack(cmdbuf, frame_id, out, out_width, out_height);
        CountFrame(-1);
        return color_swapped[frame_id] && out.swapped_view ? out.swapped_view : out.view;
    }
    CountFrame(-1);
    // sotc: the backend leaves its output ShaderReadOnlyOptimal: the presenter samples it next.
    return color_swapped[frame_id] && out.swapped_view ? out.swapped_view : out.view;
}

} // namespace Vulkan::Fsr3Up
