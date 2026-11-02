// SPDX-License-Identifier: GPL-2.0-or-later
// sotc: the built-in FSR 3.1 temporal upscaler backend (FidelityFX SDK 2.3,
// third_party/fsr-vulkan): wraps the FfxVkFsr3_3_1_5 bridge lifecycle (context
// create/dispatch/destroy, jitter phase count) behind the shared Upscaler::Backend
// interface (vk_upscaler.h). The frontend (vk_fsr3up.cpp) captures the game's inputs
// and owns the HUD split, reactive mask and stats; this file only runs the SDK.

#include "video_core/renderer_vulkan/vk_upscaler.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <vector>
#include <vk_mem_alloc.h>

#include <ffx_vk_fsr3_3_1_5_bridge.h>

#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan::Upscaler {
namespace {

constexpr vk::ImageUsageFlags kUsage =
    vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
    vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;

struct OwnedImage {
    vk::Image image;
    VmaAllocation alloc{};
    vk::ImageView view;
    FfxVkFsr3_3_1_5Resource resource{}; ///< bridge import token for shared images (dispatch inputs)
    u32 width{}, height{};
};

struct InFlight {
    u64 frame_id;
    u64 tick;
    FfxVkFsr3_3_1_5Resource color, depth, mvec, reactive, out;
};

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

class Fsr3Builtin final : public Backend {
public:
    Kind GetKind() const override {
        return Kind::Fsr3Builtin;
    }

    const char* Name() const override {
        return "FSR 3.1 (built-in)";
    }

    bool Init(const Instance& instance) override {
        std::scoped_lock lock{mutex};
        if (inited) {
            return supported;
        }
        inited = true;
        g_instance = &instance;
        if (const char* l = std::getenv("SOTC_FSR3UP_LOG")) {
            log_level = std::atoi(l);
        }
        const auto props = instance.GetPhysicalDevice().getFormatProperties(
            vk::Format::eR16G16B16A16Sfloat);
        // sotc: the bridge copies through an fp16 intermediate and renders the output into
        // UAV storage: without blit-destination support the game's colour cannot be imported.
        supported = (props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eBlitDst) !=
                    vk::FormatFeatureFlags{};
        if (!supported) {
            std::fprintf(stderr, "SOTC FSR3 upscaler: not supported here\n");
        }
        return supported;
    }

    bool Dispatch(vk::CommandBuffer cmdbuf, const Inputs& in, const Resource& output) override {
        std::scoped_lock lock{mutex};
        if (!inited || !supported || !g_instance) {
            return false;
        }
        const u32 rw = in.render_size.width, rh = in.render_size.height;
        const u32 ow = output.size.width, oh = output.size.height;
        if (!rw || !rh || !ow || !oh || in.color.image == vk::Image{} ||
            in.depth.image == vk::Image{} || in.motion_vectors.image == vk::Image{} ||
            output.image == vk::Image{}) {
            return false;
        }
        if (!EnsureContext(rw, rh, ow, oh)) {
            return false;
        }
        Retire(false);
        // The output is the caller's image, fully rewritten each dispatch. Its import token
        // (out_res) retires with the in-flight entry; the image barrier below is recorded on
        // the caller's command buffer, so the dispatch's writes happen before it.
        const vk::MemoryBarrier2 earlier = {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        };
        std::vector<vk::ImageMemoryBarrier2> pre;
        pre.push_back(Barrier(output.image, vk::PipelineStageFlagBits2::eAllCommands,
                              vk::AccessFlagBits2::eNone, output.layout,
                              vk::PipelineStageFlagBits2::eComputeShader,
                              vk::AccessFlagBits2::eShaderWrite, vk::ImageLayout::eGeneral));
        if (!shared_ready) {
            shared_ready = true;
            for (auto& sh : shared) {
                pre.push_back(Barrier(sh.image, vk::PipelineStageFlagBits2::eAllCommands,
                                      vk::AccessFlagBits2::eNone, vk::ImageLayout::eUndefined,
                                      vk::PipelineStageFlagBits2::eComputeShader,
                                      vk::AccessFlagBits2::eShaderWrite,
                                      vk::ImageLayout::eGeneral));
            }
        }
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &earlier,
            .imageMemoryBarrierCount = static_cast<u32>(pre.size()),
            .pImageMemoryBarriers = pre.data(),
        });

        // The frame's colour (ShaderReadOnlyOptimal), depth (R32F, General) and motion vectors
        // (ShaderReadOnlyOptimal), imported for this dispatch and released once the GPU is done.
        FfxVkFsr3_3_1_5Resource color_res{}, depth_res{}, mvec_res{}, reactive_res{};
        if (in.reactive.image != vk::Image{} &&
            !Import(reactive_res, in.reactive.image, rw, rh, in.reactive.format,
                    in.reactive.layout, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_COMPUTE_READ)) {
            reactive_res = {};
        }
        if (!Import(color_res, in.color.image, rw, rh, in.color.format,
                    vk::ImageLayout::eShaderReadOnlyOptimal,
                    FFX_VK_FSR3_3_1_5_RESOURCE_STATE_COMPUTE_READ) ||
            !Import(depth_res, in.depth.image, rw, rh, vk::Format::eR32Sfloat,
                    vk::ImageLayout::eGeneral,
                    FFX_VK_FSR3_3_1_5_RESOURCE_STATE_COMPUTE_READ) ||
            !Import(mvec_res, in.motion_vectors.image, rw, rh, in.motion_vectors.format,
                    vk::ImageLayout::eShaderReadOnlyOptimal,
                    FFX_VK_FSR3_3_1_5_RESOURCE_STATE_COMPUTE_READ)) {
            static u32 errors = 0;
            if (errors++ < 5) {
                std::fprintf(stderr, "SOTC FSR3 upscaler: cannot import the frame inputs\n");
            }
            for (auto* r : {&color_res, &depth_res, &mvec_res, &reactive_res}) {
                if (r->resource) {
                    ffxVkFsr3_3_1_5BridgeReleaseImportedImage(bridge, *r);
                }
            }
            return false;
        }
        FfxVkFsr3_3_1_5UpscalerDispatchInfo d{};
        d.commandBuffer = static_cast<VkCommandBuffer>(cmdbuf);
        d.color = color_res;
        d.depth = depth_res;
        d.motionVectors = mvec_res;
        d.reactive = reactive_res;
        d.dilatedDepth = shared[0].resource;
        d.dilatedMotionVectors = shared[1].resource;
        d.reconstructedPrevNearestDepth = shared[2].resource;
        // sotc: the SDK writes the upscaled frame into the caller's image (UAV); the
        // backend only keeps the shared (history) images itself.
        FfxVkFsr3_3_1_5Resource out_res{};
        if (!Import(out_res, output.image, ow, oh, output.format, vk::ImageLayout::eGeneral,
                    FFX_VK_FSR3_3_1_5_RESOURCE_STATE_UNORDERED_ACCESS)) {
            static u32 errors = 0;
            if (errors++ < 5) {
                std::fprintf(stderr, "SOTC FSR3 upscaler: cannot import the output\n");
            }
            for (auto* r : {&color_res, &depth_res, &mvec_res, &reactive_res}) {
                ffxVkFsr3_3_1_5BridgeReleaseImportedImage(bridge, *r);
            }
            return false;
        }
        d.output = out_res;
        // sotc: SOTC_FSR3_JITTER=x|y|xy negates the jitter given to FSR 3.1 (diagnostic: the DLSS
        // convention is used as is; a wrong sign makes FSR 3.1 reject its history, edges stay
        // stepped).
        static const std::pair<float, float> jitter_sign = [] {
            const char* v = std::getenv("SOTC_FSR3_JITTER");
            const bool x = v && std::strchr(v, 'x'), y = v && std::strchr(v, 'y');
            return std::pair{x ? -1.0f : 1.0f, y ? -1.0f : 1.0f};
        }();
        d.jitterOffsetX = in.jitter_x * jitter_sign.first;
        d.jitterOffsetY = in.jitter_y * jitter_sign.second;
        d.motionVectorScaleX = in.mv_scale_x;
        d.motionVectorScaleY = in.mv_scale_y;
        d.renderWidth = rw;
        d.renderHeight = rh;
        d.upscaleWidth = ow;
        d.upscaleHeight = oh;
        d.frameTimeMilliseconds = in.frame_time_ms > 0.0f ? in.frame_time_ms : 16.6f;
        d.preExposure = in.pre_exposure > 0.0f ? in.pre_exposure : 1.0f;
        d.enableSharpening = in.sharpness >= 0.0f ? VK_TRUE : VK_FALSE;
        d.sharpness = std::clamp(in.sharpness, 0.0f, 1.0f);
        // sotc: a fresh context has no history: the first dispatch resets like a camera cut.
        const bool first = frame_counter == 0;
        d.cameraNear = in.camera_near;
        d.cameraFar = std::min(in.camera_far, 1.0e6f);
        d.cameraVerticalFovRadians = in.camera_fov_y;
        d.reset = (first || reset_pending || in.reset) ? VK_TRUE : VK_FALSE;
        reset_pending = false;
        d.frameId = ++frame_counter;
        const auto start = std::chrono::steady_clock::now();
        const FfxVkFsr3_3_1_5Result r = ffxVkFsr3_3_1_5UpscalerContextRecordDispatch(context, &d);
        in_flight.push_back({d.frameId, scheduler_ ? scheduler_->CurrentTick() : 0,
                             color_res, depth_res, mvec_res, reactive_res, out_res});
        if (r != FFX_VK_FSR3_3_1_5_OK) {
            static u32 errors = 0;
            if (errors++ < 5) {
                std::fprintf(stderr, "SOTC FSR3 upscaler: dispatch failed (%d)\n",
                             static_cast<int>(r));
            }
            return false;
        }
        if (log_level >= 1 && (d.frameId % 600 == 1 || d.frameId < 4)) {
            std::fprintf(stderr,
                         "SOTC FSR3 upscaler: frame %llu %ux%u -> %ux%u jitter %.3f %.3f px near "
                         "%.3f far %.1f fov %.3f reset %d frame time %.1f ms\n",
                         static_cast<unsigned long long>(d.frameId), rw, rh, ow, oh, in.jitter_x,
                         in.jitter_y, in.camera_near, in.camera_far, in.camera_fov_y,
                         in.reset ? 1 : 0, d.frameTimeMilliseconds);
        }
        // sotc: the dispatch is recorded on the caller's command buffer (present thread): no
        // submit of our own. Bridge-owned import views retire with the in-flight entry, whose
        // scheduler tick (Retire) is behind the submission holding this dispatch.
        (void)start;
        // sotc: the caller samples the output next: leave it ShaderReadOnlyOptimal, as the
        // FfxApi and Fsr4 backends do.
        const auto to_sample = Barrier(output.image, vk::PipelineStageFlagBits2::eComputeShader,
                                       vk::AccessFlagBits2::eShaderWrite,
                                       vk::ImageLayout::eGeneral,
                                       vk::PipelineStageFlagBits2::eFragmentShader |
                                           vk::PipelineStageFlagBits2::eComputeShader,
                                       vk::AccessFlagBits2::eShaderRead,
                                       vk::ImageLayout::eShaderReadOnlyOptimal);
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                                   .pImageMemoryBarriers = &to_sample});
        return true;
    }

    s32 JitterPhaseCount(u32 render_width, u32 output_width) const override {
        if (!render_width || !output_width) {
            return 8;
        }
        // sotc: ffxFsr3UpscalerGetJitterPhaseCount is scaffold-private in the vendored bridge,
        // so closed form from ffx_fsr3upscaler.cpp: 8 * (display/render)^2.
        const double ratio = double(output_width) / double(render_width);
        return s32(std::ceil(8.0 * ratio * ratio));
    }

    void Shutdown() override {
        std::scoped_lock lock{mutex};
        if (!inited) {
            return;
        }
        Retire(true);
        if (g_instance) {
            (void)g_instance->GetDevice().waitIdle();
        }
        for (auto& s : shared) {
            DestroyShared(s);
        }
        // (no owned output: the SDK writes the caller's image)
        if (context) {
            ffxVkFsr3_3_1_5UpscalerContextDestroy(context);
            context = nullptr;
            bridge = nullptr;
        }
        shared_ready = false;
        frame_counter = 0;
        in_flight.clear();
        inited = false;
        supported = false;
        g_instance = nullptr;
    }
    void DestroyContext() {
        if (!context) {
            return;
        }
        Retire(true);
        if (g_instance) {
            (void)g_instance->GetDevice().waitIdle();
        }
        for (auto& s : shared) {
            DestroyShared(s);
        }
        // (no owned output: the SDK writes the caller's image)
        ffxVkFsr3_3_1_5UpscalerContextDestroy(context);
        context = nullptr;
        bridge = nullptr;
        shared_ready = false;
    }


    void NoteScheduler(Scheduler& scheduler) {
        scheduler_ = &scheduler;
    }

private:
    static bool Fsr3Flag(const char* name, bool fallback) {
        const char* v = std::getenv(name);
        return v && v[0] ? v[0] != '0' : fallback;
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
        if (vmaCreateImage(g_instance->GetAllocator(), &info, &alloc_info, &raw, &img.alloc,
                           nullptr) != VK_SUCCESS) {
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
        img.width = width;
        img.height = height;
        return true;
    }

    bool Import(FfxVkFsr3_3_1_5Resource& resource, vk::Image image, u32 width, u32 height,
                vk::Format format, vk::ImageLayout layout, u32 state) {
        FfxVkFsr3_3_1_5ImportedImageDescription d{};
        d.image = static_cast<VkImage>(image);
        d.format = static_cast<VkFormat>(format);
        d.width = width;
        d.height = height;
        d.mipCount = 1;
        d.arrayLayers = 1;
        d.layout = static_cast<VkImageLayout>(layout);
        d.state = state;
        d.usage = static_cast<VkImageUsageFlags>(kUsage);
        resource = ffxVkFsr3_3_1_5BridgeImportImage(bridge, &d);
        return resource.resource != nullptr;
    }

    void DestroyImage(OwnedImage& img) {
        // sotc: bridge import tokens (per-dispatch inputs, the output, the shared images) are
        // released in Retire, never here: DestroyImage only frees our image and its view.
        if (img.view) {
            g_instance->GetDevice().destroyImageView(img.view);
            img.view = vk::ImageView{};
        }
        if (img.image) {
            vmaDestroyImage(g_instance->GetAllocator(), img.image, img.alloc);
            img = {};
        }
    }

    void Retire(bool wait_all) {
        // sotc: the bridge must retire a frame only after the submission holding its dispatch
        // (the present thread's) signaled: scheduler ticks, as vk_fsrfg.cpp does.
        while (!in_flight.empty() && scheduler_) {
            const InFlight f = in_flight.front();
            if (wait_all) {
                scheduler_->Wait(f.tick);
            } else if (!scheduler_->IsFree(f.tick)) {
                break;
            }
            ffxVkFsr3_3_1_5UpscalerContextRetireFrame(context, f.frame_id);
            ffxVkFsr3_3_1_5BridgeReleaseImportedImage(bridge, f.color);
            ffxVkFsr3_3_1_5BridgeReleaseImportedImage(bridge, f.depth);
            ffxVkFsr3_3_1_5BridgeReleaseImportedImage(bridge, f.mvec);
            if (f.reactive.resource) {
                ffxVkFsr3_3_1_5BridgeReleaseImportedImage(bridge, f.reactive);
            }
            ffxVkFsr3_3_1_5BridgeReleaseImportedImage(bridge, f.out);
            in_flight.pop_front();
        }
    }

    void DestroyShared(OwnedImage& img) {
        // sotc: shared images are imported once with the context (unlike per-dispatch inputs,
        // which Retire releases): release the import before destroying the image, as Lucas does.
        if (img.resource.resource) {
            ffxVkFsr3_3_1_5BridgeReleaseImportedImage(bridge, img.resource);
            img.resource = {};
        }
        DestroyImage(img);
    }


    bool EnsureContext(u32 render_w, u32 render_h, u32 out_w, u32 out_h) {
        if (context && render_w <= ctx_render_w && render_h <= ctx_render_h && out_w == ctx_out_w &&
            out_h == ctx_out_h) {
            return true;
        }
        reset_pending = true; // fresh history: ignore it on the next dispatch
        DestroyContext();
        const FfxVkFsr3_3_1_5UpscalerCreateInfo info = {
            .physicalDevice = static_cast<VkPhysicalDevice>(g_instance->GetPhysicalDevice()),
            .device = static_cast<VkDevice>(g_instance->GetDevice()),
            .maxRenderWidth = render_w,
            .maxRenderHeight = render_h,
            .maxUpscaleWidth = out_w,
            .maxUpscaleHeight = out_h,
            // sotc: the colour input is the game's final, tonemapped frame (8 bits per channel),
            // not HDR. Flagged as HDR with auto exposure (as it was until 0.70-plus2), FSR 3.1
            // compared colours wrongly and dropped its history in motion: grass in blotches, the
            // horse's mane in steps, while a still scene looked fine (2mg, fork issue #5).
            // SOTC_FSR3_HDR=1 / SOTC_FSR3_AUTOEXP=1 bring the old flags back.
            .hdrColorInput = Fsr3Flag("SOTC_FSR3_HDR", false) ? VK_TRUE : VK_FALSE,
            .autoExposure = Fsr3Flag("SOTC_FSR3_AUTOEXP", false) ? VK_TRUE : VK_FALSE,
        };
        const auto start = std::chrono::steady_clock::now();
        if (ffxVkFsr3_3_1_5UpscalerContextCreate(&info, &context) != FFX_VK_FSR3_3_1_5_OK) {
            std::fprintf(stderr, "SOTC FSR3 upscaler: context creation failed, off\n");
            context = nullptr;
            return false;
        }
        bridge = ffxVkFsr3_3_1_5UpscalerContextGetBridge(context);
        FfxVkFsr3_3_1_5SharedResourceDescriptions sd{};
        bool ok = ffxVkFsr3_3_1_5UpscalerContextGetSharedResourceDescriptions(context, &sd) ==
                  FFX_VK_FSR3_3_1_5_OK;
        const FfxVkFsr3_3_1_5SharedResourceDescription* descs[3] = {
            &sd.dilatedDepth, &sd.dilatedMotionVectors, &sd.reconstructedPrevNearestDepth};
        for (int i = 0; ok && i < 3; ++i) {
            const auto format = static_cast<vk::Format>(descs[i]->format);
            ok = format != vk::Format::eUndefined &&
                 CreateImage(shared[i], format, descs[i]->width, descs[i]->height) &&
                 Import(shared[i].resource, shared[i].image, shared[i].width, shared[i].height,
                        format, vk::ImageLayout::eGeneral,
                        FFX_VK_FSR3_3_1_5_RESOURCE_STATE_UNORDERED_ACCESS);
            shared[i].resource = ok ? shared[i].resource : FfxVkFsr3_3_1_5Resource{};
            // sotc: shared[i].resource holds the bridge import token for dispatch; the
            // image/view/alloc stay for DestroyImage.
        }
        if (!ok) {
            std::fprintf(stderr, "SOTC FSR3 upscaler: cannot create its images, off\n");
            DestroyContext();
            return false;
        }
        ctx_render_w = render_w;
        ctx_render_h = render_h;
        ctx_out_w = out_w;
        ctx_out_h = out_h;
        FfxVkFsr3_3_1_5MemoryUsage usage{};
        ffxVkFsr3_3_1_5UpscalerContextGetMemoryUsage(context, &usage);
        std::fprintf(stderr,
                     "SOTC FSR3 upscaler: context %ux%u -> %ux%u, %.0f MB, created in %.0f ms\n",
                     render_w, render_h, out_w, out_h, usage.totalUsageInBytes / 1048576.0,
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                               start)
                         .count());
        return true;
    }

    std::mutex mutex;
    const Instance* g_instance{};
    Scheduler* scheduler_{};
    bool inited{};
    bool supported{};
    int log_level{};
    FfxVkFsr3_3_1_5UpscalerContext* context{};
    FfxVkFsr3_3_1_5Bridge* bridge{};
    u32 ctx_render_w{}, ctx_render_h{}, ctx_out_w{}, ctx_out_h{};
    std::array<OwnedImage, 3> shared; // dilated depth, dilated motion vectors, previous near depth
    bool shared_ready{};
    // (no owned output: the SDK writes the caller's image, fully rewritten each dispatch)
    bool reset_pending = true; // fresh context (Init never dispatched): ignore history next time
    u64 frame_counter{};
    std::deque<InFlight> in_flight;
};

} // namespace

std::unique_ptr<Backend> MakeFsr3Builtin() {
    return std::make_unique<Fsr3Builtin>();
}

// sotc: the frontend (vk_fsr3up.cpp Dispatch, present thread) hands the backend the scheduler
// whose ticks the retirement waits on (as FsrFg::Submitted does). The Backend interface has no
// scheduler parameter, so this side channel carries it; safe because Dispatch and this call are
// on the same thread with no interleaving backend use.
void NoteFsr3Scheduler(Scheduler& scheduler);

void NoteFsr3Scheduler(Scheduler& scheduler) {
    // Reach the live instance without a registry: the backend is a singleton behind Active().
    // The frontend calls Active() first, so this resolves to the same object.
    if (Backend* backend = Active()) {
        if (auto* builtin = dynamic_cast<Fsr3Builtin*>(backend)) {
            builtin->NoteScheduler(scheduler);
        }
    }
}

} // namespace Vulkan::Upscaler
