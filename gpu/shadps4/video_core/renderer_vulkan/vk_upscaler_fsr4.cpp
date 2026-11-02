// SPDX-License-Identifier: GPL-2.0-or-later
// sotc: the FSR 4.1.1 INT8 backend: AMD's 4.1.1 upscaler model replayed natively on Vulkan
// from SPIR-V assets (tools/fsr4cap records AMD's D3D12 DLL, tools/fsr4cap/extract.py builds
// the asset sets: 29 passes per (tier, model) plus initializer.bin). Shares fsr411/ with the
// old bbport game driver (vk_fsr4.cpp at f491c3e^). The fsr411 replay takes raw Vulkan
// handles (plain Vulkan, no renderer types); this backend keeps the hpp handles and converts
// at its boundary.

#include "video_core/renderer_vulkan/vk_upscaler.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

#include "video_core/renderer_vulkan/fsr411/fsr411.h"
#include "video_core/renderer_vulkan/vk_instance.h"

namespace Vulkan::Upscaler {
namespace {

// Pass names in fsr411, needed to hold the backend's transient state for the constant ring
// (Fsr411::Upscaler owns its own lifetime images; the caller only borrows its command buffer).
// The caller keeps the command buffer's frame alive for kFramesInFlight frames; the fsr411
// uniform ring has exactly that many slots, so no extra tracking is needed.

/// Backend-owned Vulkan objects, freed on Shutdown (device idle) or recreation.
struct Owned {
    vk::Device device;
    bool device_idle = false; // waitIdle ran after the last use
    struct Image {
        vk::Image image;
        vk::DeviceMemory memory;
        vk::ImageView view;
        u32 width = 0, height = 0;
        vk::Format format{};
    };
    Image color, depth, motion, output;
    u32 render_w = 0, render_h = 0, out_w = 0, out_h = 0;
    bool ultra_performance = false;

    void DestroyImage(Image& img) {
        if (img.view) {
            device.destroyImageView(img.view);
        }
        if (img.image) {
            device.destroyImage(img.image);
        }
        if (img.memory) {
            device.freeMemory(img.memory);
        }
        img = {};
    }

    void Destroy() {
        if (!device) {
            return;
        }
        if (!device_idle) {
            (void)device.waitIdle();
        }
        for (Image* img : {&color, &depth, &motion, &output}) {
            DestroyImage(*img);
        }
        device = nullptr;
        device_idle = false;
    }
};

bool MakeImage(vk::PhysicalDevice physical, vk::Device device, Owned::Image& img,
               vk::Format format, u32 w, u32 h, std::string& error) {
    img.image = Check(device.createImage(vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {w, h, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
                 vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc,
    }));
    if (!img.image) {
        error = "image creation failed";
        return false;
    }
    const vk::MemoryRequirements req = device.getImageMemoryRequirements(img.image);
    const vk::PhysicalDeviceMemoryProperties memory = physical.getMemoryProperties();
    u32 type = UINT32_MAX;
    for (u32 i = 0; i < memory.memoryTypeCount; ++i) {
        if ((req.memoryTypeBits & (1u << i)) &&
            (memory.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
            type = i;
            break;
        }
    }
    if (type == UINT32_MAX) {
        error = "no memory type";
        return false;
    }
    img.memory = Check(device.allocateMemory(vk::MemoryAllocateInfo{
        .allocationSize = req.size,
        .memoryTypeIndex = type,
    }));
    if (!img.memory) {
        error = "image memory failed";
        return false;
    }
    Check(device.bindImageMemory(img.image, img.memory, 0));
    vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor;
    if (format == vk::Format::eD32Sfloat) {
        aspect = vk::ImageAspectFlagBits::eDepth;
    }
    img.view = Check(device.createImageView(vk::ImageViewCreateInfo{
        .image = img.image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = {aspect, 0, 1, 0, 1},
    }));
    if (!img.view) {
        error = "image view creation failed";
        return false;
    }
    img.width = w;
    img.height = h;
    img.format = format;
    return true;
}

void Barrier(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout old_layout,
             vk::ImageLayout new_layout, vk::PipelineStageFlags2 src_stage,
             vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst_stage,
             vk::AccessFlags2 dst_access, vk::ImageAspectFlags aspect) {
    const vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .image = image,
        .subresourceRange = {aspect, 0, 1, 0, 1},
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier,
    });
}

/// Barrier `img` from `old_layout` to General for compute reads.
void ToGeneral(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout old_layout,
               vk::AccessFlags2 src_access, vk::ImageAspectFlags aspect) {
    using namespace vk;
    Barrier(cmd, image, old_layout, ImageLayout::eGeneral, PipelineStageFlagBits2::eAllCommands,
            src_access, PipelineStageFlagBits2::eComputeShader, AccessFlagBits2::eShaderRead,
            aspect);
}

/// Copy `src` (in `src_layout`) into `dst` (already in General, same size, colour aspects).
/// Blit when the formats differ, copy when they match; the result is General for compute.
bool CopyInto(vk::CommandBuffer cmd, const Instance& instance, const Resource& src,
              Owned::Image& dst, bool linear, std::string& error) {
    using namespace vk;
    if (src.format == dst.format) {
        Barrier(cmd, src.image, src.layout, ImageLayout::eTransferSrcOptimal,
                PipelineStageFlagBits2::eAllCommands, AccessFlagBits2::eMemoryWrite,
                PipelineStageFlagBits2::eBlit, AccessFlagBits2::eTransferRead,
                ImageAspectFlagBits::eColor);
        Barrier(cmd, dst.image, ImageLayout::eGeneral, ImageLayout::eTransferDstOptimal,
                PipelineStageFlagBits2::eComputeShader, AccessFlagBits2::eShaderRead,
                PipelineStageFlagBits2::eBlit, AccessFlagBits2::eTransferWrite,
                ImageAspectFlagBits::eColor);
        const vk::ImageCopy copy{
            .srcSubresource = {ImageAspectFlagBits::eColor, 0, 0, 1},
            .srcOffset = {0, 0, 0},
            .dstSubresource = {ImageAspectFlagBits::eColor, 0, 0, 1},
            .dstOffset = {0, 0, 0},
            .extent = {dst.width, dst.height, 1},
        };
        cmd.copyImage(src.image, ImageLayout::eTransferSrcOptimal, dst.image,
                      ImageLayout::eTransferDstOptimal, copy);
        Barrier(cmd, dst.image, ImageLayout::eTransferDstOptimal, ImageLayout::eGeneral,
                PipelineStageFlagBits2::eBlit, AccessFlagBits2::eTransferWrite,
                PipelineStageFlagBits2::eComputeShader, AccessFlagBits2::eShaderRead,
                ImageAspectFlagBits::eColor);
        return true;
    }
    if (!instance.IsFormatSupported(src.format,
                                    FormatFeatureFlagBits2::eBlitSrc | FormatFeatureFlagBits2::eSampledImage) ||
        !instance.IsFormatSupported(dst.format, FormatFeatureFlagBits2::eBlitDst |
                                                    FormatFeatureFlagBits2::eStorageImage)) {
        error = "colour conversion blit not supported by the device";
        return false;
    }
    Barrier(cmd, src.image, src.layout, ImageLayout::eTransferSrcOptimal,
            PipelineStageFlagBits2::eAllCommands, AccessFlagBits2::eMemoryWrite,
            PipelineStageFlagBits2::eBlit, AccessFlagBits2::eTransferRead,
            ImageAspectFlagBits::eColor);
    Barrier(cmd, dst.image, ImageLayout::eGeneral, ImageLayout::eTransferDstOptimal,
            PipelineStageFlagBits2::eComputeShader, AccessFlagBits2::eShaderRead,
            PipelineStageFlagBits2::eBlit, AccessFlagBits2::eTransferWrite,
            ImageAspectFlagBits::eColor);
    const vk::ImageBlit blit{
        .srcSubresource = {ImageAspectFlagBits::eColor, 0, 0, 1},
        .srcOffsets = std::array{vk::Offset3D{0, 0, 0},
                                 vk::Offset3D{static_cast<s32>(src.size.width),
                                              static_cast<s32>(src.size.height), 1}},
        .dstSubresource = {ImageAspectFlagBits::eColor, 0, 0, 1},
        .dstOffsets = std::array{vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<s32>(dst.width),
                                                                     static_cast<s32>(dst.height), 1}},
    };
    cmd.blitImage(src.image, ImageLayout::eTransferSrcOptimal, dst.image,
                  ImageLayout::eTransferDstOptimal, blit, linear ? Filter::eLinear : Filter::eNearest);
    Barrier(cmd, dst.image, ImageLayout::eTransferDstOptimal, ImageLayout::eGeneral,
            PipelineStageFlagBits2::eBlit, AccessFlagBits2::eTransferWrite,
            PipelineStageFlagBits2::eComputeShader, AccessFlagBits2::eShaderRead,
            ImageAspectFlagBits::eColor);
    return true;
}

/// Asset directory: SOTC_FSR4_DIR, else $SOTC_DATA_DIR/fsr4_411, else two levels above the
/// running executable (bb-probe lives in <repo>/out/) + /fsr4_411.
std::string AssetDir() {
    if (const char* env = std::getenv("SOTC_FSR4_DIR"); env && env[0]) {
        return env;
    }
    if (const char* data = std::getenv("SOTC_DATA_DIR"); data && data[0]) {
        return std::string(data) + "/fsr4_411";
    }
    char self[4096]{};
    const ssize_t n = ::readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (n > 0) {
        const std::filesystem::path exe(std::string(self, size_t(n)));
        // <repo>/out/bb-probe -> <repo>/out -> <repo>.
        return (exe.parent_path().parent_path() / "fsr4_411").string();
    }
    return "fsr4_411";
}

} // namespace

class Fsr4Backend final : public Backend {
public:
    Kind GetKind() const override {
        return Kind::Fsr4;
    }
    const char* Name() const override {
        return "FSR 4.1.1 INT8";
    }
    bool Init(const Instance& instance) override;
    bool Dispatch(vk::CommandBuffer cmdbuf, const Inputs& in, const Resource& output) override;
    s32 JitterPhaseCount(u32 render_width, u32 output_width) const override;
    void Shutdown() override;

private:
    const Instance* instance{};
    std::unique_ptr<Fsr411::Upscaler> upscaler;
    std::string problem; // last Dispatch failure, for the one stderr line
    Owned owned;
    bool recorded_once = false;
    bool output_first_use = true; // owned.output still Undefined: the replay wrote it as General

    bool CheckDevice(const char* what, std::string& error);
};

bool Fsr4Backend::CheckDevice(const char* what, std::string& error) {
    if (!instance->IsFsr411Supported()) {
        error = std::string(what) +
                " needs shaderFloat16, shaderInt8, shaderInt16, integer dot products and "
                "VK_VALVE_shader_mixed_float_dot_product (RADV: update Mesa)";
        return false;
    }
    return true;
}

bool Fsr4Backend::Init(const Instance& inst) {
    instance = &inst;
    std::string error;
    if (!CheckDevice("FSR 4.1.1", error)) {
        std::fprintf(stderr, "SOTC upscaler: fsr4 unavailable: %s\n", error.c_str());
        return false;
    }
    const std::string dir = AssetDir();
    std::error_code ec;
    if (const bool has_dir = std::filesystem::is_directory(dir, ec); ec || !has_dir) {
        std::fprintf(stderr, "SOTC upscaler: fsr4 unavailable: missing %s (SOTC_FSR4_DIR, or "
                             "$SOTC_DATA_DIR/fsr4_411; tools/fsr4cap/build_assets.sh builds it)\n",
                     dir.c_str());
        return false;
    }
    // No fallback to official FSR 4.1.1: every required set must be present, or Init
    // fails and Active() falls back to Fsr3Builtin.
    static constexpr const char* kPasses[] = {
        "spd",         "prepass",     "pass0_post",  "pass1",       "pass1_post",  "pass2",
        "pass2_post",  "pass3",       "pass3_post",  "pass4",       "pass4_post",  "pass5",
        "pass5_post",  "pass6",       "pass6_post",  "pass7",       "pass7_post",  "pass8",
        "pass8_post",  "pass9",       "pass9_post",  "pass10",      "pass10_post", "pass11",
        "pass11_post", "pass12",      "pass12_post", "postpass",    "rcas",
    }; // keep in sync with kPasses in fsr411/fsr411.cpp
    for (const char* set : {"t1080_m0", "t1080_m1", "t2160_m0", "t2160_m1"}) {
        for (const char* pass : kPasses) {
            std::error_code ec2;
            if (!std::filesystem::is_regular_file(std::filesystem::path(dir) / set /
                                                      (std::string(pass) + ".spv"),
                                                  ec2)) {
                std::fprintf(stderr, "SOTC upscaler: fsr4 unavailable: missing %s/%s/%s.spv\n",
                             dir.c_str(), set, pass);
                return false;
            }
        }
        std::error_code ec3;
        if (!std::filesystem::is_regular_file(std::filesystem::path(dir) / set / "initializer.bin",
                                              ec3)) {
            std::fprintf(stderr, "SOTC upscaler: fsr4 unavailable: missing %s/%s/initializer.bin\n",
                         dir.c_str(), set);
            return false;
        }
    }
    upscaler = std::make_unique<Fsr411::Upscaler>(
        static_cast<VkPhysicalDevice>(inst.GetPhysicalDevice()),
        static_cast<VkDevice>(inst.GetDevice()), dir);
    owned.device = inst.GetDevice();
    // Sizes are validated per frame; Record reports the missing set there.
    return true;
}

bool Fsr4Backend::Dispatch(vk::CommandBuffer cmdbuf, const Inputs& in, const Resource& output) {
    if (!upscaler || !instance) {
        return false;
    }
    const u32 rw = in.render_size.width, rh = in.render_size.height;
    const u32 ow = in.output_size.width, oh = in.output_size.height;
    if (!rw || !rh || !ow || !oh || rw > ow || rh > oh) {
        return false;
    }
    std::string error;
    if (!CheckDevice("FSR 4.1.1", error)) {
        if (problem != error) {
            problem = error;
            std::fprintf(stderr, "SOTC upscaler: fsr4 unavailable: %s\n", error.c_str());
        }
        return false;
    }
    const vk::Device device = instance->GetDevice();
    const vk::PhysicalDevice physical = instance->GetPhysicalDevice();
    // The replay's constant uploads are device-side (vkCmdUpdateBuffer), but its images,
    // pipelines and tensor sizes are recreated when the set or sizes change: the oldest
    // frame must be done first.
    const bool ultra_performance = float(ow) / float(rw) > 2.5f;
    if (!recorded_once || rw != owned.render_w || rh != owned.render_h || ow != owned.out_w ||
        oh != owned.out_h || ultra_performance != owned.ultra_performance) {
        (void)device.waitIdle();
        owned.Destroy();
        owned.device = device;
        owned.device_idle = true;
        recorded_once = true;
        output_first_use = true; // a fresh image: Undefined, not the last blit's TransferSrc
    }
    auto ensure = [&](Owned::Image& img, vk::Format format, u32 w, u32 h, const char* what) {
        if (img.image) {
            return true; // created together after the wait above; sizes match by construction
        }
        if (!MakeImage(physical, device, img, format, w, h, error)) {
            std::fprintf(stderr, "SOTC upscaler: fsr4 unavailable: %s %s\n", what, error.c_str());
            return false;
        }
        return true;
    };
    // Same formats the old game driver fed the replay: RGBA16F colour, D32 depth, RG16F
    // motion, RGBA16F output. Matching caller images are used in place (a barrier to
    // General is enough); the rest is copied into owned intermediates.
    const bool color_direct = in.color.format == vk::Format::eR16G16B16A16Sfloat &&
                              in.color.size.width == rw && in.color.size.height == rh;
    const bool depth_direct = in.depth.size.width == rw && in.depth.size.height == rh;
    const bool motion_direct = in.motion_vectors.format == vk::Format::eR16G16Sfloat &&
                               in.motion_vectors.size.width == rw &&
                               in.motion_vectors.size.height == rh;
    const bool output_direct = output.format == vk::Format::eR16G16B16A16Sfloat &&
                               output.size.width == ow && output.size.height == oh;
    if (!color_direct && !ensure(owned.color, vk::Format::eR16G16B16A16Sfloat, rw, rh, "colour")) {
        return false;
    }
    if (!motion_direct && !ensure(owned.motion, vk::Format::eR16G16Sfloat, rw, rh, "motion")) {
        return false;
    }
    // The replay writes its output image; only an owned image may be handed to it.
    if (!ensure(owned.output, vk::Format::eR16G16B16A16Sfloat, ow, oh, "output")) {
        return false;
    }
    owned.render_w = rw;
    owned.render_h = rh;
    owned.out_w = ow;
    owned.out_h = oh;
    owned.ultra_performance = ultra_performance;
    owned.device_idle = false;

    if (color_direct) {
        ToGeneral(cmdbuf, in.color.image, in.color.layout, vk::AccessFlagBits2::eMemoryWrite,
                  vk::ImageAspectFlagBits::eColor);
    } else if (!CopyInto(cmdbuf, *instance, in.color, owned.color, true, error)) {
        if (!error.empty() && problem != error) {
            problem = error;
            std::fprintf(stderr, "SOTC upscaler: fsr4 unavailable: %s\n", error.c_str());
        }
        return false;
    }
    // Depth passes straight through with the caller's depth-aspect view (as the old game
    // driver did); a depth blit would be invalid usage (depth formats must match).
    if (!depth_direct) {
        error = "depth must be the render size";
        if (problem != error) {
            problem = error;
            std::fprintf(stderr, "SOTC upscaler: fsr4 unavailable: %s\n", error.c_str());
        }
        return false;
    }
    ToGeneral(cmdbuf, in.depth.image, in.depth.layout, vk::AccessFlagBits2::eMemoryWrite,
              vk::ImageAspectFlagBits::eDepth);
    if (motion_direct) {
        ToGeneral(cmdbuf, in.motion_vectors.image, in.motion_vectors.layout,
                  vk::AccessFlagBits2::eMemoryWrite, vk::ImageAspectFlagBits::eColor);
    } else if (!CopyInto(cmdbuf, *instance, in.motion_vectors, owned.motion, false, error)) {
        if (!error.empty() && problem != error) {
            problem = error;
            std::fprintf(stderr, "SOTC upscaler: fsr4 unavailable: %s\n", error.c_str());
        }
        return false;
    }

    const vk::Image frame_color = color_direct ? in.color.image : owned.color.image;
    const vk::ImageView frame_color_view = color_direct ? in.color.view : owned.color.view;
    const vk::Image frame_motion = motion_direct ? in.motion_vectors.image : owned.motion.image;
    const vk::ImageView frame_motion_view = motion_direct ? in.motion_vectors.view : owned.motion.view;
    Fsr411::Frame frame{};
    frame.cmdbuf = static_cast<VkCommandBuffer>(cmdbuf);
    frame.color = {static_cast<VkImage>(frame_color), static_cast<VkImageView>(frame_color_view),
                   rw, rh};
    frame.depth = {static_cast<VkImage>(in.depth.image), static_cast<VkImageView>(in.depth.view),
                   rw, rh};
    frame.motion = {static_cast<VkImage>(frame_motion),
                    static_cast<VkImageView>(frame_motion_view), rw, rh};
    frame.output = {static_cast<VkImage>(owned.output.image),
                    static_cast<VkImageView>(owned.output.view), ow, oh};
    frame.render_width = rw;
    frame.render_height = rh;
    // Two models, chosen by ratio as the old game driver did: 3.0 (ultra performance) or not.
    frame.ultra_performance = ultra_performance;
    frame.jitter[0] = in.jitter_x;
    frame.jitter[1] = in.jitter_y;
    // Vectors are in render pixels; the replay divides by the render size. A zero scale is
    // an unset field, not disabled motion: the old driver always passed 1.
    frame.motion_scale[0] = in.mv_scale_x != 0.0f ? in.mv_scale_x : 1.0f;
    frame.motion_scale[1] = in.mv_scale_y != 0.0f ? in.mv_scale_y : 1.0f;
    frame.pre_exposure = in.pre_exposure > 0.0f ? in.pre_exposure : 1.0f;
    frame.sharpness = std::clamp(in.sharpness, 0.0f, 1.0f);
    frame.sharpen = in.sharpness >= 0.0f;
    frame.reset = in.reset;
    frame.auto_exposure = !in.exposure.image;
    if (!upscaler->Record(frame)) {
        const std::string err = upscaler->Error();
        if (!err.empty() && problem != err) {
            problem = err;
            std::fprintf(stderr, "SOTC upscaler: fsr4 unavailable: %s\n", err.c_str());
        }
        return false;
    }
    problem.clear();

    // To the caller's output in ShaderReadOnlyOptimal: same-size blit from the owned image
    // the replay wrote. The owned image is a replay frame image (GENERAL there); the first
    // dispatch after creation transitions it out of Undefined, afterwards out of the
    // TransferSrcOptimal the previous blit left it in.
    using namespace vk;
    Barrier(cmdbuf, owned.output.image,
            output_first_use ? ImageLayout::eUndefined : ImageLayout::eGeneral,
            ImageLayout::eTransferSrcOptimal, PipelineStageFlagBits2::eAllCommands,
            AccessFlagBits2::eNone, PipelineStageFlagBits2::eBlit,
            AccessFlagBits2::eTransferRead, ImageAspectFlagBits::eColor);
    Barrier(cmdbuf, output.image, output.layout, ImageLayout::eTransferDstOptimal,
            PipelineStageFlagBits2::eAllCommands, AccessFlagBits2::eMemoryWrite,
            PipelineStageFlagBits2::eBlit, AccessFlagBits2::eTransferWrite,
            ImageAspectFlagBits::eColor);
    const vk::ImageBlit blit{
        .srcSubresource = {ImageAspectFlagBits::eColor, 0, 0, 1},
        .srcOffsets = std::array{vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<s32>(ow),
                                                                     static_cast<s32>(oh), 1}},
        .dstSubresource = {ImageAspectFlagBits::eColor, 0, 0, 1},
        .dstOffsets = std::array{vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<s32>(ow),
                                                                     static_cast<s32>(oh), 1}},
    };
    cmdbuf.blitImage(owned.output.image, ImageLayout::eTransferSrcOptimal, output.image,
                     ImageLayout::eTransferDstOptimal, blit, Filter::eNearest);
    Barrier(cmdbuf, owned.output.image, ImageLayout::eTransferSrcOptimal, ImageLayout::eGeneral,
            PipelineStageFlagBits2::eBlit, AccessFlagBits2::eTransferRead,
            PipelineStageFlagBits2::eComputeShader,
            AccessFlagBits2::eShaderRead | AccessFlagBits2::eShaderWrite,
            ImageAspectFlagBits::eColor);
    Barrier(cmdbuf, output.image, ImageLayout::eTransferDstOptimal,
            ImageLayout::eShaderReadOnlyOptimal, PipelineStageFlagBits2::eBlit,
            AccessFlagBits2::eTransferWrite, PipelineStageFlagBits2::eFragmentShader,
            AccessFlagBits2::eShaderRead, ImageAspectFlagBits::eColor);
    // The replay's images stay in use by this frame; the caller owns the command buffer's
    // lifetime (its own in-flight frames), as with the old constant ring.
    output_first_use = false;
    return true;
}

s32 Fsr4Backend::JitterPhaseCount(u32 render_width, u32 output_width) const {
    if (!render_width || !output_width) {
        return 8;
    }
    // FSR's phases, as the old game driver computed them: ceil(8 * (output/render)^2).
    const double ratio = double(output_width) / double(render_width);
    return s32(std::clamp(std::ceil(8.0 * ratio * ratio), 8.0, 256.0));
}

void Fsr4Backend::Shutdown() {
    upscaler.reset();
    owned.Destroy();
    instance = nullptr;
    problem.clear();
}

std::unique_ptr<Backend> MakeFsr4() {
    return std::make_unique<Fsr4Backend>();
}

} // namespace Vulkan::Upscaler
