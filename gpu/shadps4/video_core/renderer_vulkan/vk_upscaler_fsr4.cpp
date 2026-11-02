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

/// Convert or copy `src` (in `src.layout`) into `dst` (already in General, render size); the
/// result is General for compute. Both images are owned by this backend; the game's input
/// attachments are only read.
void ConvertInto(vk::CommandBuffer cmd, const Resource& src, Owned::Image& dst,
                 vk::Format dst_format, bool linear) {
    using namespace vk;
    if (src.format == dst_format && src.size.width == dst.width && src.size.height == dst.height) {
        // Same pixels, maybe another layout: a barrier is enough.
        Barrier(cmd, src.image, src.layout, ImageLayout::eGeneral, PipelineStageFlagBits2::eAllCommands,
                AccessFlagBits2::eMemoryWrite, PipelineStageFlagBits2::eComputeShader,
                AccessFlagBits2::eShaderRead, ImageAspectFlagBits::eColor);
        return;
    }
    // Different format or size: blit (depth is compared separately below; this is colour).
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
    // The replay owns its intermediates (created on size change); idle the device so that no
    // frame still reads them. In-flight frames only ever touch the fsr411 object itself.
    if ((rw != owned.render_w || rh != owned.render_h || ow != owned.out_w ||
         oh != owned.out_h) &&
        (owned.color.image || owned.depth.image)) {
        (void)device.waitIdle();
        owned.device_idle = true;
        owned.Destroy();
        owned.device = device;
        owned.device_idle = true;
    }
    auto ensure = [&](Owned::Image& img, vk::Format format, u32 w, u32 h, const char* what) {
        if (img.image && img.width == w && img.height == h && img.format == format) {
            return true;
        }
        if (img.image) {
            (void)device.waitIdle();
            owned.device_idle = true;
            owned.DestroyImage(img);
        }
        if (!MakeImage(physical, device, img, format, w, h, error)) {
            std::fprintf(stderr, "SOTC upscaler: fsr4 unavailable: %s %s\n", what, error.c_str());
            return false;
        }
        return true;
    };
    // Same formats the old game driver fed the replay: RGBA16F colour, D32 depth, RG16F
    // motion, RGBA16F output.
    if (!ensure(owned.color, vk::Format::eR16G16B16A16Sfloat, rw, rh, "colour")) {
        return false;
    }
    if (!ensure(owned.depth, vk::Format::eD32Sfloat, rw, rh, "depth")) {
        return false;
    }
    if (!ensure(owned.motion, vk::Format::eR16G16Sfloat, rw, rh, "motion")) {
        return false;
    }
    if (!ensure(owned.output, vk::Format::eR16G16B16A16Sfloat, ow, oh, "output")) {
        return false;
    }
    owned.render_w = rw;
    owned.render_h = rh;
    owned.out_w = ow;
    owned.out_h = oh;
    owned.device_idle = false;

    // Convert or copy the inputs into the replay's formats (each becomes General). Depth is
    // compared by format, not aspect: the game may hand it in a depth layout.
    ConvertInto(cmdbuf, in.color, owned.color, vk::Format::eR16G16B16A16Sfloat, true);
    {
        using namespace vk;
        if (in.depth.format == Format::eD32Sfloat && in.depth.size.width == rw &&
            in.depth.size.height == rh) {
            Barrier(cmdbuf, in.depth.image, in.depth.layout, ImageLayout::eGeneral,
                    PipelineStageFlagBits2::eAllCommands, AccessFlagBits2::eMemoryWrite,
                    PipelineStageFlagBits2::eComputeShader, AccessFlagBits2::eShaderRead,
                    ImageAspectFlagBits::eDepth);
        } else {
            // Non-D32 depth: copy through a blit into the owned D32 image.
            Barrier(cmdbuf, in.depth.image, in.depth.layout, ImageLayout::eTransferSrcOptimal,
                    PipelineStageFlagBits2::eAllCommands, AccessFlagBits2::eMemoryWrite,
                    PipelineStageFlagBits2::eBlit, AccessFlagBits2::eTransferRead,
                    ImageAspectFlagBits::eDepth);
            Barrier(cmdbuf, owned.depth.image, ImageLayout::eUndefined,
                    ImageLayout::eTransferDstOptimal, PipelineStageFlagBits2::eComputeShader,
                    AccessFlagBits2::eNone, PipelineStageFlagBits2::eBlit,
                    AccessFlagBits2::eTransferWrite, ImageAspectFlagBits::eDepth);
            const vk::ImageBlit blit{
                .srcSubresource = {ImageAspectFlagBits::eDepth, 0, 0, 1},
                .srcOffsets = std::array{vk::Offset3D{0, 0, 0},
                                         vk::Offset3D{static_cast<s32>(in.depth.size.width),
                                                      static_cast<s32>(in.depth.size.height), 1}},
                .dstSubresource = {ImageAspectFlagBits::eDepth, 0, 0, 1},
                .dstOffsets = std::array{vk::Offset3D{0, 0, 0},
                                         vk::Offset3D{static_cast<s32>(rw), static_cast<s32>(rh), 1}},
            };
            cmdbuf.blitImage(in.depth.image, ImageLayout::eTransferSrcOptimal, owned.depth.image,
                             ImageLayout::eTransferDstOptimal, blit, Filter::eNearest);
            Barrier(cmdbuf, owned.depth.image, ImageLayout::eTransferDstOptimal,
                    ImageLayout::eGeneral, PipelineStageFlagBits2::eBlit,
                    AccessFlagBits2::eTransferWrite, PipelineStageFlagBits2::eComputeShader,
                    AccessFlagBits2::eShaderRead, ImageAspectFlagBits::eDepth);
        }
    }
    ConvertInto(cmdbuf, in.motion_vectors, owned.motion, vk::Format::eR16G16Sfloat, false);

    Fsr411::Frame frame{};
    frame.cmdbuf = static_cast<VkCommandBuffer>(cmdbuf);
    frame.color = {static_cast<VkImage>(owned.color.image), static_cast<VkImageView>(owned.color.view),
                   rw, rh};
    frame.depth = {static_cast<VkImage>(owned.depth.image), static_cast<VkImageView>(owned.depth.view),
                   rw, rh};
    frame.motion = {static_cast<VkImage>(owned.motion.image),
                    static_cast<VkImageView>(owned.motion.view), rw, rh};
    frame.output = {static_cast<VkImage>(owned.output.image),
                    static_cast<VkImageView>(owned.output.view), ow, oh};
    frame.render_width = rw;
    frame.render_height = rh;
    // Two models, chosen by ratio as the old game driver did: 3.0 (ultra performance) or not.
    frame.ultra_performance = float(ow) / float(rw) > 2.5f;
    frame.jitter[0] = in.jitter_x;
    frame.jitter[1] = in.jitter_y;
    // Vectors are in render pixels; the replay divides by the render size.
    frame.motion_scale[0] = in.mv_scale_x;
    frame.motion_scale[1] = in.mv_scale_y;
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

    // To the caller's output in ShaderReadOnlyOptimal. The replay's output is always exactly
    // the output size, so this is a same-size blit.
    using namespace vk;
    Barrier(cmdbuf, owned.output.image, ImageLayout::eGeneral, ImageLayout::eTransferSrcOptimal,
            PipelineStageFlagBits2::eComputeShader, AccessFlagBits2::eShaderWrite,
            PipelineStageFlagBits2::eBlit, AccessFlagBits2::eTransferRead,
            ImageAspectFlagBits::eColor);
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
    Barrier(cmdbuf, output.image, ImageLayout::eTransferDstOptimal,
            ImageLayout::eShaderReadOnlyOptimal, PipelineStageFlagBits2::eBlit,
            AccessFlagBits2::eTransferWrite, PipelineStageFlagBits2::eFragmentShader,
            AccessFlagBits2::eShaderRead, ImageAspectFlagBits::eColor);
    // The replay's images stay in use by this frame; the caller owns the command buffer's
    // lifetime (its own in-flight frames), as with the old constant ring.
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
