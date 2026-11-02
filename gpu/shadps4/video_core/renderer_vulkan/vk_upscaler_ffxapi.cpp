// SPDX-License-Identifier: GPL-2.0-or-later
// sotc: the FFX API backend (amd_fidelityfx_vk.dll, hooked by OptiScaler).
//
// The DLL is loaded at run time (SOTC_FFX_API_DLL, else amd_fidelityfx_vk.dll on Windows /
// libamd_fidelityfx_vk.so on POSIX); its ffxCreateContext/ffxDestroyContext/ffxConfigure/
// ffxQuery/ffxDispatch are resolved by name, never linked. Context: ffxCreateContextDescUpscale
// under an ffxCreateBackendVKDesc holding the Instance's VkDevice/VkPhysicalDevice and its
// vkGetDeviceProcAddr. Dispatch fills ffxDispatchDescUpscale from Inputs and records into the
// caller's command buffer (an entry stays outstanding until the next one: no queue submission
// of our own). The contract guarantees SHADER_READ_ONLY_OPTIMAL inputs, but anything readable
// passes through and anything else is moved there first; FSR 3.1 mips come from its own
// SOTC_MIP_BIAS knob, so no format copies here. Output leaves SHADER_READ_ONLY_OPTIMAL.

#include "video_core/renderer_vulkan/vk_upscaler.h"

#ifndef _WIN32
#include <dlfcn.h>
#else
#include <windows.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// The SDK marks its entry points __declspec(dllexport): only names here (resolved at run
// time), and not syntax GCC outside Windows accepts.
#if !defined(_WIN32) && !defined(__declspec)
#define __declspec(x)
#define SOTC_DEFINED_DECLSPEC
#endif
#include "ffx_api/ffx_api.h"
#include "ffx_api/ffx_api_types.h"
#include "ffx_api/ffx_upscale.h"
#include "ffx_api/vk/ffx_api_vk.h"
#ifdef SOTC_DEFINED_DECLSPEC
#undef __declspec
#undef SOTC_DEFINED_DECLSPEC
#endif

#include "video_core/renderer_vulkan/vk_instance.h"

namespace Vulkan::Upscaler {

namespace {

vk::ImageAspectFlags AspectOf(vk::Format format) {
    switch (format) {
    case vk::Format::eD16Unorm:
    case vk::Format::eX8D24UnormPack32:
    case vk::Format::eD32Sfloat:
        return vk::ImageAspectFlagBits::eDepth;
    case vk::Format::eS8Uint:
        return vk::ImageAspectFlagBits::eStencil;
    case vk::Format::eD16UnormS8Uint:
    case vk::Format::eD24UnormS8Uint:
    case vk::Format::eD32SfloatS8Uint:
        return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
    default:
        return vk::ImageAspectFlagBits::eColor;
    }
}

struct Api {
#ifdef _WIN32
    HMODULE dll{};
#else
    void* dll{};
#endif
    PfnFfxCreateContext create{};
    PfnFfxDestroyContext destroy{};
    PfnFfxConfigure configure{};
    PfnFfxQuery query{};
    PfnFfxDispatch dispatch{};

    bool Load(const char* path, bool explicit_path) {
#ifdef _WIN32
        dll = explicit_path ? LoadLibraryExA(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH)
                            : LoadLibraryW(L"amd_fidelityfx_vk.dll");
#else
        dll = dlopen(explicit_path ? path : "libamd_fidelityfx_vk.so", RTLD_NOW | RTLD_LOCAL);
#endif
        if (!dll) {
#ifdef _WIN32
            std::fprintf(stderr, "SOTC upscaler: FFX API DLL not loaded: %s (error %lu)\n",
                         path, GetLastError());
#else
            std::fprintf(stderr, "SOTC upscaler: FFX API library not loaded: %s (%s)\n", path,
                         dlerror());
#endif
            return false;
        }
        auto resolve = [&](auto& out, const char* name) {
#ifdef _WIN32
            out = reinterpret_cast<std::decay_t<decltype(out)>>(GetProcAddress(dll, name));
#else
            out = reinterpret_cast<std::decay_t<decltype(out)>>(dlsym(dll, name));
#endif
            if (!out) {
                std::fprintf(stderr, "SOTC upscaler: %s missing in %s\n", name, path);
                return false;
            }
            return true;
        };
        return resolve(create, "ffxCreateContext") && resolve(destroy, "ffxDestroyContext") &&
               resolve(configure, "ffxConfigure") && resolve(query, "ffxQuery") &&
               resolve(dispatch, "ffxDispatch");
    }

    void Unload() {
        if (!dll) {
            return;
        }
#ifdef _WIN32
        FreeLibrary(dll);
#else
        dlclose(dll);
#endif
        dll = {};
        create = {};
        destroy = {};
        configure = {};
        query = {};
        dispatch = {};
    }
};

FfxApiResource ToResource(const Resource& r, uint32_t usage, uint32_t state) {
    const VkFormat format = static_cast<VkFormat>(r.format);
    FfxApiResourceDescription desc{};
    desc.usage = ffxApiIsDepthFormat(format) ? FFX_API_RESOURCE_USAGE_DEPTHTARGET
                                             : FFX_API_RESOURCE_USAGE_READ_ONLY;
    desc.usage |= usage;
    desc.format = ffxApiGetSurfaceFormatVK(format);
    desc.width = r.size.width;
    desc.height = r.size.height;
    desc.depth = 1;
    desc.mipCount = 1;
    desc.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    FfxApiResource out{};
    out.resource = reinterpret_cast<void*>(static_cast<VkImage>(r.image));
    out.description = desc;
    out.state = state;
    return out;
}

// SHADER_READ_ONLY_OPTIMAL and GENERAL pass through as COMPUTE_READ (the state the SDK maps
// to those layouts); e.g. an attachment-optimal input is moved to SHADER_READ_ONLY_OPTIMAL
// on the same command buffer first, so COMPUTE_READ is what the dispatch then sees.
constexpr bool Readable(vk::ImageLayout layout) {
    return layout == vk::ImageLayout::eShaderReadOnlyOptimal || layout == vk::ImageLayout::eGeneral;
}

vk::ImageMemoryBarrier ToShaderRead(const Resource& r) {
    vk::AccessFlags src = vk::AccessFlagBits::eMemoryWrite;
    vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands;
    if (r.layout == vk::ImageLayout::eColorAttachmentOptimal) {
        src = vk::AccessFlagBits::eColorAttachmentWrite;
        stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    } else if (r.layout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
        src = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
        stage = vk::PipelineStageFlagBits::eLateFragmentTests;
    } else if (r.layout == vk::ImageLayout::eTransferDstOptimal) {
        src = vk::AccessFlagBits::eTransferWrite;
        stage = vk::PipelineStageFlagBits::eTransfer;
    } else if (r.layout == vk::ImageLayout::eTransferSrcOptimal) {
        src = vk::AccessFlagBits::eTransferRead;
        stage = vk::PipelineStageFlagBits::eTransfer;
    }
    (void)stage;
    return vk::ImageMemoryBarrier{
        .srcAccessMask = src,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
        .oldLayout = r.layout,
        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = r.image,
        .subresourceRange =
            {
                .aspectMask = AspectOf(r.format),
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
    };
}

class FfxApiBackend : public Backend {
public:
    Kind GetKind() const override {
        return Kind::FfxApiDll;
    }

    const char* Name() const override {
        return "FSR 3.1 (FFX API DLL)";
    }

    bool Init(const Instance& instance) override {
        if (inited_) {
            return true;
        }
        const char* env = std::getenv("SOTC_FFX_API_DLL");
#ifdef _WIN32
        const std::string path = env && *env ? env : "amd_fidelityfx_vk.dll";
#else
        const std::string path = env && *env ? env : "libamd_fidelityfx_vk.so";
#endif
        if (!api_.Load(path.c_str(), env && *env)) {
            return false;
        }
        backend_ = {
            .header = {FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK, nullptr},
            .vkDevice = static_cast<VkDevice>(instance.GetDevice()),
            .vkPhysicalDevice = static_cast<VkPhysicalDevice>(instance.GetPhysicalDevice()),
            .vkDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr,
        };
        inited_ = true; // the context is created at the first dispatch's sizes.
        return true;
    }

    bool Dispatch(vk::CommandBuffer cmdbuf, const Inputs& in, const Resource& output) override {
        if (!inited_) {
            return false;
        }
        if (in.color.image == vk::Image{} || in.depth.image == vk::Image{} ||
            in.motion_vectors.image == vk::Image{} || output.image == vk::Image{}) {
            std::fprintf(stderr,
                         "SOTC upscaler: FFX API dispatch without color/depth/motion/output\n");
            return false;
        }
        if (!Recreate(in)) {
            return false;
        }
        std::vector<vk::ImageMemoryBarrier> transitions;
        FfxApiResource color =
            ToResource(in.color, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        FfxApiResource depth =
            ToResource(in.depth, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        FfxApiResource motion = ToResource(in.motion_vectors, FFX_API_RESOURCE_USAGE_READ_ONLY,
                                           FFX_API_RESOURCE_STATE_COMPUTE_READ);
        for (const Resource* r : {&in.color, &in.depth, &in.motion_vectors}) {
            if (r->image != vk::Image{} && !Readable(r->layout)) {
                transitions.push_back(ToShaderRead(*r));
            }
        }
        FfxApiResource reactive{};
        if (in.reactive.image != vk::Image{}) {
            reactive = ToResource(in.reactive, FFX_API_RESOURCE_USAGE_READ_ONLY,
                                  FFX_API_RESOURCE_STATE_COMPUTE_READ);
            if (!Readable(in.reactive.layout)) {
                transitions.push_back(ToShaderRead(in.reactive));
            }
        }
        FfxApiResource composition{};
        if (in.transparency_composition.image != vk::Image{}) {
            composition = ToResource(in.transparency_composition, FFX_API_RESOURCE_USAGE_READ_ONLY,
                                     FFX_API_RESOURCE_STATE_COMPUTE_READ);
            if (!Readable(in.transparency_composition.layout)) {
                transitions.push_back(ToShaderRead(in.transparency_composition));
            }
        }
        FfxApiResource exposure{};
        if (in.exposure.image != vk::Image{}) {
            exposure = ToResource(in.exposure, FFX_API_RESOURCE_USAGE_READ_ONLY,
                                  FFX_API_RESOURCE_STATE_COMPUTE_READ);
            if (!Readable(in.exposure.layout)) {
                transitions.push_back(ToShaderRead(in.exposure));
            }
        }
        // UAV output: GENERAL passes through, anything else is moved there first, and it
        // leaves SHADER_READ_ONLY_OPTIMAL (the presenter's HUD pass samples it next).
        if (output.layout != vk::ImageLayout::eGeneral) {
            transitions.push_back(vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
                .dstAccessMask = vk::AccessFlagBits::eShaderWrite,
                .oldLayout = output.layout,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = output.image,
                .subresourceRange =
                    {
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
            });
        }
        if (!transitions.empty()) {
            cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                   vk::PipelineStageFlagBits::eComputeShader,
                                   vk::DependencyFlagBits::eByRegion, {}, {}, transitions);
        }
        FfxApiResource out =
            ToResource(output, FFX_API_RESOURCE_USAGE_UAV, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        ffxDispatchDescUpscale desc{};
        desc.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        desc.commandList = static_cast<VkCommandBuffer>(cmdbuf);
        desc.color = color;
        desc.depth = depth;
        desc.motionVectors = motion;
        desc.exposure = exposure;
        desc.reactive = reactive;
        desc.transparencyAndComposition = composition;
        desc.output = out;
        // Subpixel pixels, as the presenter feeds them: the core normalises by the motion
        // vector target size itself (ffx_fsr3upscaler.cpp motionVectorJitterCancellation).
        desc.jitterOffset = {in.jitter_x, in.jitter_y};
        desc.motionVectorScale = {in.mv_scale_x, in.mv_scale_y};
        desc.renderSize = {in.render_size.width, in.render_size.height};
        desc.upscaleSize = {in.output_size.width, in.output_size.height};
        desc.enableSharpening = in.sharpness >= 0.f;
        desc.sharpness = std::clamp(in.sharpness, 0.f, 1.f);
        desc.frameTimeDelta = in.frame_time_ms;
        desc.preExposure = in.pre_exposure != 0.f ? in.pre_exposure : 1.f;
        desc.reset = in.reset;
        desc.cameraNear = in.camera_near;
        desc.cameraFar = in.camera_far;
        desc.cameraFovAngleVertical = in.camera_fov_y;
        desc.viewSpaceToMetersFactor = 1.f;
        if (const ffxReturnCode_t rc = api_.dispatch(&context_, &desc.header);
            rc != FFX_API_RETURN_OK) {
            std::fprintf(stderr, "SOTC upscaler: ffxDispatch failed (%u)\n", rc);
            return false;
        }
        const vk::ImageMemoryBarrier output_barrier{
            .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = output.image,
            .subresourceRange =
                {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
        };
        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                               vk::PipelineStageFlagBits::eFragmentShader |
                                   vk::PipelineStageFlagBits::eComputeShader,
                               vk::DependencyFlagBits::eByRegion, {}, {}, output_barrier);
        return true;
    }

    s32 JitterPhaseCount(u32 render_width, u32 output_width) const override {
        if (!inited_ || !context_) {
            return 0;
        }
        ffxQueryDescUpscaleGetJitterPhaseCount query{};
        query.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT;
        int32_t phases = 0;
        query.renderWidth = render_width;
        query.displayWidth = output_width;
        query.pOutPhaseCount = &phases;
        ffxContext context = context_; // by handle: the query only reads through it.
        if (api_.query(&context, &query.header) != FFX_API_RETURN_OK || phases <= 0) {
            return 0;
        }
        return static_cast<s32>(phases);
    }

    void Shutdown() override {
        if (context_) {
            api_.destroy(&context_, nullptr);
            context_ = nullptr;
        }
        api_.Unload();
        render_size_ = vk::Extent2D{0, 0};
        output_size_ = vk::Extent2D{0, 0};
        inited_ = false;
    }

private:
    bool Recreate(const Inputs& in) {
        const bool size_changed = in.render_size.width != render_size_.width ||
                                  in.render_size.height != render_size_.height ||
                                  in.output_size.width != output_size_.width ||
                                  in.output_size.height != output_size_.height;
        const uint32_t want_flags = CreateFlags(in);
        if (context_ && !size_changed && want_flags == flags_) {
            return true;
        }
        if (context_) {
            api_.destroy(&context_, nullptr);
            context_ = nullptr;
        }
        render_size_ = in.render_size;
        output_size_ = in.output_size;
        flags_ = want_flags;
        ffxCreateContextDescUpscale create{};
        create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        create.header.pNext = reinterpret_cast<ffxApiHeader*>(&backend_);
        create.flags = flags_;
        create.maxRenderSize = {std::max(render_size_.width, 1u), std::max(render_size_.height, 1u)};
        create.maxUpscaleSize = {std::max(output_size_.width, 1u), std::max(output_size_.height, 1u)};
        if (const ffxReturnCode_t rc = api_.create(&context_, &create.header, nullptr);
            rc != FFX_API_RETURN_OK) {
            std::fprintf(stderr, "SOTC upscaler: ffxCreateContext failed (%u)\n", rc);
            context_ = nullptr;
            return false;
        }
        return true;
    }

    static uint32_t CreateFlags(const Inputs& in) {
        uint32_t flags = in.hdr ? FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE : 0;
        if (in.depth_inverted) {
            flags |= FFX_UPSCALE_ENABLE_DEPTH_INVERTED;
        }
        if (in.depth_infinite) {
            flags |= FFX_UPSCALE_ENABLE_DEPTH_INFINITE;
        }
        // The SDK warns with both a manual exposure and AUTO_EXPOSURE (and ignores the
        // texture): automatic only when the presenter sends none.
        if (in.exposure.image == vk::Image{}) {
            flags |= FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
        }
        return flags;
    }

    Api api_{};
    ffxCreateBackendVKDesc backend_{};
    ffxContext context_{};
    vk::Extent2D render_size_{}, output_size_{};
    uint32_t flags_{};
    bool inited_{};
};

} // namespace

std::unique_ptr<Backend> MakeFfxApiDll() {
    return std::make_unique<FfxApiBackend>();
}

} // namespace Vulkan::Upscaler
