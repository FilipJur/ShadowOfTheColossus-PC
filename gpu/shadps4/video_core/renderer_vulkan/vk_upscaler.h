// SPDX-License-Identifier: GPL-2.0-or-later
// sotc: shared contract for the temporal upscaler backends: FSR 3.1 built in
// (vk_upscaler_fsr3.cpp), or AMD's FFX API through a dynamically loaded
// amd_fidelityfx_vk.dll, which OptiScaler hooks on Windows/Proton (vk_upscaler_ffxapi.cpp).
// The presenter captures the game's colour/depth/motion vectors once; the active backend
// consumes them (everything OptiScaler's FFX/NGX translator needs is in Inputs).

#pragma once

#include <memory>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {
class Instance;
}

namespace Vulkan::Upscaler {

enum class Kind { None, Fsr3Builtin, FfxApiDll };

struct Resource {
    vk::Image image;
    vk::ImageView view;
    vk::Format format;
    vk::Extent2D size;
    vk::ImageLayout layout;
};

struct Inputs {
    // everything OptiScaler's FFX/NGX translator needs
    Resource color, depth, motion_vectors;
    Resource reactive; // optional (null image = none)
    Resource transparency_composition; // optional
    Resource exposure; // optional 1x1; null => backend auto-exposure
    vk::Extent2D render_size, output_size;
    float jitter_x, jitter_y; // pixels, render space
    float mv_scale_x, mv_scale_y; // multiply MV to get render-space pixels
    float camera_near, camera_far, camera_fov_y; // fov in radians
    float frame_time_ms, pre_exposure;
    float sharpness; // 0..1, <0 = disabled
    bool reset, hdr, depth_inverted, depth_infinite;
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual Kind GetKind() const = 0;
    virtual const char* Name() const = 0; // e.g. "FSR 3.1 (built-in)"
    virtual bool Init(const Instance& instance) = 0; // false = unavailable on this device/host
    virtual bool Dispatch(vk::CommandBuffer cmdbuf, const Inputs& in, const Resource& output) = 0;
    virtual i32 JitterPhaseCount(u32 render_width, u32 output_width) const = 0;
    virtual void Shutdown() = 0;
};

std::unique_ptr<Backend> MakeFsr3Builtin(); // defined in vk_upscaler_fsr3.cpp
std::unique_ptr<Backend> MakeFfxApiDll(); // defined in vk_upscaler_ffxapi.cpp
Kind Requested(); // SOTC_UPSCALER: "fsr3" -> Fsr3Builtin, "ffxapi" -> FfxApiDll, else None
// The device-creation hook (call beside Dlssg::OnDeviceCreated in vk_instance.cpp): Active
// needs the Instance for Backend::Init but takes no arguments, so it is delivered here.
void OnDeviceCreated(const Instance& instance);
Backend* Active(); // created+Init'ed lazily on first call; if FfxApiDll Init fails, fall back
                   // to Fsr3Builtin and log one line; nullptr when None
void Shutdown();

} // namespace Vulkan::Upscaler
