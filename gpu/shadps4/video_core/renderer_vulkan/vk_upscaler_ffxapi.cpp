// SPDX-License-Identifier: GPL-2.0-or-later
// sotc: stub for the FFX API backend (amd_fidelityfx_vk.dll, hooked by OptiScaler);
// the backend worker replaces this file.

#include "video_core/renderer_vulkan/vk_upscaler.h"

namespace Vulkan::Upscaler {

std::unique_ptr<Backend> MakeFfxApiDll() {
    return nullptr;
}

} // namespace Vulkan::Upscaler
