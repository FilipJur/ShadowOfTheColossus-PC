// SPDX-License-Identifier: GPL-2.0-or-later
// sotc: stub for the built-in FSR 3.1 backend; the backend worker replaces this file.

#include "video_core/renderer_vulkan/vk_upscaler.h"

namespace Vulkan::Upscaler {

std::unique_ptr<Backend> MakeFsr3Builtin() {
    return nullptr;
}

} // namespace Vulkan::Upscaler
