// SPDX-License-Identifier: GPL-2.0-or-later
// sotc: shared temporal-upscaler factory, see vk_upscaler.h.

#include "video_core/renderer_vulkan/vk_upscaler.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace Vulkan::Upscaler {

namespace {

const Instance* g_instance{};
std::unique_ptr<Backend> g_backend;
bool g_tried{};
std::mutex g_mutex;

} // namespace

Kind Requested() {
    const char* u = std::getenv("SOTC_UPSCALER");
    if (u && std::strcmp(u, "fsr3") == 0) {
        return Kind::Fsr3Builtin;
    }
    if (u && std::strcmp(u, "ffxapi") == 0) {
        return Kind::FfxApiDll;
    }
    if (u && std::strcmp(u, "fsr4") == 0) {
        return Kind::Fsr4;
    }
    return Kind::None;
}

void OnDeviceCreated(const Instance& instance) {
    std::lock_guard lock(g_mutex);
    g_instance = &instance;
}

Backend* Active() {
    std::lock_guard lock(g_mutex);
    if (g_tried) {
        return g_backend.get();
    }
    g_tried = true;
    const Kind kind = Requested();
    if (kind == Kind::None || !g_instance) {
        return nullptr;
    }
    std::unique_ptr<Backend> first = kind == Kind::FfxApiDll  ? MakeFfxApiDll()
                                     : kind == Kind::Fsr4    ? MakeFsr4()
                                                             : MakeFsr3Builtin();
    if (first && first->Init(*g_instance)) {
        std::fprintf(stderr, "SOTC upscaler: %s\n", first->Name());
        g_backend = std::move(first);
        return g_backend.get();
    }
    // FfxApiDll unavailable (no amd_fidelityfx_vk.dll on this host), or Fsr4 unavailable
    // (no fsr4_411 assets, or the GPU lacks INT8 dot products): fall back to the built-in
    // FSR 3.1, one line.
    if (kind == Kind::FfxApiDll || kind == Kind::Fsr4) {
        std::unique_ptr<Backend> fallback = MakeFsr3Builtin();
        if (fallback && fallback->Init(*g_instance)) {
            std::fprintf(stderr, "SOTC upscaler: %s (%s unavailable)\n", fallback->Name(),
                         kind == Kind::Fsr4 ? "fsr4" : "ffxapi");
            g_backend = std::move(fallback);
            return g_backend.get();
        }
    }
    return nullptr;
}

void Shutdown() {
    std::lock_guard lock(g_mutex);
    if (g_backend) {
        g_backend->Shutdown();
        g_backend.reset();
    }
    g_tried = false;
    g_instance = nullptr;
}

} // namespace Vulkan::Upscaler
