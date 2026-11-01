# Parallel Architecture (`soc-recomp`)

## 1. Roles (Role != Spawn Depth)

| Role | Agent | Model Role (`.omp/config.yml`) | Fallback Chain | May |
|---|---|---|---|---|
| **Lead** | Top-level session | `@plan` / `@slow` (`google-antigravity/gemini-4-argon:high`) | `gemini-4-argon-medium:high` -> `deepseek-v4.1-flash:high` | Plan, run `judge_batch` triage, spawn workers/helpers, review diffs, merge to `main`. No feature code. |
| **Worker** | `worker` | `@worker` (`opencode-go/muse-spark-1.3-contributor:high`) | `deepseek-v4.1-flash:high` -> `gemini-3.8-flash:high` | Edit its exclusive file set on isolated `omp/task/<name>` branch, run checks, commit, spawn helpers. |
| **UI Worker** | `ui-worker` | `@ui` (`google-antigravity/gemini-3.8-flash:high`) | `deepseek-v4.1-flash:high` | Same for `launcher/` GUI files. |
| **Helper** | `scout`, `reviewer`, `sonic`, `task` | `@task` (`opencode-go/deepseek-v4.1-flash:high`) | `muse-spark-1.3-contributor:high` -> `gemini-3.8-flash:high` | Read-only exploration (`scout`), correctness diff review (`reviewer`), small mechanical edits (`sonic`). |
| **Advisor** | Watchdog (`advisor: true`) | `@advisor` (`google-antigravity/gemini-4-argon:medium`) | `deepseek-v4.1-flash:high` | Reviews each worker every 4th turn against `.omp/WATCHDOG.md`. |
| **Judge** | `eval` `judge_batch` | `@judge` (`opencode-zen/jev-1.13:low`) | `opencode-zen/jev-1.13` | 32-way concurrent commit classification, `SOTC_*` switch scoring, and pre-merge diff gating. |

Depth limit: `task.maxRecursionDepth: 2` (`Lead -> Worker -> Helper`). `prewalk` is disabled.

## 2. Run Modes
- **Native Isolated Mode (default)**: `task.isolation: { enabled: true, merge: branch, apply: false }`. Each worker runs in an `overlayfs` worktree (`~/.omp/wt/<id>`) and commits to `omp/task/<name>` while `main` remains untouched.
- **Scripted `workflowz` Mode**: Used in `eval` via `judge_batch()` or `workpool()` for bulk commit triage across remotes (`origin`, `oussama`, `lucas`, `hidden`, `gabriel`, `rhys`) and `SOTC_*` switch matrix evaluation.

## 3. Disjoint Subsystem Slices & File Sets

| Slice | Worker Name | Exclusive File Set | Upstream/Fork Sources | Runnable Checks |
|---|---|---|---|---|
| **0. Contract** *(merged first)* | `ContractEnvAbi` | `build.sh`, `gpu/CMakeLists.txt`, `gpu/bbgpu.h`, `src/runtime.h`, `gpu/shadps4/video_core/sotc_stats.h`, `gpu/shadps4/video_core/sotc_fps_cap.h`, `gpu/shim/bbport_mem_report.h`, `.gitignore` | `gabriel/linux-native` + `oussama/main` + `lucas/main` + CMake `FetchContent` (`robin-map`, `VMA`, `Zydis` 4.1.1) | `cmake -S gpu -B out/gpu -G Ninja` |
| **1. Loader & Guest Runtime** | `MemLoaderStutter` | `src/probe.c`, `src/runtime_memory.c`, `src/runtime_mutex.c`, `src/runtime_file.c`, `src/runtime_content.c`, `src/runtime_thread.c`, `src/runtime_fiber.c`, `src/runtime_sotc.c`, `tests/test_pad.c`, `tests/test_runtime.c`, `tests/test_sema.c` | `gabriel/linux-native` (Linux `<2 GiB` eboot, `SIGTRAP` BPE heap guard, case-insensitive `/app0`) + `oussama/main` (`SOTC_MUTEX_SPIN_US=20`, multi-VMA region cache) + `lucas/main` (`72e6d08bfb` flexible mapping split fix) | `./build.sh --test` (`pad-test`, `file-mods-test`, `content-test`) |
| **2. VRAM, Readbacks & Cache** | `VramReadbackCache` | `gpu/shadps4/video_core/buffer_cache/*`, `gpu/shadps4/video_core/texture_cache/*`, `gpu/shadps4/video_core/cache_storage.cpp`, `gpu/shadps4/video_core/page_manager.cpp`, `gpu/shadps4/common/io_file.*`, `gpu/shadps4/common/path_util.cpp` | `oussama/main` (PR #349 partial readback finish, `SOTC_RB_HOT` per-page lock, PR #372 `cache.sotcpack` loose-file priority fix, whole small-texture hash) + `hidden/main` (`6f1eaab` 2-flip `sotc_deferred_flip` Bluepoint logo/menu fix) + `gabriel/linux-native` (`page_manager.cpp`, `io_file.cpp`, `path_util.cpp`) | `ninja -C out/gpu bbgpu` |
| **3. Vulkan, FSR 3.1 & Pacer** | `VulkanFsr31Pacer` | `gpu/shadps4/video_core/renderer_vulkan/*`, `gpu/shadps4/video_core/host_shaders/*`, `gpu/shadps4/core/libraries/videoout/*`, `gpu/shim/bbgpu.cpp`, `gpu/shim/bbport_mem_report.cpp`, `gpu/shim/bbport_overlay.cpp` | `oussama/main` (PR #351 `F6` flip pacer + 60 FPS pacing, PR #350 off-thread `SOTC MEM`, PR #352 8-deep swapchain, PR #371 HDR10) + `gabriel/linux-native` (Linux build guards in `vk_framegen.cpp`, `vk_dlssg.cpp`, `vk_fsrfg.cpp`) (`SOTC_GENERAL_SYNC` already upstream in `texture_cache/image.cpp`; `lucas/main` FSR 3.1 temporal upscaler + `fsr3_reactive.comp` NOT ported yet: next slice) | `ninja -C out/gpu bbgpu && ./build.sh && out/bb-probe --vulkan-only` |
| **4. Launcher, Modes & Tools** | `LauncherTuningCli` | `launcher/*`, `scripts/*`, `sotc.py`, `sotc.sh`, `tools/sotc_route.py`, `docs/sotc/LINUX.md`, `tests/test_packaged_vulkan.py`, `tests/test_update.py` | `gabriel/linux-native` (cross-platform `sotc.py`, `sotc.sh`, `launcher/` Linux support) + `oussama/main` (`tools/sotc_route.py` `--boss 1..16`, `CUSA08034` US support, HDR mode in `sotc_modes.py`) | `python3 -m unittest discover -s tests -p "test_*.py"` |

## 4. Shared Registries & Conflict Prevention
- `gpu/CMakeLists.txt`, `build.sh`, `gpu/bbgpu.h`, `src/runtime.h`, `gpu/shadps4/video_core/sotc_stats.h`, and `gpu/shadps4/video_core/sotc_fps_cap.h` are frozen after `ContractEnvAbi` merges.
- `README.md` and `CHANGELOG.md` are owned by Lead at final cutover (workers must not touch them to avoid trivial merge conflicts).
