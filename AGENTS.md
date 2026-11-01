# AGENTS.md — `/mnt/nvme/Games/soc-recomp`

Shadow of the Colossus (PS4 2018 Remake, `CUSA08809` EU 1.01 / `CUSA08034` US 1.00) unified PC native port repository (`AjncSystem` v0.70 base + `Oussama268` v0.70-opt12 + `Lucas-blip-png` v0.70-plus4 + `HiddenUserHere` PR #339 + `Gabriel-TwoBits` PR #329 native Linux).

---

## 0. Non-Negotiables
1. **Never launch the game executable yourself** (`SotC.exe --play`, `./sotc.sh`, or `out/bb-probe boot.bin`). Only run offline/smoke checks: `./build.sh --test`, `out/bb-probe --vulkan-only`, `out/bb-gpu-capabilities`, and `python3 -m unittest discover -s tests`.
2. **Respect `parallel-lead` roles and file sets**:
   - Lead (`@plan`/`@slow` -> `gemini-4-argon:high`) orchestrates, runs `judge_batch` (`jev-1.13:low`), verifies, and merges. Lead writes zero feature code.
   - Workers (`@worker` -> `muse-spark-1.3-contributor:high`) run with `isolated: true` on `omp/task/<name>` and touch ONLY their prompt-listed file set.
   - Watchdog (`@advisor` -> `gemini-4-argon:medium`) audits workers every 4 turns against `.omp/WATCHDOG.md`.
   - Never use Claude Opus (`anthropic/claude-opus-*` is excluded from `.omp/config.yml`).
3. **Never commitSony firmware or game dumps**: `.sprx`, `.prx`, `eboot.bin`, `*.psarc`, `out/`, `pgo/`, and `user/` must never be committed.

---

## 1. Host & Target Hardware Profile
- **OS / WM**: CachyOS Linux (`7.2.6-2-cachyos`), Hyprland Wayland, `/dev/ntsync` active.
- **CPU**: AMD Ryzen 5 5600 (6C/12T, Zen 3) — supports `avx2`, `bmi1`, `bmi2`, `lzcnt`, `movbe`, and native `sse4a`. Primary performance limiter (`shadPS4:GpuDraw` + `GpuCommandProcessor`).
- **GPU / Driver**: AMD Radeon RX 6800 (Navi 21, RDNA2, 16 GB VRAM), `Mesa 26.2.5-arch3.1` (`radv`), Vulkan `1.4.354`.
- **Host Quirks**:
  - **Keep PS4 Pro mode OFF (`SOTC_NEO=0`)**: On Mesa 26.2.x RADV, `SOTC_NEO=1` renders Wander's cape flat lime-green (PR #329) and adds ~25% CPU/GPU load.
  - **Internal resolution `720p` -> FSR upscale**: Cuts CPU draw commands per frame by 14% (+29% real FPS on Zen 3).
  - **2-flip upload deferral (`kMaxDeferFlips = 2`)**: Prevents `SOTC_UPLOAD_BUDGET_MB=32` from starving the Bluepoint logo and main-menu text (#341).

---

## 2. Remotes & Build Commands

| Remote | URL | Purpose |
|---|---|---|
| `origin` | `https://github.com/AjncSystem/ShadowOfTheColossus-PC.git` | Upstream `v0.70` base |
| `oussama` | `https://github.com/Oussama268/ShadowOfTheColossus-PC.git` | `v0.70-opt12` (PR #351 flip pacer, PR #349 partial readbacks, PR #372 shader cache fix, `SOTC_MUTEX_SPIN_US=20`, PR #371 HDR10) |
| `lucas` | `https://github.com/Lucas-blip-png/ShadowOfTheColossus-PC.git` | `v0.70-plus4` (FSR 3.1 reactive mask, `SOTC_GENERAL_SYNC` Shrine barriers, `BB_MARCH`) |
| `hidden` | `https://github.com/HiddenUserHere/ShadowOfTheColossus-PC.git` | `v0.70.1` / PR #339 (`6f1eaab` 2-flip upload deferral fix for missing menu text) |
| `gabriel` | `https://github.com/Gabriel-TwoBits/ShadowOfTheColossus-PC.git` | `linux-native` / PR #329 (native Linux loader, `<2 GiB` eboot, `SIGTRAP` BPE heap guard, case-insensitive `/app0`) |

### Build & Verification Commands
```bash
# Configure & build native Linux GPU library + loader + smoke tests
./build.sh
./build.sh --test

# Run Vulkan 1.3 headless smoke check & capability probe (safe: does not launch the game)
./out/bb-probe --vulkan-only
./out/bb-gpu-capabilities

# Run Python unit test suite
python3 -m unittest discover -s tests -p "test_*.py"
```

See `docs/PARALLEL.md` and `docs/LEAD.md` for worker slice boundaries and the dispatch/review protocol.
