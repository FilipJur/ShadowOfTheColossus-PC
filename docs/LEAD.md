# Lead Operating Playbook (`soc-recomp`)

## 1. Lead Mandate
- **Decides, never implements**: The Lead plans slices, runs `judge_batch` (`opencode-zen/jev-1.13:low`) commit triage, writes worker prompts, reviews `omp/task/<name>` diffs, runs integration checks, and merges serially into `main`.
- **Zero feature code by Lead**: Even shared contracts (`gpu/bbgpu.h`, `src/runtime.h`, `gpu/CMakeLists.txt`, `build.sh`) go through a dedicated isolated contract worker (`ContractEnvAbi`) first.
- **No game launches**: Per `/mnt/nvme/Games/AGENTS.md` §6, neither Lead nor workers may launch the full game executable (`SotC.exe --play` or `bb-probe boot.bin`). Verification uses `./build.sh --test`, `out/bb-probe --vulkan-only`, `out/bb-gpu-capabilities`, and `python3 -m unittest discover -s tests`.

## 2. Pre-Dispatch Checklist
1. **Clean working tree**: `git status --short` MUST be empty before calling `task` with `isolated: true` (`overlayfs` snapshots the live checkout).
2. **Contract-first order**: Any shared header (`gpu/bbgpu.h`, `src/runtime.h`, `gpu/shadps4/video_core/sotc_stats.h`, `gpu/shadps4/video_core/sotc_fps_cap.h`, `gpu/shim/bbport_mem_report.h`) or build script (`build.sh`, `gpu/CMakeLists.txt`) is updated and merged to `main` before parallel workers fan out.
3. **Disjoint file sets**: Every parallel worker prompt lists explicit backticked paths under `Files you may change: …`. No file may appear in two concurrent workers' lists.
4. **Unique task names**: Never reuse a `name` across `task` calls (`omp/task/<name>` branch names must be fresh).

## 3. Dispatch & Steering Loop
1. Spawn up to 4 concurrent `worker` / `ui-worker` tasks in a single `task` batch with `isolated: true`.
2. Watchdog (`@advisor` -> `google-antigravity/gemini-4-argon:medium`) automatically audits each worker every 4 turns against `.omp/WATCHDOG.md` (logs in `__advisor.lead-rules.jsonl`).
3. Steer live workers with `write agent://<name>` if needed; wait with `wait`.

## 4. Review & Serial Merge Protocol
For each completed `omp/task/<name>` branch:
1. **Inspect diff & transcript**:
   - `git diff --stat main..omp/task/<name>` (verify file-set boundary).
   - `git diff main..omp/task/<name>` (inspect hunks).
   - Read `history://<name>` if any anomalies appear.
2. **Automated `judge_batch` (`jev-1.13`) Pre-Merge Gate**:
   - Evaluate `{scope_clean, guards_preserved, tests_verified}` across all completed worker branches in `eval`.
3. **Helper `reviewer` Audit**:
   - Spawn `agent: "reviewer"` to check every requirement in the worker prompt against `git diff main..omp/task/<name>`, reporting only correctness gaps.
4. **Serial Merge & Verification**:
   - `git merge --no-ff omp/task/<name>` one branch at a time.
   - Run `./build.sh --test`, `out/bb-probe --vulkan-only`, and `python3 -m unittest discover -s tests` on the merged `main`.
   - Delete the merged branch (`git branch -d omp/task/<name>`).
5. **Fix Round (if a branch fails review)**:
   - Spawn a new isolated worker (`<name>Fix1`) starting with `git cherry-pick main..omp/task/<name>` plus the required fixes.
