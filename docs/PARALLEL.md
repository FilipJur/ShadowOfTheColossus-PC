# Parallel Architecture (`soc-recomp`)

## Roles (Role != Spawn Depth)

| Role | Agent | Model Role | Responsibility |
|---|---|---|---|
| **Lead** | Top-level session | `@plan` / `@slow` (`gemini-4-argon:high`) | Plans, runs `judge_batch` (`jev-1.13`) commit triage, dispatches contract + parallel workers, reviews diffs, runs final build/smoke checks, merges serially to `main`. Writes zero feature code. |
| **Worker** | `worker` | `@worker` (`muse-spark-1.3-contributor:high`) | Implements one isolated slice on `omp/task/<name>`, edits only its assigned file set, runs tests, makes atomic commits. |
| **UI Worker** | `ui-worker` | `@ui` (`gemini-3.8-flash:high`) | Same for `launcher/` UI tasks. |
| **Helper** | `scout`, `reviewer`, `sonic`, `task` | `@task` (`deepseek-v4.1-flash:high`) | Fast read-only lookups (`scout`), post-branch correctness review (`reviewer`), mechanical fixes (`sonic`). |
| **Advisor** | Watchdog (`advisor: true`) | `@advisor` (`gemini-4-argon:medium`) | Audits each worker every 4th turn against `.omp/WATCHDOG.md`. |
| **Judge** | `eval` `judge_batch` | `@judge` (`jev-1.13:low`) | 32-way concurrent commit triage across remotes and pre-merge diff gate. |

## File-Set & Contract Rules
1. Shared headers (`gpu/bbgpu.h`, `src/runtime.h`, `gpu/sotc/sotc_stats.h`) and build scripts (`CMakeLists.txt`, `gpu/CMakeLists.txt`, `build.sh`) belong to the **Contract Worker** (`contract-env-abi`), which runs and merges to `main` *before* parallel workers fan out.
2. Parallel workers run with `isolated: true` (`task.isolation: { merge: branch, apply: false }`) so commits land on `omp/task/<name>` while `main` stays clean.
3. Lead verifies every branch via `git diff main..omp/task/<name>`, `judge_batch` pre-merge gate, `reviewer` subagent, and direct test execution before merging one branch at a time.
