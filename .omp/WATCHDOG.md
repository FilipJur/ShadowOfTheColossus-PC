# Watchdog Review Rules

## Blockers (Must Flag Immediately)
1. **Out-of-Scope File Edits**: Worker modified or created any tracked file not explicitly listed in its `Files you may change` prompt section.
2. **Forbidden Git Operations**: Worker ran `git checkout`, `git switch`, `git merge`, `git push`, `git rebase`, `git stash`, or committed build outputs (`out/`, `dist/`).
3. **Peer Polling / Waiting**: Worker attempted to inspect or wait on sibling worker branches or jobs.
4. **Unsafe Execution**: Worker attempted to launch a full game binary or modify files outside `/mnt/nvme/Games/soc-recomp`.

## Concerns (Warn and Correct)
1. **Unverified Claims**: Worker claims build/tests pass without running the acceptance command or quoting real output.
2. **Dropped Guards**: Worker accidentally removed v0.70 crash/memory guards (`0x49800` allocator fix, RDNA2 wave32 path, `SOTC_VIEW_CHUNK_MB`) while cherry-picking or editing.
3. **Scope Creep**: Worker refactored unrelated functions or added unrequested features/abstractions.
4. **Repeated Failing Calls**: Worker repeated the same failing build/git command without diagnosing the error log.
