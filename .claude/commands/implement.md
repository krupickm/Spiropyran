---
description: Implement an approved plan. Use after /plan and explicit approval.
---

Implement the previously-approved plan.

Rules:
1. **Stick to the plan.** If you find that something must change, stop and report — do not silently expand scope.
2. **One file at a time.** After each file, briefly state what you did.
3. **Tests come with code, not after.** Write the test, then make it pass. If you wrote production code first, write the test before moving to the next file.
4. **Use sub-agents for verbose work** — `output-parser` for inspecting QC outputs, `test-runner` for running pytest. Do not paste long output into the main thread.
5. **Run `/run-tests` when implementation is complete.** Do not declare done until tests pass.
6. **Run `/check-spec` if `project.md` might be affected.**

Do not commit. Stop when implementation and tests are clean, then summarise what changed.
