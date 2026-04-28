---
description: Check the current code (or staged diff) against insilico-screening.md and project.md for drift.
---

Delegate to the `spec-checker` sub-agent.

Pass it:
- The current `git diff` (staged if anything is staged, otherwise unstaged).
- If there is no diff, the most recently modified files in `src/`.

Ask it to report DRIFT, STALE SPEC, and AMBIGUOUS issues per its standard output format.

When the sub-agent returns, summarise the verdict in one or two sentences and ask whether to:
1. Update `project.md` to match the code (if STALE SPEC dominates),
2. Revise the code to match the spec (if DRIFT dominates), or
3. Open a discussion to resolve ambiguities.
