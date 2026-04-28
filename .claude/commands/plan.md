---
description: Produce a written implementation plan and wait for approval before any code changes.
---

You are about to make changes to the codebase. Before any file is edited, produce an implementation plan.

The plan must include:

1. **Goal restated in one sentence.**
2. **Files to be touched** — list them, with a one-line description of what changes in each.
3. **Function-level changes** — for each function added or modified, a one-line description of its signature and behaviour.
4. **Tests added or changed** — list test names and what they assert.
5. **Spec impact** — does this change require updating `project.md` or `insilico-screening.md`? If yes, which sections?
6. **Risks and open questions** — anything where you would otherwise have to guess. Especially: anything that touches CLAUDE.md hard rules.
7. **Out of scope** — what this change deliberately does NOT do.

After producing the plan, **stop and wait for approval.** Do not edit any files. Do not run any commands beyond what was needed to read the codebase for planning.

If the plan reveals that the change is bigger than expected, say so. If it reveals that the change is trivial and a plan is overkill, say that too.

Use the `spec-checker` sub-agent if you need to verify any spec details.
