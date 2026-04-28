---
description: Add a new pipeline stage (CREST, DFT, analysis, etc.) following the project conventions.
---

You are adding a new stage to the in-silico screening pipeline.

Before writing code, gather the following from the user (ask if not given):
1. **Stage name** (e.g., `crest_stage`, `dft_stage`).
2. **Inputs** — what does this stage receive from the previous stage?
3. **Outputs** — what does it produce, in what format, where?
4. **External tools** — which CLI tools (CREST, ORCA, xTB, Multiwfn) are invoked?
5. **PBS resource hints** — wall time, queue, ncpus, memory.

Then produce a `/plan` covering:
- A new module under `src/` named `<stage>_stage.py`.
- Functions: `prepare_inputs`, `submit`, `poll`, `parse_outputs`. Adjust if the spec dictates otherwise.
- A PBS script template under `templates/` (or wherever templates live in the repo).
- Fixture output files under `tests/fixtures/<stage>/`. **At least one success fixture and one failure fixture.**
- Tests in `tests/test_<stage>_stage.py` covering: input preparation, mock submission, output parsing (using the fixtures), error handling on failed jobs.
- Updates to `project.md` describing the new stage.

Use the `spec-checker` sub-agent to verify the proposed signatures match the patterns established for existing stages.

Wait for approval before implementing.
