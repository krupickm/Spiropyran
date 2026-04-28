# Working agreement for Claude Code in this repo

## Project context
This is the in-silico screening pipeline for spiropyran photoswitches.
- **Specification:** `insilico-screening.md` (scientific scope) and `project.md` (architecture, modules, CLI, config).
- **Read these before proposing architecture changes or adding modules.** They are the source of truth.
- The package is responsible for: writing input files, submitting PBS jobs via `qsub`, polling for completion via the filesystem, and parsing output files into Python data structures. It is **not** responsible for installing or environment-loading external tools (ORCA, CREST, xTB, Multiwfn) — those are loaded by MetaCentrum's `module` system inside the PBS scripts.

## Hard rules
1. **Pure-Python orchestration.** Do not introduce ASE, xtb-python, or workflow DSLs (Snakemake, Nextflow, Prefect, Airflow). If you think one is needed, stop and ask — there is almost certainly a reason it was excluded.
2. **No real `qsub` calls in tests.** All tests must run on the developer's laptop with no MetaCentrum access. Use the mock submission backend and committed fixture output files in `tests/fixtures/`.
3. **Conformers are always lists.** Treat conformers per diastereomer as `List[Conformer]` with `N >= 1`, never as a single object. This is to accommodate macrocycle handling later.
4. **Diastereomer labelling is geometric, not SMILES-based.** Do it after MM optimisation by inspecting 3D coordinates around the spiro centre.
5. **Output paths follow `{molecule_id}/{anti,syn}/conf_{i}.xyz` with a JSON sidecar.** Do not change this without updating `project.md` in the same commit.
6. **SMARTS patterns and reaction templates live in YAML config**, not hardcoded. The spec calls this out — keep it that way so scaffolds can be swapped without code changes.

## Workflow rules
- **Plan before code.** For any change touching more than one file, produce a written plan (file-by-file, function-by-function) and wait for approval before editing. Use `/plan`.
- **Use sub-agents for verbose work.** When you would otherwise dump test logs, ORCA outputs, or grep results into the main context, delegate to the appropriate sub-agent.
- **Spec drift check.** When you change a module, check whether `project.md` still matches. If it doesn't, update it in the same commit. Use `/check-spec` to verify.
- **Tests first when adding parsers.** New output-file parsers must come with a fixture file in `tests/fixtures/` and a test before the parser is considered done.
- **Document why, not what.** Inline comments should explain chemistry-specific decisions (why this MECP distance, why this functional). Do not narrate code mechanics.

## Style
- Direct, technical, no marketing prose.
- No emojis in code, comments, or commit messages.
- Type hints throughout.
- `ruff` for lint/format. Default config.

## Things to ask about, never assume
- Choice of DFT functional, basis set, or solvation model.
- PBS queue selection (default is `oven` for long control jobs).
- Any change to the diastereomer labelling logic.
- Adding a new external dependency.
