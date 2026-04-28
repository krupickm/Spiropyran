# Stage notes

## 2026-04-28 — Stage 1 (prep)

### What was done
Implemented Stage 1 (`prep`) of the spiropyran d.r. pipeline. New files:
- `pyproject.toml` (rdkit, pyyaml; pytest + ruff dev).
- `spiropyran_dr/{__init__.py, config_utils.py}`.
- `spiropyran_dr/stages/{__init__.py, base.py, prep.py}`.
- `spiropyran_dr/config/{default.yaml, smarts.yaml}`.
- `tests/{__init__.py, conftest.py, test_config_utils.py, test_prep.py}`.

`prep.submit` canonicalises SMILES, runs sanity checks (charge, radical, heavy-atom floor), applies required/forbidden SMARTS filters, locates the spiro carbon and chromene oxygen via SMARTS in `config/smarts.yaml`, picks one stereochemistry at the spiro centre, and writes `prep/stereocentres.json` atomically. `prep.collect` is a no-op. Neither function touches the manifest file.

### Why
Stage 1 in `project.md` §10.1 is the single entry point that turns a raw user SMILES into a validated, atom-indexed reference frame for every downstream stage. Without fixed atom indices, the C-O distance constraint at MECP (§10.4) cannot be expressed unambiguously.

### What was decided
- **Stage outputs are returned, never written to the manifest** (matches §4 — orchestrator owns manifest I/O).
- **`smiles_anti == smiles_syn == smiles_canonical` at this stage** — anti/syn is geometric and assigned by `mm` (§10.1, CLAUDE.md hard rule 4).
- **`Stage` Protocol uses callable attributes, not methods with `self`** — modules satisfy it structurally. Spec-checker flagged the original method form as misleading for type-checkers.
- **v0 SMARTS patterns committed as drafts**: `[#6;X4;R2](-[#7])(-[#8])` for spiro carbon, `[#8;X2;R1]-[#6;X4;R2]` for chromene oxygen. Both require chemist review before non-BIPS scaffolds; warning block is in `smarts.yaml`.
- **`config["paths"]["smarts"]`** is read by `prep.submit` to allow tests (and a future CLI) to override the smarts file path. Not yet documented in §8.
- **Heavy-atom floor of 10** as a small-input guard; rejects `CCO` cleanly.

### What was deferred
- Orchestrator (`pipeline.py`), `pbs_utils`, `io_utils`, CLI — all out of scope.
- `compute_config_hash` (§5.3) belongs in `config_utils.py` but isn't needed until the orchestrator lands.
- `tests/` is at repo root; spec §9 shows it inside the package. Decide alongside packaging layout.
- `paths.smarts` config key — either add to §8 or remove the lookup in `prep.submit`.
- `config/default.yaml` only carries `filtering`; needs `mecp`, `crest`, `xtb_constr`, `dft`, `ensemble`, `pbs`, `polling` before later stages run.

### Test status
31 passed, 0 failed (1.4s) on `d:/devel/spiro/.venv` (rdkit 2026.03.1, pyyaml 6.0.3, pytest 9.0.3).

### Spec impact
None applied this session. Two pending items that will require touching `project.md` when their stages land: documenting `paths.smarts` in §8, and either confirming the prep `outputs` extras (`spiro_cip`, `smarts_filter`, `stereocentres_path`) in §5.1 or removing them.
