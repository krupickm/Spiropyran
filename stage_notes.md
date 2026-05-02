# Stage notes

## 2026-05-02 — Stage 2 (mm) + dihedral-based labeller

### What was done
Implemented Stage 2 (`mm`) of the pipeline. New / changed files:
- `spiropyran_dr/stages/mm.py` (new): ETKDGv3 embed + MMFF94 optimise + geometric labelling + greedy RMSD clustering + per-conformer XYZ + sidecar JSON.
- `spiropyran_dr/io_utils.py` (new): minimal XYZ writer/reader and atomic JSON writer (factored out of `prep.py`).
- `spiropyran_dr/stages/prep.py`: now also surfaces `indoline_nitrogen_idx` and `gem_carbon_idx`; reuses `io_utils.atomic_write_json`.
- `spiropyran_dr/config/{default.yaml,smarts.yaml}`: added `mm:` and `ensemble:` config blocks; added `indoline_nitrogen` and `gem_carbon` SMARTS roles.
- `spiropyran_dr/config_utils.py`: `load_smarts` now requires all four atom roles; `load_config` surfaces `mm` and `ensemble` defaults including `energy_window_kj_mol`.
- `spiropyran_dr/cli.py`: new `mm` subcommand (runs prep + mm back-to-back).
- `tests/{test_io_utils.py,test_mm.py}` (new) and updates to `test_prep.py`, `test_cli.py`, `test_config_utils.py`, `conftest.py`.

`mm.submit` consumes prep's outputs (atom indices), embeds N (~50) conformers, MMFF-optimises, labels each by a signed dihedral, RMSD-clusters within each label, writes `mm/{anti,syn}/conf_{i}.xyz` plus `mm/conformers.json`. Stage `failed` if either label is empty after clustering. `mm.collect` is a no-op.

### Why
Stage 2 in `project.md` §10.2 produces the per-diastereomer conformer ensembles that feed CREST and the constrained xTB stages. Without geometric labelling at this point, the downstream pipeline has no anti/syn split.

### What was decided
- **Anti/syn labelling = signed dihedral `chromene_O – C_spiro – indoline_N – indoline_anchor`**, where `indoline_anchor` is the unique indoline-ring atom bonded to N other than the spiro carbon. Positive sign → `anti`, negative → `syn`. Arbitrary but deterministic; chirality inversion at C_spiro flips the sign cleanly. This is the convention that lets BIPS (achiral, gem-dimethyl) split correctly because ETKDG samples both spiro enantiomers from the connectivity-only canonical SMILES. Plane-displacement variants were rejected because they were invariant under spiro-chirality flip when both atoms moved together.
- **RMSD clustering = greedy / energy-ascending exemplar selection** with `rdMolAlign.GetBestRMS`. Factored into a standalone `cluster_by_rmsd()` so the algorithm can be swapped (Butina, hierarchical, ML) without touching the orchestration code.
- **MM run knobs (`n_embed`, `mmff_max_iters`, `rmsd_threshold_angstrom`, `random_seed`) live in config**, not as constants in code. Defaults documented in `config/default.yaml` and in `project.md` §8.
- **Atom-role identification stays in prep**, not duplicated in mm. `mm.is_ready` checks that all four indices are present in prep's outputs before doing any work.
- **Test fixture for asymmetric scaffold:** `chiral_bips_smiles` (1-ethyl-1-methyl spiro) added to `conftest.py` for tests that need real diastereomer asymmetry; the `bips_smiles` fixture stays for the symmetric case (now also passing).

### What was deferred
- Ruff lint (2 F541 in `cli.py`) and `ruff format` on 7 files — user opted to skip; leftover from the earlier prep-CLI work.
- BIPS validation (§13): unblocked by the dihedral labeller, but the chemistry plausibility of the dihedral-sign convention should be sanity-checked on a real ground-truth molecule before stages 3+ are wired up.
- `compute_config_hash` (§5.3) — still not needed; will be required when chemistry-relevant config sections (`mecp`, `dft`, `crest`, `xtb`) land.

### Test status
61 passed, 0 failed (≈3 s) on `d:/devel/spiro/.venv` (rdkit 2026.03.1, pyyaml 6.0.3).

### Spec impact
`project.md` updated this session: §5 (expanded mm outputs example), §8 (added `mm:` config block), §10.1 (extended prep outputs list), §10.2 (specified the labelling dihedral and rationale; replaced the vague "dihedral around the spiro centre, sign of out-of-plane displacement" with a concrete dihedral chain). No outstanding drift after these edits.

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
