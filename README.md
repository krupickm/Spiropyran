# spiropyran_dr

In-silico screening pipeline for spiropyran photoswitches: SMILES to predicted
diastereomeric ratio.

For the scientific scope and rationale, see [`insilico-screening.md`](insilico-screening.md).
For architecture, stage contracts, manifest schema, and config layout, see
[`project.md`](project.md). [`CLAUDE.md`](CLAUDE.md) holds the working agreement
for this repo.

## Status

Stages 1 (`prep`) and 2 (`mm`) are implemented, with a minimal CLI for
invoking each on a single SMILES. The orchestrator (`pipeline.py`), PBS
infrastructure, and stages 3-7 are not yet written.

## Requirements

- Python 3.10 or newer.
- A C/C++ toolchain is not needed: `rdkit` ships as a wheel for the supported
  Python versions on Windows / macOS / Linux.

## Environment setup

PowerShell (Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

bash / zsh (Linux, macOS, Git Bash):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`-e` installs the package in editable mode so edits to `spiropyran_dr/` are
picked up without reinstalling. `.[dev]` adds `pytest` and `ruff`.

## Running the test suite

```bash
pytest
```

Tests live under `tests/` and run entirely on the developer's laptop with no
cluster access (CLAUDE.md hard rule 2). They cover the pure helpers and
full `submit` / `collect` / `is_ready` contract for stages 1 and 2, the
XYZ writer and atomic JSON writer in `io_utils`, the YAML loaders, and the
CLI's exit codes and output.

## Linting and formatting

```bash
ruff check .
ruff format .
```

Default `ruff` configuration; no project overrides yet.

## Layout (current)

```
spiropyran_dr/
  __init__.py
  __main__.py                # enables `python -m spiropyran_dr`
  cli.py                     # argparse CLI (subcommands: prep, mm)
  config_utils.py            # YAML loading for config + smarts
  io_utils.py                # XYZ writer/reader, atomic JSON writer
  config/
    default.yaml             # filtering, mm, ensemble blocks at this stage
    smarts.yaml              # atom-role SMARTS (chemist review pending)
  stages/
    base.py                  # Stage Protocol
    prep.py                  # stage 1: SMILES validation, atom-role lookup,
                             # stereocentre handling, sidecar JSON write
    mm.py                    # stage 2: ETKDGv3 + MMFF94, dihedral-based
                             # anti/syn labelling, RMSD clustering, XYZ + sidecar
tests/
  conftest.py                # shared SMILES fixtures (BIPS, methyl-BIPS,
                             # chiral-BIPS, ...)
  test_cli.py
  test_config_utils.py
  test_io_utils.py
  test_mm.py
  test_prep.py
stage_notes.md               # dated decision log appended per session
```

The full target layout (orchestrator, PBS templates, all stages) is documented
in [`project.md` section 9](project.md). Files appear as their stages land.

## Running the prep stage

Via the console script (after `pip install -e .`):

```bash
spiropyran-dr prep "CC1(C)c2ccccc2N(C)C13Oc4ccccc4C=C3"
```

Or as a module without installing the entry point:

```bash
python -m spiropyran_dr prep "CC1(C)c2ccccc2N(C)C13Oc4ccccc4C=C3"
```

Useful flags:

- `--workspace PATH` — directory for `prep/stereocentres.json` (default: `./run_scratch`).
- `--config PATH` — pipeline config YAML (default: bundled `config/default.yaml`).
- `--smarts PATH` — atom-role SMARTS YAML (default: bundled `config/smarts.yaml`).
- `--json` — dump the full `submit()` return dict on stdout.

Exit code is 0 on success, 1 when the stage returns `failed`, 2 on usage
errors. On failure the reason is printed on stderr.

## Running the mm stage

Runs stage 1 and stage 2 back-to-back: prep produces the atom indices, mm
embeds N (~50) ETKDGv3 conformers, MMFF94-optimises them, labels each
conformer `anti` or `syn` from the signed dihedral
`chromene_O – C_spiro – indoline_N – indoline_anchor` (positive → anti,
negative → syn; see [`project.md`](project.md) §10.2 and the
[`mm.py`](spiropyran_dr/stages/mm.py) module docstring for the rationale),
RMSD-clusters within each label, and writes
`mm/{anti,syn}/conf_{i}.xyz` plus a `mm/conformers.json` sidecar.

```bash
spiropyran-dr mm "CC1(C)c2ccccc2N(C)C13Oc4ccccc4C=C3"
```

Useful flags (in addition to those listed for `prep`):

- `--n-embed N` — override `mm.n_embed` from config (e.g. `20` for fast
  smoke runs).
- `--seed S` — override `mm.random_seed`.

The stage returns `failed` if either diastereomer ends up with zero
conformers after labelling and clustering.

## Using the prep stage from Python

```python
from pathlib import Path
from spiropyran_dr.config_utils import load_config
from spiropyran_dr.stages import prep

config = load_config(Path("spiropyran_dr/config/default.yaml"))
manifest = {"smiles_input": "CC1(C)c2ccccc2N(C)C13Oc4ccccc4C=C3"}
workspace = Path("./run_scratch")

result = prep.submit(manifest, workspace, config)
print(result["status"])           # "done" or "failed"
print(result["outputs"])           # canonical SMILES, atom indices, CIP, ...
```

`submit` writes `<workspace>/prep/stereocentres.json` and returns the dict
that the orchestrator (when it lands) will merge into `manifest['stages']['prep']`.
The function does not write the manifest itself.
