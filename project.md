# Spiropyran d.r. Prediction Pipeline

A deterministic computational pipeline that takes a SMILES string for a spiropyran
precursor and returns a predicted diastereomeric ratio (d.r.) of the ring-closed
product, using a TS-mimetic constrained DFT protocol on the MetaCentrum HPC cluster.

This document is the architectural source of truth. It does not contain implementation
code; it specifies interfaces, file layouts, and stage contracts so that the
implementation in `spiropyran_dr/` can be built incrementally without architectural
drift.

---

## 1. Scientific basis

Spiropyran ring closure produces two diastereomers (anti / syn) at the spiro centre.
The ratio is set by kinetic control at a transition state near the minimum-energy
crossing point (MECP) between the S₁ and S₀ surfaces, *not* by ground-state
thermodynamics. CREST conformer searches on closed ground states have been shown
empirically to give the wrong answer because the ground-state energy landscape is
intrinsically flat (≤ 5 kJ/mol diastereomer energy differences).

The pipeline therefore approximates the diastereomeric transition states by
**TS-mimetic constrained optimisation**: the forming C–O bond is fixed at the
literature MECP distance (~3.4 Å, from Prager et al. 2014 and Bálint & Bende
ChemPhotoChem 2026), and the rest of the geometry is relaxed on the ground-state
surface with implicit solvation. The energy difference between the two diastereomeric
constrained geometries is taken as the proxy for ΔΔG‡.

The pipeline computes the d.r. along **two prediction pathways**, both reported,
because each one carries different information:

- **MECP / kinetic** (primary): ΔΔ between the constrained `{anti_mecp, syn_mecp}`
  ensembles — the TS-mimetic numbers. This is the pipeline's headline d.r.
- **Ground-state / thermodynamic** (secondary): ΔΔ between the unconstrained
  `{anti_min, syn_min}` ensembles — reported alongside for cross-checks and
  downstream descriptor work, but is *not* a substitute for the MECP number
  (the closed-form energy landscape is too flat to give a reliable d.r. on
  its own — see opening paragraph).

For each pathway the pipeline produces **two numbers**, both reported:

- **ΔΔE‡** — electronic energy difference at the relevant geometry (always computed)
- **ΔΔG‡** — Gibbs free energy difference including thermal corrections from
  vibrational frequencies at the relevant geometry (optional stage)

And **two ensemble treatments**, both reported:

- **Lowest-energy conformer** per label (primary within the pathway)
- **Boltzmann-weighted average** over the surviving conformer ensemble (secondary)

The d.r. is computed from each ΔΔ value via:

    d.r.(anti:syn) = exp(−ΔΔ / RT) : 1     at T = 298.15 K

No machine learning. No fitted parameters. The only domain choices are encoded in
`config/default.yaml`.

---

## 2. Pipeline overview

```
SMILES (one molecule)
    │
    ▼
[1] prep            local      SMILES → canonical, enumerate spiro stereocentre,
                               SMARTS sanity checks
    ▼
[2] mm              local      RDKit ETKDG + MMFF, N conformers per diastereomer,
                               geometric anti/syn labelling
    ▼
[3] xtb_constr      PBS        TS-mimetic seed: GFN2-xTB constrained opt with
                               C-O distance fixed at the MECP value. Two jobs
                               (one per diastereomer); each produces a single
                               MECP-mimic seed geometry that is later fed to
                               the constrained CREST branch.
    ▼
[4] crest           PBS        Four parallel jobs:
                                 - {anti,syn}_min : unconstrained GFN2-xTB
                                   conformational sampling on the closed
                                   ground state, seeded from the lowest MM
                                   conformer.
                                 - {anti,syn}_mecp: GFN2-xTB conformational
                                   sampling under the C-O distance constraint,
                                   seeded from the xtb_constr output. The
                                   constraint is enforced via .xcontrol passed
                                   through sub_crest.sh as `--cinp .xcontrol`.
    ▼
[5] dft_sp          PBS        ORCA single point: r2SCAN-3c → ωB97X-D3BJ/def2-TZVP
                               with SMD, per surviving conformer per label
                               (4 labels).
    ▼
[6] dft_freq        PBS        OPTIONAL: ORCA frequency calc at the (constrained
                               or unconstrained) CREST geometry → thermal
                               corrections → G(298 K), per surviving conformer
                               per label.
    ▼
[7] aggregate       local      Collapse conformer ensemble (lowest + Boltzmann)
                               per label. Compute ΔΔE‡/ΔΔG‡ and d.r. twice:
                               primary from the {anti_mecp, syn_mecp} ensembles
                               (kinetic, MECP-mimetic), secondary from the
                               {anti_min, syn_min} ensembles (thermodynamic,
                               ground-state).
```

Stages 1, 2, 7 are local Python (run inside the orchestrator process).
Stages 3, 4, 5, 6 submit PBS jobs and the orchestrator polls for completion.
Stage 6 is skipped if `--no-thermal` (default) is set; the pipeline then reports
only ΔΔE‡-based d.r.

---

## 3. Orchestrator model

The orchestrator is a single long-walltime PBS job in the OVEN queue. It:

1. Reads `manifest.json` for the molecule.
2. Walks the stage list. For the first non-`done` stage:
   - If `pending` and inputs are ready: do the work (local) or write inputs +
     `qsub` (PBS). Update manifest.
   - If `submitted` or `running`: check whether the PBS job finished. If yes,
     parse outputs and advance manifest. If no, sleep and re-check.
   - If `failed`: log and exit; manual intervention required.
3. When all stages are `done`, write `result.json` and exit.

The orchestrator process holds no state beyond what is in `manifest.json` on disk.
Restart-on-failure is `qsub` of the orchestrator script — it reads the manifest
and resumes from where it stopped. Polling interval should be configurable
(default 60 s).

### 3.1 Job-state detection

A PBS job is considered finished when **either** of the following is true:

- `qstat -f <job_id>` returns a terminal state (`C`, `F`, or job not found).
- The expected output sentinel file exists in the stage's work directory.

The output-file check is authoritative for success; `qstat` is the running
indicator. After detecting finish, parse outputs and decide success vs
failure by inspecting the stage's natural success signal. Per-stage
sentinels:

- **xtb_constr**: per base label (`anti`, `syn`): `xtbopt.xyz` present in
  the per-label work directory, and the measured C-O distance between
  `spiro_carbon_idx` and `chromene_oxygen_idx` within
  `xtb_constr.co_distance_tolerance_angstrom` of
  `mecp.c_o_distance_angstrom`. Final energy parsed from the
  `TOTAL ENERGY ... Eh` line in `xtb.out`.
- **crest**: per label (`anti_min`, `syn_min`, `anti_mecp`, `syn_mecp`):
  presence of *both* `crest_conformers.xyz` and `crest.energies` in the
  per-label work directory. CREST writes them only on a clean exit;
  absence of either in any label => failure. No separate `crest_done`
  flag file.
- **dft_sp / dft_freq**: `ORCA TERMINATED NORMALLY` string in `orca.out`,
  per conformer per label.

### 3.2 Module loading

The Python package never invokes the Linux `module` system. Each PBS template
contains its own `module add ...` lines. The orchestrator is responsible for
*writing inputs* and *parsing outputs*; environment is the cluster's job.

---

## 4. Stage interface

Every stage module under `stages/` exposes the same three functions. This uniform
contract is what makes `pipeline.py` a short, debuggable loop.

```
def is_ready(manifest: dict, workspace: Path) -> bool:
    """Are this stage's inputs available on disk?"""

def submit(manifest: dict, workspace: Path, config: dict) -> dict:
    """Do the work (local stages) or write inputs + qsub (PBS stages).
    Returns a dict to merge into manifest['stages'][stage_name]:
      - status: 'done' | 'submitted' | 'failed'
      - started_at: ISO timestamp (always)
      - For PBS stages: pbs_job_ids (dict[label_or_conf_id, str]),
        submitted_at. Stages that submit a fixed number of jobs keyed by
        label key by that label string -- xtb_constr by base diastereomer
        ('anti', 'syn'); crest by the four label scheme ('anti_min',
        'syn_min', 'anti_mecp', 'syn_mecp'). Stages that submit per
        conformer (dft_sp, dft_freq) key by a flat '{label}/conf_{i}'
        string. Per-job work directories are derivable from the workspace
        and stage conventions; they are not stored in the manifest.
      - For local stages: outputs (stage-specific dict), finished_at
      - For status=='failed': failure_reason (str), finished_at
    """

def collect(manifest: dict, workspace: Path, config: dict) -> dict:
    """Called once a PBS job has finished. Parse outputs.
    Returns a dict to merge into manifest['stages'][stage_name]:
      - status: 'done' | 'failed'
      - outputs: stage-specific dict (energies, paths to xyz, etc.)
      - finished_at: ISO timestamp
      - failure_reason: str if status == 'failed'
    For local stages, this is a no-op (work was done in submit()).
    """
```

Stages must be **idempotent**: calling `submit` when status is already `done`
is a no-op. Calling `collect` when output files don't exist returns `failed`
with a reason; the orchestrator decides whether to re-submit.

Stages must **not** read or write `manifest.json` directly. They receive the
manifest dict and return a dict update. Manifest I/O is done by the orchestrator.

---

## 5. Manifest schema

Single source of truth per molecule. JSON, hand-editable, lives at
`{workspace}/molecules/{molecule_id}/manifest.json`.

```json
{
  "schema_version": 1,
  "molecule_id": "sp_0001",
  "smiles_input": "...",
  "smiles_canonical": "...",
  "smiles_anti": "...",
  "smiles_syn": "...",
  "config_hash": "sha256:...",
  "config_path": "config/default.yaml",
  "created_at": "2026-04-27T12:00:00Z",
  "options": {
    "thermal": false
  },
  "stages": {
    "prep":        { "status": "done", "started_at": "...", "finished_at": "...",
                     "outputs": { ... } },
    "mm":          { "status": "done", "outputs":
                     { "n_conformers_anti": 12, "n_conformers_syn": 14,
                       "anti_xyz_dir": "mm/anti", "syn_xyz_dir": "mm/syn",
                       "anti": [ { "conf_id": 0, "embed_id": 7,
                                   "xyz": "mm/anti/conf_0.xyz",
                                   "mmff_energy_kcal_mol": 12.34,
                                   "label": "anti" }, "..." ],
                       "syn":  [ "..." ],
                       "sidecar_path": "mm/conformers.json" } },
    "xtb_constr":  { "status": "done",
                     "pbs_job_ids":
                       { "anti": "12300.meta-pbs", "syn": "12301.meta-pbs" },
                     "submitted_at": "...", "finished_at": "...",
                     "outputs":
                       { "anti": [ { "conf_id": 0,
                                     "xyz": "xtb_constr/anti/xtbopt.xyz",
                                     "energy_hartree": -22.123,
                                     "co_distance_final_ang": 3.402,
                                     "label": "anti" } ],
                         "syn":  [ "..." ] } },
    "crest":       { "status": "running",
                     "pbs_job_ids":
                       { "anti_min":  "12345.meta-pbs",
                         "syn_min":   "12346.meta-pbs",
                         "anti_mecp": "12347.meta-pbs",
                         "syn_mecp":  "12348.meta-pbs" },
                     "submitted_at": "..." },
    "dft_sp":      { "status": "pending" },
    "dft_freq":    { "status": "skipped" },
    "aggregate":   { "status": "pending" }
  },
  "result": null
}
```

### 5.1 Status values

`pending | submitted | running | done | failed | skipped`

- `pending` — not yet attempted
- `submitted` — `qsub` returned a job ID, not yet observed running
- `running` — observed in queue (informational only; we don't strictly need this)
- `done` — completed successfully, outputs parsed
- `failed` — terminated abnormally; orchestrator stops
- `skipped` — explicitly disabled (e.g. `dft_freq` when `--no-thermal`)

### 5.2 Conformer-level data

Stage 3 (`xtb_constr`) operates per base diastereomer (`anti`, `syn`) and
produces a single MECP-mimic seed geometry per label. Stages 4–6 (`crest`,
`dft_sp`, `dft_freq`) operate per surviving conformer per **label**, where
"label" is one of the four `{anti_min, syn_min, anti_mecp, syn_mecp}` keys
(the `_min` branches come from unconstrained CREST on the closed ground
state; the `_mecp` branches come from constrained CREST on the
`xtb_constr` seed). The manifest always records a list of conformer
entries inside each stage's `outputs`, regardless of length:

```json
"xtb_constr": {
  "status": "done",
  "outputs": {
    "anti": [
      { "conf_id": 0,
        "xyz": "xtb_constr/anti/xtbopt.xyz",
        "energy_hartree": -22.123,
        "co_distance_final_ang": 3.402,
        "label": "anti" }
    ],
    "syn": [ { "conf_id": 0, "...": "..." } ]
  }
}
```

```json
"dft_sp": {
  "status": "done",
  "outputs": {
    "anti_min":  [ { "conf_id": 0, "...": "..." }, { "conf_id": 1, "...": "..." } ],
    "syn_min":   [ "..." ],
    "anti_mecp": [ "..." ],
    "syn_mecp":  [ "..." ]
  }
}
```

Even when there is one conformer (e.g. every `xtb_constr` label, or any
label whose CREST search collapsed to a single basin) this is a list. The
list costs nothing now and saves a refactor when macrocyclic systems with
multiple conformers per diastereomer arrive (as well as when a future
extension sends multiple seeds per diastereomer into `xtb_constr`).

### 5.3 Config hash

`config_hash` = sha256 over the canonical-JSON serialisation of the
chemistry-relevant config sections (`mecp`, `xtb_constr`, `crest`, `dft`,
`solvent`).
On orchestrator start, recompute and compare to the stored hash. Mismatch =>
refuse to resume; the user must explicitly invoke a "force-rerun" mode.
Walltime / queue settings are not part of the hash.

---

## 6. Workspace layout

```
/storage/<group>/spiropyran_dr/
└── runs/
    └── <run_id>/
        ├── orchestrator.log
        ├── orchestrator.pbs.sh
        └── molecules/
            └── sp_0001/
                ├── manifest.json
                ├── result.json
                ├── prep/
                │   └── stereocentres.json
                ├── mm/
                │   ├── anti/
                │   │   └── conf_{0..N}.xyz
                │   └── syn/
                │       └── conf_{0..M}.xyz
                ├── xtb_constr/
                │   ├── anti/
                │   │   ├── input.xyz                   (copy of mm/anti/conf_0.xyz)
                │   │   ├── xtb.inp                     ($constrain block, --input target)
                │   │   ├── jobid
                │   │   ├── xtbopt.xyz
                │   │   └── xtb.out
                │   └── syn/ ...
                ├── crest/
                │   ├── anti_min/
                │   │   ├── input.xyz                   (copy of mm/anti/conf_0.xyz)
                │   │   ├── CRESTJOB_input_<pid>.sh     (written by sub_crest.sh)
                │   │   ├── jobid
                │   │   ├── input.crest.log
                │   │   ├── crest_conformers.xyz
                │   │   ├── crest.energies
                │   │   └── filtered/
                │   │       └── conf_{0..K}.xyz
                │   ├── syn_min/    ...
                │   ├── anti_mecp/
                │   │   ├── input.xyz                   (copy of xtb_constr/anti/xtbopt.xyz)
                │   │   ├── .xcontrol                   ($constrain block, --cinp target)
                │   │   ├── CRESTJOB_input_<pid>.sh
                │   │   ├── jobid
                │   │   ├── input.crest.log
                │   │   ├── crest_conformers.xyz
                │   │   ├── crest.energies
                │   │   └── filtered/
                │   │       └── conf_{0..K}.xyz
                │   └── syn_mecp/   ...
                ├── dft_sp/
                │   ├── anti_min/
                │   │   ├── conf_0/
                │   │   │   ├── orca.inp
                │   │   │   ├── pbs.sh
                │   │   │   ├── jobid
                │   │   │   └── orca.out
                │   │   └── conf_1/ ...
                │   ├── syn_min/    ...
                │   ├── anti_mecp/  ...
                │   └── syn_mecp/   ...
                └── dft_freq/  (optional, same layout as dft_sp)
```

`run_id` = ISO date or user-supplied string. One run = one orchestrator job =
one molecule (in v1).

---

## 7. Output: result.json

Always written when stage `aggregate` completes. CLI verbosity flags choose
which fields are echoed to stdout; the file is always complete.

```json
{
  "molecule_id": "sp_0001",
  "smiles_input": "...",
  "smiles_canonical": "...",

  "predictions": {
    "mecp": {
      "lowest_conformer": {
        "delta_e_kj_mol": 3.2,
        "dr_anti_syn_from_e": 0.78,
        "delta_g_kj_mol": 2.9,
        "dr_anti_syn_from_g": 0.76,
        "selected_conf": { "anti_mecp": 4, "syn_mecp": 1 }
      },
      "boltzmann": {
        "delta_e_kj_mol": 2.8,
        "dr_anti_syn_from_e": 0.76,
        "delta_g_kj_mol": 2.6,
        "dr_anti_syn_from_g": 0.74,
        "n_conformers_used": { "anti_mecp": 7, "syn_mecp": 9 }
      }
    },
    "ground_state": {
      "lowest_conformer": {
        "delta_e_kj_mol": 1.1,
        "dr_anti_syn_from_e": 0.62,
        "delta_g_kj_mol": 0.9,
        "dr_anti_syn_from_g": 0.59,
        "selected_conf": { "anti_min": 0, "syn_min": 2 }
      },
      "boltzmann": {
        "delta_e_kj_mol": 0.8,
        "dr_anti_syn_from_e": 0.58,
        "delta_g_kj_mol": 0.7,
        "dr_anti_syn_from_g": 0.57,
        "n_conformers_used": { "anti_min": 11, "syn_min": 12 }
      }
    }
  },

  "thermal_included": true,

  "energies": {
    "anti_min":  [ { "conf_id": 0, "e_dft_hartree": -1234.567,
                     "g_corr_hartree": 0.0421, "g_total_hartree": -1234.525 },
                   "..." ],
    "syn_min":   [ "..." ],
    "anti_mecp": [ "..." ],
    "syn_mecp":  [ "..." ]
  },

  "config_hash": "sha256:...",
  "config_path": "config/default.yaml",
  "wall_time_seconds": {
    "xtb_constr": { "anti": 35, "syn": 41 },
    "crest":      { "anti_min":  8421, "syn_min":  9123,
                    "anti_mecp": 7544, "syn_mecp": 8002 },
    "dft_sp":     31200,
    "dft_freq":   18400
  }
}
```

The CLI's headline number (printed when `--verbose` is not set) is
`predictions.mecp.lowest_conformer.delta_e_kj_mol` and the corresponding
d.r. — the kinetic, lowest-conformer prediction. The ground-state
prediction is reported alongside but is secondary.

If `--no-thermal`, `predictions.{mecp,ground_state}.{lowest_conformer,boltzmann}.delta_g_kj_mol`
and the corresponding `dr_anti_syn_from_g` are `null`, `thermal_included` is
`false`, and `g_corr_hartree` / `g_total_hartree` are absent.

---

## 8. Configuration

`config/default.yaml`. Everything chemistry-specific lives here. Code reads;
code does not decide.

```yaml
mecp:
  c_o_distance_angstrom: 3.4
  constraint_force_constant: 1.0   # xTB/CREST $constrain force constant

temperature_kelvin: 298.15

mm:
  n_embed: 50                    # ETKDGv3 attempts; balances diastereomer
                                 # coverage against MM cost
  mmff_max_iters: 200            # MMFF94 optimisation cap per conformer
  rmsd_threshold_angstrom: 0.5   # heavy-atom greedy clustering threshold
  random_seed: 42                # ETKDG seed for reproducible embeds

crest:
  walltime_hours: 24                # int, passed as 1st positional arg to sub_crest.sh
  script_path: "/storage/brno2/home/krupickm/bin/sub_crest.sh"
                                    # absolute path to the user-maintained CREST submission
                                    # wrapper. The wrapper hardcodes NPROC=7 (experimentally
                                    # optimal on MetaCentrum) and calls qsub itself; the
                                    # Python orchestrator does not render its own PBS script
                                    # for CREST. For the {anti,syn}_min labels the
                                    # orchestrator passes no CREST flags and trusts the
                                    # wrapper plus CREST defaults (GFN2 search,
                                    # ewin = 6 kcal/mol). For the {anti,syn}_mecp labels
                                    # it additionally writes a .xcontrol file in the work
                                    # directory and appends `--cinp .xcontrol` so the
                                    # conformational search runs under the C-O distance
                                    # constraint defined in the `mecp` section.

xtb_constr:
  walltime_hours: 1                                     # int, 1st positional arg to sub_xtb.sh
  script_path: "/storage/brno2/home/krupickm/bin/sub_xtb.sh"
                                                        # user-maintained xTB submission wrapper.
                                                        # Usage: sub_xtb.sh <walltime_hours>
                                                        #        <coord.xyz> <other-xtb-args>...
                                                        # xTB always runs on 1 CPU on MetaCentrum;
                                                        # this is owned by sub_xtb.sh.
  method: gfn2                                          # passed through as `--gfn 2` to xtb
  co_distance_tolerance_angstrom: 0.01                  # collect() rejects if the final C-O
                                                        # distance is farther than this from
                                                        # mecp.c_o_distance_angstrom

dft:
  small_basis_method: "r2SCAN-3c"
  large_basis_method: "wB97X-D3BJ def2-TZVP D3BJ"
  solvent:
    model: SMD
    name: acetonitrile
  freq_method: "r2SCAN-3c"        # cheap level for thermal corrections
  walltime_sp: "48:00:00"
  walltime_freq: "72:00:00"

ensemble:
  energy_window_kj_mol: 20.0      # discard conformers above this from Boltzmann avg
  max_conformers_per_diastereomer: 20

pbs:
  queue_default: "default@meta-pbs.metacentrum.cz"
  queue_orchestrator: "oven@meta-pbs.metacentrum.cz"
  walltime_orchestrator: "720:00:00"

filtering:
  smarts_required: []
  smarts_forbidden: []

polling:
  interval_seconds: 60
```

A separate `config/smarts.yaml` holds reaction templates and atom-role maps for
the prep/mm stages, kept out of the main config to allow scaffold swapping
without touching DFT settings.

---

## 9. Module layout

```
spiropyran_dr/
├── pipeline.py              # orchestrator main loop
├── pbs_utils.py             # qsub wrapper, qstat polling, jobid I/O,
│                            # template rendering
├── io_utils.py              # XYZ multi-frame reader/writer, ORCA / CREST /
│                            # xTB parsers, manifest read/write, atomic JSON
│                            # write
├── config_utils.py          # config loading, hash, schema validation
├── stages/
│   ├── __init__.py          # STAGE_ORDER, registry
│   ├── base.py              # Stage protocol (typing only)
│   ├── prep.py
│   ├── mm.py
│   ├── crest_stage.py
│   ├── xtb_stage.py
│   ├── dft_sp_stage.py
│   ├── dft_freq_stage.py
│   └── aggregate.py
├── templates/
│   ├── pbs_orchestrator.j2
│   ├── pbs_orca_sp.j2
│   └── pbs_orca_freq.j2
│   # Note: no pbs_crest.j2 or pbs_xtb_constrained.j2. Both CREST and
│   # constrained xTB submission are delegated to user-maintained wrappers
│   # (sub_crest.sh / sub_xtb.sh). Only ORCA jobs use rendered templates.
├── config/
│   ├── default.yaml
│   └── smarts.yaml
├── tools/
│   ├── inspect_manifest.py
│   └── resubmit_failed.py
└── cli.py                   # entry point: predict_dr.py

# Tests live at the repo root, alongside spiropyran_dr/, not inside it:
tests/
├── conftest.py
├── fixtures/                # canned QC outputs (CREST, ORCA, xTB)
│   ├── xtb_constr/{anti,syn}/{xtbopt.xyz, xtb.out}
│   └── crest/{anti_min,syn_min,anti_mecp,syn_mecp}/{crest_conformers.xyz, crest.energies}
├── test_io_utils.py
├── test_config_utils.py
├── test_pbs_utils.py
├── test_prep.py
├── test_mm.py
├── test_xtb_stage.py
├── test_crest_stage.py
├── test_aggregate.py
└── test_cli.py
```

---

## 10. Per-stage specifications

### 10.1 prep (local)

- Input: `smiles_input`
- Action:
  - Canonicalise with RDKit.
  - Enumerate the spiro stereocentre (R/S). Both enantiomers exist; we work
    with one and rely on geometric labelling later for anti/syn.
  - Apply `filtering.smarts_required` / `smarts_forbidden`. If any forbidden
    pattern matches or any required pattern is missing, mark stage `failed`
    with a reason.
  - Identify the spiro carbon and the chromene oxygen by SMARTS. Record their
    atom indices in the canonical SMILES atom order. These indices are used
    later for the C–O constraint.
- Output:
  - `smiles_canonical`, `smiles_anti`, `smiles_syn` (the latter two are the
    same connectivity SMILES; the anti/syn distinction is geometric and
    assigned at the mm stage).
  - `spiro_carbon_idx`, `chromene_oxygen_idx`,
    `indoline_nitrogen_idx`, `gem_carbon_idx` (atom indices in the canonical
    SMILES order; the latter two are required by the mm-stage geometric
    labeller and by the xtb_constr stage's distance constraint).
  - `spiro_cip` (R/S of the arbitrarily-fixed spiro stereocentre, recorded
    in `prep/stereocentres.json` for downstream traceability;
    `smiles_canonical` itself is connectivity-only).
  - `smarts_filter`: required/forbidden SMARTS check result.
  - `stereocentres_path`: relative path to the stereocentres JSON sidecar.

### 10.2 mm (local)

- Input: SMILES + atom indices from prep.
- Action:
  - RDKit ETKDGv3 to embed N (~50) conformers of the closed spiropyran.
    The canonical SMILES is connectivity-only, so ETKDG samples both
    spiro-carbon enantiomers across embeds, which is what makes the
    geometric label split.
  - MMFF94 optimise; energies in kcal/mol.
  - Label each conformer by the signed dihedral
    `chromene_O – C_spiro – indoline_N – indoline_anchor`, where
    `indoline_anchor` is the unique indoline-ring atom bonded to the
    indoline N other than the spiro carbon (the aromatic C in BIPS).
    Convention: positive sign → `anti`, negative → `syn`. The choice is
    arbitrary but deterministic; chirality inversion at C_spiro flips the
    sign, which is what we want. See `spiropyran_dr/stages/mm.py` for the
    full rationale and the alternatives that were considered.
  - Cluster within each label by RMSD (greedy, lowest-energy retained per
    cluster); keep up to `ensemble.max_conformers_per_diastereomer`
    lowest-energy per label.
  - Write `mm/{anti,syn}/conf_{i}.xyz` plus `mm/conformers.json` sidecar.
- Output: per-diastereomer lists of XYZ paths and MMFF energies.
- Failure modes: ETKDG fails to embed any conformer; or after labelling
  one of the two diastereomers ends up empty (log + fail).

### 10.3 xtb_constr (PBS, two parallel jobs — one per diastereomer)

This stage produces the MECP-mimic seed geometry that the constrained
CREST branch (stage 4, `_mecp` labels) starts from. It runs *before* CREST,
not after it. There is no per-CREST-conformer constrained refinement step
in this pipeline -- the constrained ensemble comes out of constrained
CREST (10.4) directly.

- Input: lowest-energy MM conformer per diastereomer
  (`mm/{anti,syn}/conf_0.xyz`) plus `spiro_carbon_idx` and
  `chromene_oxygen_idx` from prep.
- Action, per base label (`anti`, `syn`):
  - Copy the MM conformer to `xtb_constr/{label}/input.xyz`.
  - Write `xtb_constr/{label}/xtb.inp` containing an xtb-style
    `$constrain` block:
    ```
    $constrain
      force constant=<mecp.constraint_force_constant>
      distance: <i>,<j>,<mecp.c_o_distance_angstrom>
    $end
    ```
    where `<i>` = `spiro_carbon_idx + 1` and `<j>` = `chromene_oxygen_idx + 1`
    (xtb uses 1-based atom indices; prep stores 0-based RDKit indices).
  - Invoke
    `sub_xtb.sh <xtb_constr.walltime_hours> input.xyz --opt --gfn 2 --input xtb.inp`
    from the work directory. The wrapper takes `<walltime_hours> <coord.xyz>
    <other-xtb-args>...` and runs xTB on 1 CPU; the orchestrator passes
    `--opt` (geometry optimisation), `--gfn 2` (method, derived from
    `xtb_constr.method == "gfn2"`), and `--input xtb.inp` (read the
    constraint block) as the pass-through args.
  - Capture the wrapper stdout to record the PBS job id.
- `collect`, per base label:
  - Parse `xtb_constr/{label}/xtbopt.xyz` (single frame).
  - Parse the final energy in Hartree from the line containing
    `TOTAL ENERGY` in `xtb_constr/{label}/xtb.out`.
  - Measure the final C-O distance between the two constrained atoms.
    Reject (status `failed`) if
    `|measured - mecp.c_o_distance_angstrom| > xtb_constr.co_distance_tolerance_angstrom`.
- Output: per base label, a one-element list (per the "conformers always
  lists" rule in CLAUDE.md):
  ```
  outputs[label] = [
    { conf_id: 0,
      xyz: "xtb_constr/<label>/xtbopt.xyz",
      energy_hartree: <float>,
      co_distance_final_ang: <float>,
      label: "<label>" }
  ]
  ```

### 10.4 crest (PBS, four parallel jobs)

This stage runs CREST four times in parallel, keyed by the flat label set
`{anti_min, syn_min, anti_mecp, syn_mecp}`. The first axis (`anti` vs
`syn`) is the diastereomer; the second axis (`_min` vs `_mecp`) is the
seed source and constraint state.

- Input, per label:
  - `_min`: the lowest-energy MM conformer for the matching base label
    (`mm/{anti,syn}/conf_0.xyz`).
  - `_mecp`: the seed geometry from `xtb_constr.outputs[<base>][0].xyz`
    (i.e. `xtb_constr/{anti,syn}/xtbopt.xyz`).
- Action, per label:
  - Copy the seed to `crest/{label}/input.xyz`.
  - For `_mecp` labels only: write `crest/{label}/.xcontrol` with the
    same `$constrain` block as in 10.3 (force constant and distance from
    the `mecp` config, atom indices = prep indices + 1).
  - Invoke `sub_crest.sh <crest.walltime_hours> input.xyz` from the work
    directory; for `_mecp` labels additionally append `--cinp .xcontrol`
    to the wrapper args. The wrapper writes its own PBS script
    (`CRESTJOB_input_<pid>.sh`) and submits via `qsub`; we capture stdout
    to record the PBS job id. NPROC=7 (experimentally optimal on
    MetaCentrum), method (GFN2), and ewin (6 kcal/mol) all come from the
    wrapper plus CREST defaults and are owned by `sub_crest.sh`. No other
    CREST flags are threaded from Python.
- `collect`, per label: parse `crest_conformers.xyz` (multi-frame XYZ)
  and `crest.energies`. Take up to
  `ensemble.max_conformers_per_diastereomer` lowest-energy conformers (no
  extra ewin filter; CREST already applied its own). Write survivors as
  `crest/{label}/filtered/conf_{i}.xyz` for downstream stages.
- Output: per label, a list of filtered conformer XYZ paths and CREST
  energies. The `_mecp` labels' conformers carry the constraint
  implicitly; downstream DFT does not need to re-impose it because the
  geometries are already at the MECP-mimic distance.

### 10.5 dft_sp (PBS, one job per surviving conformer per label)

- Input: filtered CREST conformers from each of the four labels
  (`anti_min`, `syn_min`, `anti_mecp`, `syn_mecp`). The `_mecp` geometries
  carry the C-O distance constraint implicitly (set by constrained CREST);
  the `_min` geometries are unconstrained ground-state minima.
- Action: ORCA single-point with two-stage protocol:
  1. r2SCAN-3c (fast, sanity check)
  2. ωB97X-D3BJ / def2-TZVP with SMD on the same geometry
  Single ORCA input file with two `! ...` lines and a `%scf` block, or two
  separate jobs — implementer's choice; the manifest must record both
  energies.
- `collect`: parse final SCF energies. Verify normal termination
  (`ORCA TERMINATED NORMALLY` in stdout).
- Output: r2SCAN-3c and ωB97X-D3BJ electronic energies in Hartree, per
  conformer, keyed by label.

### 10.6 dft_freq (PBS, optional, one job per surviving conformer per label)

- Skipped unless `options.thermal == true`.
- Input: filtered CREST conformers from each of the four labels.
- Action: ORCA frequency calculation at the cheap level
  (`dft.freq_method`, default r2SCAN-3c) at the CREST geometry. The
  geometry is *not* re-optimised before the freq calc -- we want thermal
  corrections at the geometry the d.r. is computed from, not at a
  different stationary point. For the `_mecp` labels this means imaginary
  frequencies are expected (a constrained geometry is not a true
  stationary point); for the `_min` labels they should be absent or rare.
  `dft_freq` records but does not reject on imaginary frequencies, and
  the thermal correction is computed treating low/imaginary modes via the
  standard quasi-harmonic approximation (Truhlar / Grimme rigid-rotor
  harmonic with frequencies below 100 cm⁻¹ raised to 100 cm⁻¹).
- `collect`: parse the thermochemistry block from ORCA output. Extract
  ZPE, thermal enthalpy correction, entropy, and Gibbs free energy
  correction at `temperature_kelvin`. Record `g_corr_hartree`. Combine with
  large-basis SP energy: `g_total = E(ωB97X-D3BJ) + g_corr`.
- Output: per-conformer thermal corrections and total G.

### 10.7 aggregate (local)

- Input: per-conformer energies from dft_sp (and optionally dft_freq) for
  all four labels.
- Action: compute the d.r. **twice** -- once from each ensemble pair --
  and report both.
  - **MECP / kinetic** prediction: use the `{anti_mecp, syn_mecp}`
    ensembles. ΔΔE‡ = E(anti_mecp) − E(syn_mecp); ΔΔG‡ analogous when
    thermal data is present. This is the primary number; it is the
    pipeline's headline d.r. and is printed by the CLI when `--verbose`
    is not set.
  - **Ground-state / thermodynamic** prediction: use the
    `{anti_min, syn_min}` ensembles. ΔΔE = E(anti_min) − E(syn_min);
    ΔΔG analogous. Reported alongside the MECP prediction; intended for
    cross-checks and downstream descriptor work, not as a substitute for
    the MECP prediction (the ground-state energy landscape is too flat
    to give a reliable d.r. on its own -- see §1).
  - For each ensemble pair, both ensemble treatments are computed:
    - **Lowest-energy**: pick the conformer with the lowest E (and lowest
      G, independently -- the lowest-E and lowest-G conformer may differ).
    - **Boltzmann**: compute Boltzmann-weighted average energy / free
      energy over conformers within `ensemble.energy_window_kj_mol` of
      the minimum. Boltzmann average:
      ⟨E⟩ = Σ Eᵢ exp(−Eᵢ/RT) / Σ exp(−Eᵢ/RT).
  - In all cases:
      d.r.(anti:syn) = exp(−ΔΔ / RT)   at `temperature_kelvin`.
- Output: write `result.json` per the schema in §7. Both
  `predictions.mecp` and `predictions.ground_state` are populated.

---

## 11. CLI

```
predict_dr.py predict <SMILES>
    --workspace PATH          (required) run directory
    --molecule-id ID          default: hash of canonical SMILES
    --config PATH             default: config/default.yaml
    --thermal / --no-thermal  default: --no-thermal
    --verbose                 echo full result.json fields
    --json                    dump result.json to stdout

predict_dr.py status <workspace>/<molecule_id>
    Print manifest summary: which stages done, which pending,
    PBS job IDs, last update time.

predict_dr.py resume <workspace>/<molecule_id>
    Re-enter the orchestrator loop on an existing manifest. Used when the
    OVEN orchestrator job died and the user re-submits.
```

The OVEN orchestrator PBS script (`pbs_orchestrator.j2`) just calls
`predict_dr.py predict ...` (or `resume` on retry).

---

## 12. Failure handling

- A stage that returns `failed` halts the orchestrator. No automatic retry.
- The user inspects `manifest.json` and the relevant stage's stdout/stderr,
  fixes the underlying issue (recompile xTB / extend walltime / fix input
  geometry), edits the manifest to set the stage back to `pending`
  (`tools/resubmit_failed.py` does this safely), and re-submits the
  orchestrator.
- This is deliberate: silent retries hide cluster pathologies. We want
  failures loud.
- The one exception is "PBS job not found in qstat and no output file" —
  this can mean the job was killed before producing output, or it never
  ran. The orchestrator treats this as `failed`, not as a reason to
  re-submit automatically.

---

## 13. Testing strategy

- **Unit tests** for `io_utils` (XYZ round-trip, ORCA parser on canned
  outputs, manifest read/write atomicity), `pbs_utils` (template rendering,
  qsub mock, qstat parsing), and `aggregate` (Boltzmann math, d.r.
  conversion — exact numerical results on small synthetic energy sets).
- **Integration test for the local subset** (prep → mm → aggregate with
  fake DFT energies injected): runs in seconds, verifies the orchestrator
  loop and manifest progression without touching the cluster.
- **One real end-to-end test**: an unsubstituted spiropyran (BIPS) on
  MetaCentrum, all stages, manual inspection of every output. This is
  the validation milestone before the pipeline is declared usable.

No mocking of ORCA / CREST / xTB themselves; their outputs are too
chemistry-specific to fake reliably. The unit tests use canned real
outputs from prior runs.

---

## 14. Out of scope for v1

- Batch / multi-molecule orchestration (handled later by a thin wrapper
  that submits one orchestrator job per SMILES)
- Active learning / acquisition functions
- Wavefunction analysis (Multiwfn) and descriptor calculation (Mordred,
  morfeus) — re-introduced only when ML returns
- LC property prediction
- Macrocyclic / bridged systems with strongly multi-conformer ensembles
  per diastereomer (the conformer-list data structure is in place;
  the constraint logic may need extending)
- Web tool / GUI

These are explicitly deferred. The v1 pipeline is a deterministic
SMILES → d.r. function that other components will eventually call.
