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

The pipeline produces **two numbers**, both reported:

- **ΔΔE‡** — electronic energy difference at the constrained geometry (always computed)
- **ΔΔG‡** — Gibbs free energy difference including thermal corrections from
  vibrational frequencies at the constrained geometry (optional stage)

And **two ensemble treatments**, both reported:

- **Lowest-energy conformer** per diastereomer (primary number)
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
[3] crest           PBS        GFN2-xTB conformational sampling on closed ground
                               state, per diastereomer; ewin-filtered ensemble
    ▼
[4] xtb_constr      PBS        TS-mimetic: GFN2-xTB constrained opt with C-O
                               distance fixed at MECP value, per conformer
    ▼
[5] dft_sp          PBS        ORCA single point: r2SCAN-3c → ωB97X-D3BJ/def2-TZVP
                               with SMD, per constrained geometry
    ▼
[6] dft_freq        PBS        OPTIONAL: ORCA frequency calc at constrained
                               geometry → thermal corrections → G(298 K)
    ▼
[7] aggregate       local      Collapse conformer ensemble (lowest + Boltzmann)
                               per diastereomer, compute ΔΔE‡ / ΔΔG‡, report d.r.
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
- The expected output sentinel file exists in the stage's work directory
  (e.g. `crest_done` or `orca.out` with a normal-termination string).

The output-file check is authoritative for success; `qstat` is the running
indicator. After detecting finish, parse outputs and decide success vs failure
by checking the sentinel file's contents (e.g. `ORCA TERMINATED NORMALLY`).

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
      - For PBS stages: pbs_job_id, submitted_at, work_dir
      - For local stages: outputs (stage-specific dict), finished_at
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
                       "anti_xyz_dir": "mm/anti", "syn_xyz_dir": "mm/syn" } },
    "crest":       { "status": "running", "pbs_job_ids":
                     { "anti": "12345.meta-pbs", "syn": "12346.meta-pbs" },
                     "submitted_at": "..." },
    "xtb_constr":  { "status": "pending" },
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

Stages 3–6 operate per-conformer-per-diastereomer. The manifest records a list
of conformer entries inside each stage's `outputs`:

```json
"xtb_constr": {
  "status": "done",
  "outputs": {
    "anti": [
      { "conf_id": 0, "input": "xtb_constr/anti/conf_0/input.xyz",
        "output": "xtb_constr/anti/conf_0/xtbopt.xyz",
        "energy_hartree": -1234.567, "co_distance_final_ang": 3.402 },
      { "conf_id": 1, ... }
    ],
    "syn": [ ... ]
  }
}
```

Even when there is one conformer, this is a list. This costs nothing now and
saves a refactor when macrocyclic systems with multiple conformers per
diastereomer arrive.

### 5.3 Config hash

`config_hash` = sha256 over the canonical-JSON serialisation of the
chemistry-relevant config sections (`mecp`, `solvent`, `dft`, `crest`, `xtb`).
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
                ├── crest/
                │   ├── anti/
                │   │   ├── input.xyz
                │   │   ├── pbs.sh
                │   │   ├── jobid
                │   │   ├── crest_conformers.xyz
                │   │   └── crest.energies
                │   └── syn/ ...
                ├── xtb_constr/
                │   ├── anti/
                │   │   ├── conf_0/
                │   │   │   ├── input.xyz
                │   │   │   ├── xtb.inp
                │   │   │   ├── pbs.sh
                │   │   │   ├── jobid
                │   │   │   ├── xtbopt.xyz
                │   │   │   └── xtb.out
                │   │   └── conf_1/ ...
                │   └── syn/ ...
                ├── dft_sp/
                │   ├── anti/
                │   │   ├── conf_0/
                │   │   │   ├── orca.inp
                │   │   │   ├── pbs.sh
                │   │   │   ├── jobid
                │   │   │   └── orca.out
                │   │   └── conf_1/ ...
                │   └── syn/ ...
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
    "lowest_conformer": {
      "delta_e_kj_mol": 3.2,
      "dr_anti_syn_from_e": 0.78,
      "delta_g_kj_mol": 2.9,
      "dr_anti_syn_from_g": 0.76,
      "selected_conf": { "anti": 4, "syn": 1 }
    },
    "boltzmann": {
      "delta_e_kj_mol": 2.8,
      "dr_anti_syn_from_e": 0.76,
      "delta_g_kj_mol": 2.6,
      "dr_anti_syn_from_g": 0.74,
      "n_conformers_used": { "anti": 7, "syn": 9 }
    }
  },

  "thermal_included": true,

  "energies": {
    "anti": [
      { "conf_id": 0, "e_dft_hartree": -1234.567,
        "g_corr_hartree": 0.0421, "g_total_hartree": -1234.525 },
      ...
    ],
    "syn": [ ... ]
  },

  "config_hash": "sha256:...",
  "config_path": "config/default.yaml",
  "wall_time_seconds": {
    "crest": { "anti": 8421, "syn": 9123 },
    "xtb_constr": 1203,
    "dft_sp": 31200,
    "dft_freq": 18400
  }
}
```

If `--no-thermal`, `predictions.*.delta_g_kj_mol` and `dr_anti_syn_from_g` are
`null`, `thermal_included` is `false`, and `g_corr_hartree` / `g_total_hartree`
are absent.

---

## 8. Configuration

`config/default.yaml`. Everything chemistry-specific lives here. Code reads;
code does not decide.

```yaml
mecp:
  c_o_distance_angstrom: 3.4

temperature_kelvin: 298.15

crest:
  method: gfn2
  ewin_kcal_mol: 6.0
  threads: 16
  walltime: "24:00:00"

xtb_constr:
  method: gfn2
  walltime: "4:00:00"
  # constraint applied: distance, atoms identified by SMARTS-mapped indices

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
│   ├── pbs_crest.j2
│   ├── pbs_xtb_constrained.j2
│   ├── pbs_orca_sp.j2
│   └── pbs_orca_freq.j2
├── config/
│   ├── default.yaml
│   └── smarts.yaml
├── tools/
│   ├── inspect_manifest.py
│   └── resubmit_failed.py
├── tests/
│   ├── test_io_utils.py
│   ├── test_pbs_utils.py
│   ├── test_prep.py
│   ├── test_mm.py
│   └── test_aggregate.py
└── cli.py                   # entry point: predict_dr.py
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
  - `spiro_carbon_idx`, `chromene_oxygen_idx`.

### 10.2 mm (local)

- Input: SMILES + atom indices from prep.
- Action:
  - RDKit ETKDGv3 to embed N (~50) conformers of the closed spiropyran.
  - MMFF94 optimise.
  - For each conformer, measure the relevant geometric parameter
    (dihedral around the spiro centre, sign of out-of-plane displacement of
    the indoline nitrogen relative to the chromene plane) and label as
    `anti` or `syn`.
  - Cluster within each label by RMSD; keep up to
    `ensemble.max_conformers_per_diastereomer` lowest-energy per label.
  - Write `mm/{anti,syn}/conf_{i}.xyz`.
- Output: per-diastereomer lists of XYZ paths.
- Failure modes: ETKDG fails to generate either label (log + fail).

### 10.3 crest (PBS, two parallel jobs)

- Input: lowest-energy MM conformer per diastereomer (one xyz per submission).
- Action: render `pbs_crest.j2` for each diastereomer, `qsub`, record both
  job IDs.
- `collect`: parse `crest_conformers.xyz` (multi-frame XYZ) and
  `crest.energies`. Filter by `crest.ewin_kcal_mol`, take up to
  `ensemble.max_conformers_per_diastereomer` lowest-energy. Write filtered
  conformers as `crest/{anti,syn}/filtered/conf_{i}.xyz` for downstream stages.
- Output: filtered conformer XYZ paths and CREST energies per diastereomer.

### 10.4 xtb_constr (PBS, one job per conformer per diastereomer)

- Input: filtered CREST conformers.
- Action: for each conformer, write a constrained-optimisation xTB input that
  fixes the C–O distance at `mecp.c_o_distance_angstrom` between
  `spiro_carbon_idx` and `chromene_oxygen_idx`. Submit one PBS job per
  conformer (sized small; xTB is cheap).
- `collect`: parse `xtbopt.xyz` and final energy. Verify final C–O distance
  is within tolerance of the requested value (e.g. ±0.01 Å). Discard
  conformers where the constraint failed.
- Output: per-conformer optimised geometries and xTB energies.

### 10.5 dft_sp (PBS, one job per surviving conformer per diastereomer)

- Input: constrained-optimised geometries.
- Action: ORCA single-point with two-stage protocol:
  1. r2SCAN-3c (fast, sanity check)
  2. ωB97X-D3BJ / def2-TZVP with SMD on the same geometry
  Single ORCA input file with two `! ...` lines and a `%scf` block, or two
  separate jobs — implementer's choice; the manifest must record both
  energies.
- `collect`: parse final SCF energies. Verify normal termination
  (`ORCA TERMINATED NORMALLY` in stdout).
- Output: r2SCAN-3c and ωB97X-D3BJ electronic energies in Hartree.

### 10.6 dft_freq (PBS, optional, one job per surviving conformer per diastereomer)

- Skipped unless `options.thermal == true`.
- Input: constrained-optimised geometries.
- Action: ORCA frequency calculation at the cheap level
  (`dft.freq_method`, default r2SCAN-3c) at the constrained geometry. The
  geometry is *not* re-optimised before the freq calc — we want thermal
  corrections at the constrained TS-mimic, not at a different stationary
  point. This means imaginary frequencies are expected (the constrained
  geometry is not a true stationary point); `dft_freq` should record but not
  reject on imaginary frequencies, and the thermal correction is computed
  treating low/imaginary modes via the standard quasi-harmonic
  approximation (Truhlar / Grimme rigid-rotor harmonic with frequencies
  below 100 cm⁻¹ raised to 100 cm⁻¹).
- `collect`: parse the thermochemistry block from ORCA output. Extract
  ZPE, thermal enthalpy correction, entropy, and Gibbs free energy
  correction at `temperature_kelvin`. Record `g_corr_hartree`. Combine with
  large-basis SP energy: `g_total = E(ωB97X-D3BJ) + g_corr`.
- Output: per-conformer thermal corrections and total G.

### 10.7 aggregate (local)

- Input: per-conformer energies from dft_sp (and optionally dft_freq).
- Action: for each diastereomer, compute:
  - **Lowest-energy**: pick the conformer with the lowest E (and lowest G,
    independently — the lowest-E and lowest-G conformer may differ).
  - **Boltzmann**: compute Boltzmann-weighted average energy / free energy
    over conformers within `ensemble.energy_window_kj_mol` of the minimum.
    Boltzmann average: ⟨E⟩ = Σ Eᵢ exp(−Eᵢ/RT) / Σ exp(−Eᵢ/RT).
  - Compute ΔΔE‡ = E_anti − E_syn (and ΔΔG‡ if thermal stage was run).
  - d.r.(anti:syn) = exp(−ΔΔ‡ / RT) at `temperature_kelvin`.
- Output: write `result.json` per the schema in §7.

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
