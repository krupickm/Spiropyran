# CLAUDE.md — Instructions for Coding Agents

You are implementing a deterministic computational pipeline that takes a SMILES
string and predicts the diastereomeric ratio (d.r.) of a spiropyran ring-closure
product, by running CREST + xTB + DFT calculations on the MetaCentrum HPC cluster.

**The architectural specification is `project.md`. Read it before writing or
modifying any code.** This file (`CLAUDE.md`) is about *how to work*; `project.md`
is about *what to build*. If the two ever conflict, `project.md` wins for
architecture; this file wins for working style.

---

## 1. Read this first

The user is Dr. Martin Krupička, a computational chemist. He knows the
chemistry and the HPC environment. He does not need explanations of basic
quantum chemistry, RDKit usage, or PBS scripting. He does need:

- Clean, debuggable Python that he can read and modify.
- Honest reporting of what works, what doesn't, and what was guessed.
- Direct technical communication. No filler, no hedging.

When you complete a task, say what you did and what's left. Do not summarise
in marketing language. "Implemented `io_utils.read_xyz`, tested round-trip on
3 canned files, did NOT yet handle the multi-frame case" is the right tone.

---

## 2. Scope boundaries

### You ARE responsible for

- Pure Python code under `spiropyran_dr/`.
- Input file generation (xTB inputs, ORCA inputs, PBS scripts via Jinja2
  templates).
- Output file parsing (CREST conformer files, xTB output, ORCA output).
- Manifest read / write / validation.
- The orchestrator state machine (`pipeline.py`).
- Unit tests for everything listed above.

### You are NOT responsible for

- Installing or compiling xTB, CREST, ORCA, or Multiwfn.
- Loading MetaCentrum modules (the PBS scripts do `module add ...`; the
  Python code never invokes `module`).
- Running real DFT or CREST calculations during development. Use canned
  outputs from `tests/data/` for parser tests.
- Choosing scientific parameters (DFT functional, MECP distance, ewin).
  All of these come from `config/default.yaml`. If a parameter is missing
  from config, ask before hard-coding.

If a task requires running real cluster calculations, stop and tell Martin —
he runs them himself and reports the results back.

---

## 3. Core principles

### 3.1 Pure Python, no orchestration frameworks

No ASE, no xtb-python, no Snakemake, no Airflow, no Prefect, no Nextflow.
Just Python standard library + RDKit + a YAML reader + Jinja2 + pytest.
Subprocess for `qsub` / `qstat`. Plain file I/O for everything else.

The reason: this is a PBS-based pipeline. ASE and xtb-python are designed
for in-process execution (you call xtb as a Python function and get a
result back). They do not match a "submit job, wait for file to appear,
parse output" workflow. Use them and you'll fight the framework.

### 3.2 File-based state, single source of truth

`manifest.json` per molecule is the only persistent state. The orchestrator
process holds nothing in memory across stages that isn't on disk. If the
process dies and is re-submitted, it must resume correctly from the manifest
alone.

This means:
- Every stage update is followed by a manifest write.
- Manifest writes are atomic (write to `manifest.json.tmp`, fsync, rename).
- Never hold a manifest reference across an `await` or `sleep` without
  re-reading after.

### 3.3 Stages are uniform and isolated

Every stage module under `stages/` exposes exactly `is_ready`, `submit`, and
`collect`. See `project.md` §4 for signatures. Stages do not import from
each other. Stages do not read or write `manifest.json` directly — they
receive the dict and return a dict update.

If a stage needs information from an earlier stage, it reads it from the
manifest dict it was given, not from the filesystem (except for the actual
chemistry files: XYZ, ORCA outputs, etc.).

### 3.4 Conformers are always lists

Even when there is one conformer per diastereomer, the data structure is
`outputs.anti = [conf_0]`, not `outputs.anti = conf_0`. This costs nothing
now and prevents a refactor when macrocyclic systems with many conformers
arrive.

### 3.5 Errors loud, not silent

A stage that fails returns `status: 'failed'` with a `failure_reason`. The
orchestrator stops. Do not implement automatic retries. Do not catch and
swallow exceptions to keep the pipeline limping. If something is wrong,
Martin needs to see it.

The one exception is recoverable PBS quirks (e.g. `qstat` transient errors)
— these can be retried with backoff inside `pbs_utils`, but the failure
mode of "the job actually died" must propagate.

---

## 4. Implementation order

Build in this sequence. Each step ends with something runnable, even if
later steps replace stubs with real chemistry.

1. **`io_utils.py` + `config_utils.py`** — manifest read/write (atomic),
   config load + hash, XYZ multi-frame I/O. Unit tests for everything.
2. **`pbs_utils.py`** — template render, `qsub` wrapper, `qstat` parser,
   job-finished detection. Unit tests with mocked `subprocess`.
3. **`pipeline.py` skeleton + `stages/base.py` + stub stages** — the main
   loop, with every stage replaced by a no-op that just marks itself
   `done`. End-to-end run from SMILES through fake stages to a fake
   `result.json`.
4. **`stages/prep.py` + `stages/mm.py`** — first real chemistry. Local
   only, no PBS. RDKit is enough.
5. **`stages/crest_stage.py`** — first real PBS stage. Template + parser.
6. **`stages/xtb_stage.py`** — TS-mimetic constrained opt. The
   intellectually critical stage; verify the constraint syntax against
   xTB documentation, do not guess.
7. **`stages/dft_sp_stage.py`** — ORCA two-stage SP.
8. **`stages/dft_freq_stage.py`** — optional thermal stage.
9. **`stages/aggregate.py`** — Boltzmann math + result.json.
10. **`cli.py`** — `predict`, `status`, `resume` subcommands.

Commit after each step. Each step's unit tests must pass before moving on.

---

## 5. Coding conventions

- Python 3.11+. Type hints everywhere. `from __future__ import annotations`
  at the top of every module.
- Use `pathlib.Path`, not string paths.
- `pydantic` for the manifest schema if it pulls its weight; otherwise
  plain dataclasses + jsonschema. Don't add it just for "validation".
- No global state. Pass `config` and `workspace` as arguments.
- Logging via `logging`, not `print`. The orchestrator configures a
  file handler on `orchestrator.log`.
- Functions under ~50 lines. If a function is longer, it's probably
  doing two things.
- No async. The orchestrator is sequential; sleeping is fine because
  it's running inside a long-walltime PBS job.

### 5.1 What "good code" means here

The reader is a chemist who will need to debug this when a CREST job
produces an unexpected conformer ordering or an ORCA parser misses a
new release's output format. Optimise for *traceability*, not cleverness:

- Explicit names. `n_conformers_anti` not `n_a`. `co_distance_final_ang` not
  `d_co`.
- Stage logic visible at the top level of each `submit` / `collect`
  function. If parsing logic is complex, it's a helper in the same file —
  not a new abstraction layer.
- Small, focused parsers. `parse_orca_scf_energy(output_text) -> float` is
  better than a 200-line `OrcaOutput` class that lazily computes everything.

---

## 6. Testing

- `pytest` for everything. Test discovery in `tests/`.
- Unit tests must not require: a network, the cluster, or external binaries.
- Parser tests use canned real outputs in `tests/data/` (small files,
  committed to the repo). When a new output format appears, capture a
  representative sample and add a test.
- Integration test (`tests/test_integration.py`) runs prep → mm → fake DFT
  energies → aggregate, in a temp directory, in seconds. This is the
  smoke test that the state machine works.
- Real cluster runs are not unit tests. They are validation milestones,
  performed by Martin manually, and their results are recorded in a
  `validation/` directory with a short README per molecule run.

---

## 7. Working with the user

- **Ask before you assume.** If a stage requires a parameter that isn't in
  `project.md` or `config/default.yaml`, ask. Don't invent a default.
- **Don't rewrite chemistry decisions.** The MECP distance, the DFT
  functional, the ewin window, the choice of SMD acetonitrile — these
  are settled (see project memory and references). If you think one is
  wrong, say so once, with a citation, and then implement what the spec
  says.
- **Quote the spec.** When implementing a stage, the docstring should
  cite the relevant section of `project.md` (e.g. "Implements §10.4 of
  project.md."). This makes drift between code and spec visible.
- **Surface trade-offs.** When choosing between two approaches with real
  consequences (e.g. "one ORCA input file with two methods" vs "two
  separate jobs"), describe both in one or two lines and pick. Don't
  hide the choice.

---

## 8. Things that look reasonable but are wrong

A short list of mistakes Claude Code has historically made on this kind
of project. Avoid these.

- **Using ASE / xtb-python.** They don't fit the PBS workflow. See §3.1.
- **Hand-parsing ORCA output with regex when cclib exists.** cclib is the
  right tool; use it. But — only for what cclib actually parses well
  (energies, geometries, frequencies). For things cclib doesn't expose
  cleanly (the thermochemistry block in particular), small custom parsers
  are fine.
- **Adding a database "for queryability".** No. Manifest + result.json on
  disk. Cross-molecule queries are a separate `tools/index.py` script
  that walks the runs/ tree and builds a DataFrame on demand.
- **Inventing a stage that "validates the geometry" or "checks
  reasonableness" without a defined criterion.** Either the constraint
  was satisfied (numeric tolerance) or it wasn't. Either ORCA terminated
  normally (string match) or it didn't. Don't add fuzzy validation.
- **Refactoring the stage interface "for elegance".** The interface is
  three functions because that maps cleanly onto the orchestrator's
  state machine. More abstraction = harder to debug.
- **Computing d.r. with thermal corrections and silently using the same
  number for the electronic-only result.** The two predictions are
  separate; both are reported; neither replaces the other.
- **Auto-retrying failed PBS jobs.** Failures are loud. See §3.5.
- **Hard-coding `acetonitrile`, `3.4`, `r2SCAN-3c` anywhere outside
  `config/default.yaml`.** Code reads config; code does not contain
  chemistry constants.

---

## 9. When to stop and ask

Stop and ask Martin if:

- A required parameter is missing from `config/default.yaml` and isn't
  obvious from `project.md`.
- The spec is ambiguous on a stage interface or output format.
- A real cluster run is needed to verify a parser or template (Martin
  runs these, not you).
- You believe the spec is wrong. Say so once, briefly. If he disagrees,
  implement the spec.

Don't stop and ask for:

- Cosmetic choices (variable names, whether to break a function, whether
  to inline a helper). Just make the call.
- Adding tests. Always add them.
- Adding type hints, docstrings, log statements. Always add them.

---

## 10. Reference: file layout

See `project.md` §6 (workspace layout) and §9 (module layout) for the
authoritative structure. Do not invent new top-level directories.
