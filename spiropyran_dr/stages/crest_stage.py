"""Stage 4: CREST conformational sampling — 4 parallel jobs.

Labels: anti_min, syn_min, anti_mecp, syn_mecp.

_min labels run unconstrained CREST, seeded from the lowest MM conformer.
_mecp labels run under a C-O distance constraint (written as .xcontrol),
seeded from the xtb_constr output for that base diastereomer.

After collect(), conformers are re-classified by the signed labelling
dihedral chromene_O - C_spiro - indoline_N - anchor (same convention as
mm.label_conformer). Conformers from both jobs within a pair type (min or
mecp) are pooled, sorted by energy, and split according to the geometric
label. This corrects for CREST breaking the spiro C-O bond and sampling
across the syn/anti barrier.

See project.md section 10.4.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from rdkit import Chem
from rdkit.Chem import AllChem

from spiropyran_dr.io_utils import (
    parse_crest_energy_from_comment,
    read_xyz_multiframe,
    write_xcontrol_distance_constraint,
    write_xyz_from_arrays,
)
from spiropyran_dr.pbs_utils import (
    PBSSubmitError,
    submit_via_script,
    write_jobid,
)
from spiropyran_dr.stages.mm import indoline_ring_atom_indices, label_conformer

LABELS: tuple[str, str, str, str] = ("anti_min", "syn_min", "anti_mecp", "syn_mecp")

# Pairs of (anti_label, syn_label) processed together during geometric
# re-classification.  min-seeded jobs are pooled with each other; mecp-seeded
# jobs are pooled with each other.
_PAIRS: tuple[tuple[str, str], ...] = (
    ("anti_min", "syn_min"),
    ("anti_mecp", "syn_mecp"),
)

# Each element: (energy_hartree, symbols, coords)
_Pool = list[tuple[float, list[str], list[tuple[float, float, float]]]]


def _is_mecp(label: str) -> bool:
    return label.endswith("_mecp")


def _base_label(label: str) -> str:
    return label.removesuffix("_min").removesuffix("_mecp")


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _label_dir(workspace: Path, label: str) -> Path:
    return workspace / "crest" / label


def _lowest_energy_mm_xyz(mm_outputs: dict[str, Any], base: str) -> str | None:
    """Return the relative xyz path of the lowest-energy MM conformer for a base label."""
    entries = mm_outputs.get(base) or []
    if not entries:
        return None
    return entries[0]["xyz"]


# -- geometric labeller ---------------------------------------------------


def _build_geo_labeller(
    manifest: dict[str, Any],
) -> Callable[[list[str], list[tuple[float, float, float]]], str] | None:
    """Build a callable (symbols, coords) -> "anti"|"syn" from prep outputs.

    Uses mm.label_conformer with the same dihedral convention (chromene_O -
    C_spiro - indoline_N - anchor, positive => anti). Returns None when the
    required prep outputs are absent, in which case callers fall back to
    assigning geo_label from the job name.
    """
    prep_out = (manifest.get("stages") or {}).get("prep", {}).get("outputs") or {}
    smiles = prep_out.get("smiles_canonical")
    spiro_idx = prep_out.get("spiro_carbon_idx")
    chromene_o_idx = prep_out.get("chromene_oxygen_idx")
    indoline_n_idx = prep_out.get("indoline_nitrogen_idx")

    if any(v is None for v in (smiles, spiro_idx, chromene_o_idx, indoline_n_idx)):
        return None

    spiro_idx = int(spiro_idx)
    chromene_o_idx = int(chromene_o_idx)
    indoline_n_idx = int(indoline_n_idx)

    mol_tpl = AllChem.AddHs(Chem.MolFromSmiles(smiles))
    indoline_ring = indoline_ring_atom_indices(mol_tpl, spiro_idx, indoline_n_idx)
    n_atoms = mol_tpl.GetNumAtoms()

    def _label(
        symbols: list[str], coords: list[tuple[float, float, float]]
    ) -> str:
        conf = Chem.Conformer(n_atoms)
        for i, (x, y, z) in enumerate(coords):
            conf.SetAtomPosition(i, (x, y, z))
        mol = Chem.RWMol(mol_tpl)
        cid = mol.AddConformer(conf, assignId=True)
        return label_conformer(
            mol.GetMol(),
            cid,
            indoline_ring,
            spiro_idx,
            indoline_n_idx,
            chromene_o_idx,
        )

    return _label


# -- output helpers -------------------------------------------------------


def _write_label_entries(
    pool: _Pool,
    label: str,
    geo_lbl: str,
    max_per: int,
    workspace: Path,
    outputs: dict[str, Any],
) -> None:
    """Write filtered XYZ files and populate *outputs* for one output label."""
    kept = pool[:max_per]
    e_min = kept[0][0] if kept else 0.0
    filtered_dir = _label_dir(workspace, label) / "filtered"
    filtered_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for new_id, (energy, sym, coords) in enumerate(kept):
        xyz_rel = Path("crest") / label / "filtered" / f"conf_{new_id}.xyz"
        comment = f"label={label} conf_id={new_id} crest_energy_hartree={energy:.8f}"
        write_xyz_from_arrays(workspace / xyz_rel, sym, coords, comment)
        entries.append(
            {
                "conf_id": new_id,
                "xyz": str(xyz_rel).replace("\\", "/"),
                "energy_hartree": energy,
                # 627.5094740631 kcal/mol per Hartree (CODATA)
                "relative_energy_kcal_mol": (energy - e_min) * 627.5094740631,
                "label": label,
                "geo_label": geo_lbl,
            }
        )
    outputs[label] = entries
    outputs[f"n_conformers_{label}"] = len(entries)
    outputs[f"{label}_xyz_dir"] = f"crest/{label}/filtered"


def _fill_outputs_geo(
    parsed: dict[str, list[tuple[list[str], list[tuple[float, float, float]], float]]],
    geo_labeller: Callable[[list[str], list[tuple[float, float, float]]], str],
    max_per: int,
    workspace: Path,
    outputs: dict[str, Any],
) -> None:
    """Pool within each pair type, re-classify by dihedral, split into output slots."""
    for anti_label, syn_label in _PAIRS:
        pool: _Pool = sorted(
            [
                (energy, sym, coords)
                for lbl in (anti_label, syn_label)
                for sym, coords, energy in parsed[lbl]
            ],
            key=lambda t: t[0],
        )

        anti_pool: _Pool = []
        syn_pool: _Pool = []
        for energy, sym, coords in pool:
            if geo_labeller(sym, coords) == "anti":
                anti_pool.append((energy, sym, coords))
            else:
                syn_pool.append((energy, sym, coords))

        _write_label_entries(anti_pool, anti_label, "anti", max_per, workspace, outputs)
        _write_label_entries(syn_pool, syn_label, "syn", max_per, workspace, outputs)


def _fill_outputs_by_job(
    parsed: dict[str, list[tuple[list[str], list[tuple[float, float, float]], float]]],
    max_per: int,
    workspace: Path,
    outputs: dict[str, Any],
) -> None:
    """Assign geo_label from job name (fallback when prep outputs are absent)."""
    for label in LABELS:
        geo_lbl = _base_label(label)
        pool: _Pool = sorted(
            [(energy, sym, coords) for sym, coords, energy in parsed[label]],
            key=lambda t: t[0],
        )
        _write_label_entries(pool, label, geo_lbl, max_per, workspace, outputs)


# -- stage interface ------------------------------------------------------


def is_ready(manifest: dict[str, Any], workspace: Path) -> bool:
    stages = manifest.get("stages") or {}

    xtb_stage = stages.get("xtb_constr") or {}
    if xtb_stage.get("status") != "done":
        return False
    xtb_outputs = xtb_stage.get("outputs") or {}
    if not xtb_outputs.get("anti") or not xtb_outputs.get("syn"):
        return False

    mm_stage = stages.get("mm") or {}
    if mm_stage.get("status") != "done":
        return False
    mm_outputs = mm_stage.get("outputs") or {}
    return (
        mm_outputs.get("n_conformers_anti", 0) >= 1
        and mm_outputs.get("n_conformers_syn", 0) >= 1
    )


def submit(
    manifest: dict[str, Any], workspace: Path, config: dict[str, Any]
) -> dict[str, Any]:
    started = _now_iso()
    mm_outputs = manifest["stages"]["mm"]["outputs"]
    xtb_outputs = manifest["stages"]["xtb_constr"]["outputs"]
    prep_outputs = (manifest.get("stages") or {}).get("prep", {}).get("outputs") or {}
    crest_cfg = config.get("crest") or {}
    mecp_cfg = config.get("mecp") or {}
    walltime_hours = int(crest_cfg["walltime_hours"])
    script_path = Path(str(crest_cfg["script_path"])).expanduser()

    # Atom indices are required for writing .xcontrol on _mecp labels. Both
    # come from SMARTS detection in prep; their absence is a pipeline stopper.
    if (
        "spiro_carbon_idx" not in prep_outputs
        or "chromene_oxygen_idx" not in prep_outputs
    ):
        return {
            "status": "failed",
            "started_at": started,
            "finished_at": _now_iso(),
            "failure_reason": (
                "prep outputs missing spiro_carbon_idx or chromene_oxygen_idx; "
                "re-run prep before submitting CREST"
            ),
        }
    idx_a = int(prep_outputs["spiro_carbon_idx"])
    idx_b = int(prep_outputs["chromene_oxygen_idx"])

    pbs_job_ids: dict[str, str] = {}

    for label in LABELS:
        base = _base_label(label)

        if _is_mecp(label):
            entries = xtb_outputs.get(base) or []
            if not entries:
                return {
                    "status": "failed",
                    "started_at": started,
                    "finished_at": _now_iso(),
                    "failure_reason": f"xtb_constr stage has no output for base label {base!r}",
                }
            src_rel = entries[0]["xyz"]
        else:
            src_rel = _lowest_energy_mm_xyz(mm_outputs, base)
            if src_rel is None:
                return {
                    "status": "failed",
                    "started_at": started,
                    "finished_at": _now_iso(),
                    "failure_reason": f"mm stage produced no conformers for label {base!r}",
                }

        src_abs = workspace / src_rel
        if not src_abs.is_file():
            return {
                "status": "failed",
                "started_at": started,
                "finished_at": _now_iso(),
                "failure_reason": f"missing seed xyz: {src_abs}",
            }

        work_dir = _label_dir(workspace, label)
        work_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_abs, work_dir / "input.xyz")

        wrapper_args = [str(walltime_hours), "input.xyz"]

        if _is_mecp(label):
            write_xcontrol_distance_constraint(
                work_dir / ".xcontrol",
                idx_a,
                idx_b,
                float(mecp_cfg.get("c_o_distance_angstrom", 3.4)),
                float(mecp_cfg.get("constraint_force_constant", 1.0)),
            )
            wrapper_args += ["--cinp", ".xcontrol"]

        try:
            jobid, _ = submit_via_script(script_path, wrapper_args, cwd=work_dir)
        except PBSSubmitError as exc:
            return {
                "status": "failed",
                "started_at": started,
                "finished_at": _now_iso(),
                "failure_reason": f"sub_crest.sh failed for {label}: {exc}",
            }

        write_jobid(work_dir / "jobid", jobid)
        pbs_job_ids[label] = jobid

    return {
        "status": "submitted",
        "started_at": started,
        "submitted_at": started,
        "pbs_job_ids": pbs_job_ids,
    }


def collect(
    manifest: dict[str, Any], workspace: Path, config: dict[str, Any]
) -> dict[str, Any]:
    max_per = int(
        (config.get("ensemble") or {}).get("max_conformers_per_diastereomer", 20)
    )
    geo_labeller = _build_geo_labeller(manifest)

    # Parse all 4 jobs before classifying so we can pool across pairs.
    # CREST emits crest_conformers.xyz sorted lowest-first; absolute energy
    # (Hartree) is the first token of each frame's comment line.
    parsed: dict[
        str, list[tuple[list[str], list[tuple[float, float, float]], float]]
    ] = {}
    for label in LABELS:
        work_dir = _label_dir(workspace, label)
        confs_path = work_dir / "crest_conformers.xyz"
        if not confs_path.is_file():
            return {
                "status": "failed",
                "finished_at": _now_iso(),
                "failure_reason": f"CREST output missing for {label}: {confs_path}",
            }

        frames = read_xyz_multiframe(confs_path)
        if not frames:
            return {
                "status": "failed",
                "finished_at": _now_iso(),
                "failure_reason": f"CREST produced no conformers for {label}",
            }

        try:
            energies = [
                parse_crest_energy_from_comment(comment) for _, _, comment in frames
            ]
        except ValueError as exc:
            return {
                "status": "failed",
                "finished_at": _now_iso(),
                "failure_reason": (
                    f"could not parse absolute energy from a frame comment "
                    f"in {confs_path}: {exc}"
                ),
            }

        parsed[label] = [
            (sym, coords, energy)
            for (sym, coords, _), energy in zip(frames, energies)
        ]

    outputs: dict[str, Any] = {}
    if geo_labeller is not None:
        _fill_outputs_geo(parsed, geo_labeller, max_per, workspace, outputs)
    else:
        _fill_outputs_by_job(parsed, max_per, workspace, outputs)

    return {
        "status": "done",
        "finished_at": _now_iso(),
        "outputs": outputs,
    }
