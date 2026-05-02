"""Stage 2: MM conformer generation, geometric anti/syn labelling, RMSD clustering.

See project.md section 10.2 for the spec.

Design choices worth revisiting later:

- Anti/syn convention: signed dihedral
  ``chromene_O - C_spiro - indoline_N - indoline_anchor`` where
  ``indoline_anchor`` is the unique indoline-ring atom bonded to the
  indoline N other than the spiro carbon (the aromatic C in BIPS).
  Positive sign => "anti"; negative => "syn". The choice is arbitrary
  but deterministic, and chirality inversion at the spiro carbon flips
  the sign cleanly -- which is exactly what we need. Project molecules
  are achiral or racemic, so we pick one enantiomer of the closed
  product as the "anti" reference arbitrarily and let the dihedral sign
  decide per conformer. This convention lets BIPS (gem-dimethyl,
  symmetric) split correctly because ETKDG samples both spiro
  enantiomers from the connectivity-only canonical SMILES.
  Alternatives considered: (a) signed plane-displacement of chromene-O
  vs. a "big" gem-C substituent -- rejected because it depends on
  picking a "big" substituent, which is ambiguous for BIPS and made the
  labeller invariant under spiro chirality flip; (b) improper dihedral
  centred on the spiro C -- equivalent in information content but less
  intuitive to reason about than a real chain dihedral.
- RMSD clustering: greedy / exemplar-style with rdMolAlign best-RMS over
  symmetry-equivalent atom mappings. Chosen over Butina because it directly
  yields "energy minima, deduplicated"; Butina would need a separate step
  to pick representatives. Factored into cluster_by_rmsd() so the algorithm
  can be swapped (Butina, hierarchical, ML-based) without touching the
  orchestration code in submit().
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdMolTransforms

from spiropyran_dr.io_utils import atomic_write_json, write_xyz

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SMARTS_PATH = PACKAGE_ROOT / "config" / "smarts.yaml"

Label = Literal["anti", "syn"]


class MMError(ValueError):
    pass


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# -- topology -------------------------------------------------------------


def indoline_ring_atom_indices(
    mol: Chem.Mol, spiro_idx: int, indoline_n_idx: int
) -> tuple[int, ...]:
    """Return the 5-ring atom indices containing both the spiro C and indoline N.

    Uses RDKit's RingInfo rather than a brittle 5-atom SMARTS because we
    already have both anchor atoms identified by single-atom SMARTS roles.
    """
    rings = mol.GetRingInfo().AtomRings()
    candidates = [
        r for r in rings if len(r) == 5 and spiro_idx in r and indoline_n_idx in r
    ]
    if not candidates:
        raise MMError(
            f"no 5-ring contains both spiro_carbon ({spiro_idx}) "
            f"and indoline_nitrogen ({indoline_n_idx})"
        )
    if len(candidates) > 1:
        raise MMError(
            f"multiple 5-rings contain spiro_carbon and indoline_nitrogen: "
            f"{candidates!r}"
        )
    return tuple(candidates[0])


def indoline_anchor_atom(
    mol: Chem.Mol,
    indoline_ring: tuple[int, ...],
    spiro_idx: int,
    indoline_n_idx: int,
) -> int:
    """Return the unique indoline-ring atom bonded to N other than spiro_C.

    For closed BIPS-family scaffolds this is the aromatic carbon at the
    indoline-benzene fusion, on the nitrogen side. It serves as the fourth
    atom of the labelling dihedral (chromene_O - C_spiro - N - anchor).
    """
    n_atom = mol.GetAtomWithIdx(indoline_n_idx)
    candidates = [
        nb.GetIdx()
        for nb in n_atom.GetNeighbors()
        if nb.GetIdx() in indoline_ring and nb.GetIdx() != spiro_idx
    ]
    if len(candidates) != 1:
        raise MMError(
            f"expected exactly 1 indoline anchor for N {indoline_n_idx}, "
            f"got {candidates!r}"
        )
    return candidates[0]


# -- geometric labelling --------------------------------------------------


def label_conformer(
    mol: Chem.Mol,
    conf_id: int,
    indoline_ring: tuple[int, ...],
    spiro_idx: int,
    indoline_n_idx: int,
    chromene_o_idx: int,
) -> Label:
    """Assign 'anti' or 'syn' from the signed labelling dihedral.

    Dihedral chain: chromene_O - C_spiro - indoline_N - indoline_anchor.
    Sign convention: positive => 'anti', negative => 'syn'. Arbitrary but
    deterministic; chirality inversion at the spiro carbon flips the sign,
    which is what makes BIPS split into two equally-populated labels even
    though it has no asymmetric centre. See module docstring.
    """
    anchor = indoline_anchor_atom(mol, indoline_ring, spiro_idx, indoline_n_idx)
    conf = mol.GetConformer(conf_id)
    angle_deg = rdMolTransforms.GetDihedralDeg(
        conf, chromene_o_idx, spiro_idx, indoline_n_idx, anchor
    )
    return "anti" if angle_deg > 0.0 else "syn"


# -- embedding ------------------------------------------------------------


def _embed_and_optimise(
    mol_with_h: Chem.Mol, n_embed: int, mmff_iters: int, seed: int
) -> list[tuple[int, float]]:
    """Embed N conformers (ETKDGv3), MMFF94-optimise, return [(conf_id, energy)].

    Energy is in kcal/mol from MMFF94. Conformers that fail to converge or
    fail energy evaluation are dropped. Output is sorted by energy ascending.
    """
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    conf_ids = list(AllChem.EmbedMultipleConfs(mol_with_h, numConfs=n_embed, params=params))
    if not conf_ids:
        return []
    results = AllChem.MMFFOptimizeMoleculeConfs(mol_with_h, maxIters=mmff_iters)
    out: list[tuple[int, float]] = []
    for cid, (status, energy) in zip(conf_ids, results):
        # status == 0 means converged. We accept non-converged conformers
        # only if they nevertheless produced a finite energy; this matches
        # standard practice for MM screening where partial relaxation is
        # often good enough for the next stage.
        if energy is None or not np.isfinite(energy):
            continue
        out.append((int(cid), float(energy)))
        _ = status  # currently informational only
    out.sort(key=lambda p: p[1])
    return out


# -- clustering -----------------------------------------------------------


def cluster_by_rmsd(
    mol: Chem.Mol,
    conf_energy_pairs: list[tuple[int, float]],
    rmsd_threshold_ang: float,
    max_keep: int,
) -> list[tuple[int, float]]:
    """Greedy energy-ordered RMSD deduplication.

    Iterates conformers in input order (caller is responsible for sorting
    by energy ascending). Accepts a conformer if its best symmetry-aware
    heavy-atom RMSD to every already-accepted conformer is >= threshold.
    Stops at max_keep. See module docstring for the rationale for greedy
    over Butina.
    """
    kept: list[tuple[int, float]] = []
    for cid, energy in conf_energy_pairs:
        if len(kept) >= max_keep:
            break
        too_close = False
        for kid, _ in kept:
            rms = rdMolAlign.GetBestRMS(mol, mol, prbId=cid, refId=kid)
            if rms < rmsd_threshold_ang:
                too_close = True
                break
        if not too_close:
            kept.append((cid, energy))
    return kept


# -- stage interface ------------------------------------------------------


def is_ready(manifest: dict[str, Any], workspace: Path) -> bool:
    prep_stage = (manifest.get("stages") or {}).get("prep") or {}
    if prep_stage.get("status") != "done":
        return False
    outputs = prep_stage.get("outputs") or {}
    required = (
        "smiles_canonical",
        "spiro_carbon_idx",
        "chromene_oxygen_idx",
        "indoline_nitrogen_idx",
        "gem_carbon_idx",
    )
    return all(k in outputs for k in required)


def submit(
    manifest: dict[str, Any], workspace: Path, config: dict[str, Any]
) -> dict[str, Any]:
    started = _now_iso()
    prep_outputs = manifest["stages"]["prep"]["outputs"]
    smiles = prep_outputs["smiles_canonical"]
    spiro_idx = prep_outputs["spiro_carbon_idx"]
    chromene_o_idx = prep_outputs["chromene_oxygen_idx"]
    indoline_n_idx = prep_outputs["indoline_nitrogen_idx"]

    mm_cfg = config.get("mm") or {}
    n_embed = int(mm_cfg.get("n_embed", 50))
    mmff_iters = int(mm_cfg.get("mmff_max_iters", 200))
    rmsd_thresh = float(mm_cfg.get("rmsd_threshold_angstrom", 0.5))
    seed = int(mm_cfg.get("random_seed", 42))
    max_per = int((config.get("ensemble") or {}).get("max_conformers_per_diastereomer", 20))

    mol_h = Chem.AddHs(Chem.MolFromSmiles(smiles))
    pairs = _embed_and_optimise(mol_h, n_embed=n_embed, mmff_iters=mmff_iters, seed=seed)
    if not pairs:
        return {
            "status": "failed",
            "started_at": started,
            "finished_at": _now_iso(),
            "failure_reason": f"ETKDG produced no conformers from SMILES {smiles!r}",
        }

    try:
        ring = indoline_ring_atom_indices(mol_h, spiro_idx, indoline_n_idx)
        # Validate the anchor exists; the labeller would re-discover it per
        # conformer, but failing fast here keeps error messages crisp.
        indoline_anchor_atom(mol_h, ring, spiro_idx, indoline_n_idx)
    except MMError as exc:
        return {
            "status": "failed",
            "started_at": started,
            "finished_at": _now_iso(),
            "failure_reason": str(exc),
        }

    labelled: dict[Label, list[tuple[int, float]]] = {"anti": [], "syn": []}
    for cid, energy in pairs:
        lab = label_conformer(
            mol_h, cid, ring, spiro_idx, indoline_n_idx, chromene_o_idx
        )
        labelled[lab].append((cid, energy))

    # Already energy-sorted because pairs is sorted; dedupe within each label.
    kept: dict[Label, list[tuple[int, float]]] = {
        "anti": cluster_by_rmsd(mol_h, labelled["anti"], rmsd_thresh, max_per),
        "syn": cluster_by_rmsd(mol_h, labelled["syn"], rmsd_thresh, max_per),
    }

    if not kept["anti"] or not kept["syn"]:
        missing = [lab for lab in ("anti", "syn") if not kept[lab]]
        return {
            "status": "failed",
            "started_at": started,
            "finished_at": _now_iso(),
            "failure_reason": (
                f"MM produced no conformers for diastereomer(s) {missing}; "
                f"anti={len(kept['anti'])}, syn={len(kept['syn'])}, "
                f"total_embedded={len(pairs)}"
            ),
        }

    # Write XYZ files and the sidecar JSON.
    outputs_per_label: dict[Label, list[dict[str, Any]]] = {"anti": [], "syn": []}
    for label in ("anti", "syn"):
        label_dir_rel = Path("mm") / label
        label_dir_abs = workspace / label_dir_rel
        label_dir_abs.mkdir(parents=True, exist_ok=True)
        for new_id, (cid, energy) in enumerate(kept[label]):
            xyz_rel = label_dir_rel / f"conf_{new_id}.xyz"
            write_xyz(
                workspace / xyz_rel,
                mol_h,
                conf_id=cid,
                comment=(
                    f"label={label} conf_id={new_id} mmff_kcal_mol={energy:.6f}"
                ),
            )
            outputs_per_label[label].append(
                {
                    "conf_id": new_id,
                    "embed_id": cid,
                    "xyz": str(xyz_rel).replace("\\", "/"),
                    "mmff_energy_kcal_mol": energy,
                    "label": label,
                }
            )

    sidecar_rel = Path("mm") / "conformers.json"
    atomic_write_json(
        workspace / sidecar_rel,
        {
            "anti": outputs_per_label["anti"],
            "syn": outputs_per_label["syn"],
            "config": {
                "n_embed": n_embed,
                "mmff_max_iters": mmff_iters,
                "rmsd_threshold_angstrom": rmsd_thresh,
                "max_conformers_per_diastereomer": max_per,
                "random_seed": seed,
            },
        },
    )

    return {
        "status": "done",
        "started_at": started,
        "finished_at": _now_iso(),
        "outputs": {
            "n_conformers_anti": len(outputs_per_label["anti"]),
            "n_conformers_syn": len(outputs_per_label["syn"]),
            "anti_xyz_dir": "mm/anti",
            "syn_xyz_dir": "mm/syn",
            "anti": outputs_per_label["anti"],
            "syn": outputs_per_label["syn"],
            "sidecar_path": str(sidecar_rel).replace("\\", "/"),
        },
    }


def collect(
    manifest: dict[str, Any], workspace: Path, config: dict[str, Any]
) -> dict[str, Any]:
    return {}
