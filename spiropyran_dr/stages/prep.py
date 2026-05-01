from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdkit import Chem

from spiropyran_dr.config_utils import load_smarts
from spiropyran_dr.io_utils import atomic_write_json

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SMARTS_PATH = PACKAGE_ROOT / "config" / "smarts.yaml"

MIN_HEAVY_ATOMS = 10


class PrepError(ValueError):
    pass


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def canonicalise(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol, canonical=True)


def parse_validated(smiles: str) -> Chem.Mol:
    canonical = canonicalise(smiles)
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        raise ValueError(f"failed to re-parse canonical SMILES: {canonical!r}")
    return mol


def find_atom_by_smarts(mol: Chem.Mol, smarts: str, role: str) -> int:
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        raise PrepError(f"invalid SMARTS for role {role!r}: {smarts!r}")
    matches = mol.GetSubstructMatches(pattern)
    if len(matches) == 0:
        raise PrepError(f"no match for atom role {role!r}")
    if len(matches) > 1:
        raise PrepError(f"ambiguous match for atom role {role!r}: {len(matches)} candidates")
    # Atom-role SMARTS query atom 0 by convention; downstream consumers want
    # the first matched atom (the spiro carbon for spiro_carbon, the oxygen
    # for chromene_oxygen).
    return matches[0][0]


def assign_spiro_stereo(mol: Chem.Mol, spiro_idx: int) -> tuple[Chem.Mol, str]:
    # We always pick one enantiomer at the spiro centre; the anti/syn
    # diastereomer split is geometric and assigned at the mm stage
    # (project.md section 10.1).
    mol_copy = Chem.RWMol(mol)
    atom = mol_copy.GetAtomWithIdx(spiro_idx)
    atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    finalised = mol_copy.GetMol()
    Chem.AssignStereochemistry(finalised, cleanIt=True, force=True)
    cip = finalised.GetAtomWithIdx(spiro_idx).GetPropsAsDict().get("_CIPCode", "")
    if cip not in ("R", "S"):
        raise PrepError(f"could not assign CIP code to spiro atom {spiro_idx}")
    return finalised, cip


def apply_smarts_filters(
    mol: Chem.Mol, required: list[str], forbidden: list[str]
) -> dict[str, Any]:
    required_missing: list[str] = []
    for smarts in required:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            raise PrepError(f"invalid required SMARTS: {smarts!r}")
        if not mol.HasSubstructMatch(pattern):
            required_missing.append(smarts)
    forbidden_present: list[str] = []
    for smarts in forbidden:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            raise PrepError(f"invalid forbidden SMARTS: {smarts!r}")
        if mol.HasSubstructMatch(pattern):
            forbidden_present.append(smarts)
    return {
        "passed": not required_missing and not forbidden_present,
        "required_missing": required_missing,
        "forbidden_present": forbidden_present,
    }


def sanity_check(mol: Chem.Mol) -> list[str]:
    errors: list[str] = []
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            errors.append(
                f"atom {atom.GetIdx()} ({atom.GetSymbol()}) has formal charge "
                f"{atom.GetFormalCharge()}; closed spiropyran inputs must be neutral"
            )
            break
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() != 0:
            errors.append(
                f"atom {atom.GetIdx()} ({atom.GetSymbol()}) carries radical electrons"
            )
            break
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy < MIN_HEAVY_ATOMS:
        errors.append(
            f"only {n_heavy} heavy atoms (minimum {MIN_HEAVY_ATOMS}); "
            f"input is too small to be a spiropyran"
        )
    return errors


def _find_indoline_n_and_gem_c(
    mol: Chem.Mol, spiro_idx: int, smarts_cfg: dict[str, Any]
) -> tuple[int, int]:
    """Locate the indoline N and the gem-disubstituted indoline carbon.

    Both atoms are looked up by SMARTS from smarts_cfg['atom_roles']; we then
    sanity-check that they sit in the same 5-membered ring as the spiro carbon
    so a malformed scaffold fails here rather than producing nonsense
    geometry downstream.
    """
    n_idx = find_atom_by_smarts(
        mol, smarts_cfg["atom_roles"]["indoline_nitrogen"], "indoline_nitrogen"
    )
    gem_idx = find_atom_by_smarts(
        mol, smarts_cfg["atom_roles"]["gem_carbon"], "gem_carbon"
    )
    rings = mol.GetRingInfo().AtomRings()
    five_rings_with_spiro = [r for r in rings if len(r) == 5 and spiro_idx in r]
    if not any(n_idx in r and gem_idx in r for r in five_rings_with_spiro):
        raise PrepError(
            "indoline_nitrogen and gem_carbon are not in the same 5-ring "
            f"as spiro_carbon (spiro={spiro_idx}, N={n_idx}, gem={gem_idx})"
        )
    return n_idx, gem_idx


def _stereocentres_payload(
    mol: Chem.Mol,
    spiro_idx: int,
    spiro_cip: str,
    indoline_n_idx: int,
    gem_c_idx: int,
) -> dict[str, Any]:
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    centres: list[dict[str, Any]] = []
    for atom in mol.GetAtoms():
        if atom.HasProp("_CIPCode"):
            centres.append(
                {
                    "atom_idx": atom.GetIdx(),
                    "symbol": atom.GetSymbol(),
                    "cip": atom.GetProp("_CIPCode"),
                    "is_spiro_centre": atom.GetIdx() == spiro_idx,
                }
            )
    return {
        "spiro_carbon_idx": spiro_idx,
        "spiro_cip": spiro_cip,
        "indoline_nitrogen_idx": indoline_n_idx,
        "gem_carbon_idx": gem_c_idx,
        "stereocentres": centres,
    }


def is_ready(manifest: dict[str, Any], workspace: Path) -> bool:
    return bool(manifest.get("smiles_input"))


def submit(
    manifest: dict[str, Any], workspace: Path, config: dict[str, Any]
) -> dict[str, Any]:
    started = _now_iso()
    smiles_input = manifest.get("smiles_input")
    if not smiles_input:
        return {
            "status": "failed",
            "started_at": started,
            "finished_at": _now_iso(),
            "failure_reason": "manifest has no smiles_input",
        }

    try:
        smiles_canonical = canonicalise(smiles_input)
        mol = parse_validated(smiles_canonical)
    except ValueError as exc:
        return {
            "status": "failed",
            "started_at": started,
            "finished_at": _now_iso(),
            "failure_reason": str(exc),
        }

    sanity_errors = sanity_check(mol)
    if sanity_errors:
        return {
            "status": "failed",
            "started_at": started,
            "finished_at": _now_iso(),
            "failure_reason": "; ".join(sanity_errors),
        }

    filtering = config.get("filtering") or {}
    required = list(filtering.get("smarts_required", []))
    forbidden = list(filtering.get("smarts_forbidden", []))
    try:
        filter_result = apply_smarts_filters(mol, required, forbidden)
    except PrepError as exc:
        return {
            "status": "failed",
            "started_at": started,
            "finished_at": _now_iso(),
            "failure_reason": str(exc),
        }
    if not filter_result["passed"]:
        return {
            "status": "failed",
            "started_at": started,
            "finished_at": _now_iso(),
            "failure_reason": (
                f"SMARTS filter failed; required_missing={filter_result['required_missing']!r}, "
                f"forbidden_present={filter_result['forbidden_present']!r}"
            ),
            "outputs": {"smarts_filter": filter_result},
        }

    smarts_path = Path(config.get("paths", {}).get("smarts", DEFAULT_SMARTS_PATH))
    try:
        smarts_cfg = load_smarts(smarts_path)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        return {
            "status": "failed",
            "started_at": started,
            "finished_at": _now_iso(),
            "failure_reason": f"failed to load SMARTS config: {exc}",
        }

    try:
        spiro_idx = find_atom_by_smarts(
            mol, smarts_cfg["atom_roles"]["spiro_carbon"], "spiro_carbon"
        )
        chromene_o_idx = find_atom_by_smarts(
            mol, smarts_cfg["atom_roles"]["chromene_oxygen"], "chromene_oxygen"
        )
        indoline_n_idx, gem_c_idx = _find_indoline_n_and_gem_c(mol, spiro_idx, smarts_cfg)
        stereo_mol, spiro_cip = assign_spiro_stereo(mol, spiro_idx)
    except PrepError as exc:
        return {
            "status": "failed",
            "started_at": started,
            "finished_at": _now_iso(),
            "failure_reason": str(exc),
        }

    sidecar_rel = Path("prep") / "stereocentres.json"
    sidecar_abs = workspace / sidecar_rel
    atomic_write_json(
        sidecar_abs,
        _stereocentres_payload(stereo_mol, spiro_idx, spiro_cip, indoline_n_idx, gem_c_idx),
    )

    return {
        "status": "done",
        "started_at": started,
        "finished_at": _now_iso(),
        "outputs": {
            "smiles_canonical": smiles_canonical,
            "smiles_anti": smiles_canonical,
            "smiles_syn": smiles_canonical,
            "spiro_carbon_idx": spiro_idx,
            "chromene_oxygen_idx": chromene_o_idx,
            "indoline_nitrogen_idx": indoline_n_idx,
            "gem_carbon_idx": gem_c_idx,
            "spiro_cip": spiro_cip,
            "smarts_filter": filter_result,
            "stereocentres_path": str(sidecar_rel).replace(os.sep, "/"),
        },
    }


def collect(
    manifest: dict[str, Any], workspace: Path, config: dict[str, Any]
) -> dict[str, Any]:
    return {}
