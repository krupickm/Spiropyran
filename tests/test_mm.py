from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from spiropyran_dr.config_utils import load_config
from spiropyran_dr.stages import mm, prep
from spiropyran_dr.stages.base import Stage
from spiropyran_dr.stages.mm import (
    MMError,
    cluster_by_rmsd,
    identify_big_gem_substituent,
    indoline_ring_atom_indices,
    label_conformer,
)


# -- helpers ---------------------------------------------------------------


def _config(default_config_path: Path, smarts_config_path: Path) -> dict[str, Any]:
    cfg = load_config(default_config_path)
    cfg["paths"] = {"smarts": str(smarts_config_path)}
    # Smaller embed for faster tests.
    cfg["mm"]["n_embed"] = 20
    return cfg


def _run_prep(
    smiles: str, workspace: Path, default_config_path: Path, smarts_config_path: Path
) -> dict[str, Any]:
    config = _config(default_config_path, smarts_config_path)
    manifest: dict[str, Any] = {"smiles_input": smiles, "stages": {}}
    result = prep.submit(manifest, workspace, config)
    assert result["status"] == "done", result
    manifest["stages"]["prep"] = result
    return manifest


# -- protocol --------------------------------------------------------------


def test_mm_module_satisfies_stage_protocol() -> None:
    stage: Stage = mm
    assert callable(stage.is_ready)
    assert callable(stage.submit)
    assert callable(stage.collect)


# -- pure functions: indoline ring ----------------------------------------


def test_indoline_ring_found_for_bips(
    bips_smiles: str,
    tmp_path: Path,
    default_config_path: Path,
    smarts_config_path: Path,
) -> None:
    manifest = _run_prep(bips_smiles, tmp_path, default_config_path, smarts_config_path)
    out = manifest["stages"]["prep"]["outputs"]
    mol = Chem.MolFromSmiles(out["smiles_canonical"])
    ring = indoline_ring_atom_indices(
        mol, out["spiro_carbon_idx"], out["indoline_nitrogen_idx"]
    )
    assert len(ring) == 5
    assert out["spiro_carbon_idx"] in ring
    assert out["indoline_nitrogen_idx"] in ring
    assert out["gem_carbon_idx"] in ring


def test_indoline_ring_raises_when_n_and_spiro_not_in_same_5ring() -> None:
    # Phenol: no 5-ring at all.
    mol = Chem.MolFromSmiles("Oc1ccccc1")
    with pytest.raises(MMError):
        indoline_ring_atom_indices(mol, spiro_idx=1, indoline_n_idx=0)


# -- pure functions: gem substituent --------------------------------------


def test_identify_big_gem_substituent_picks_ethyl_in_chiral_bips(
    chiral_bips_smiles: str,
    tmp_path: Path,
    default_config_path: Path,
    smarts_config_path: Path,
) -> None:
    manifest = _run_prep(
        chiral_bips_smiles, tmp_path, default_config_path, smarts_config_path
    )
    out = manifest["stages"]["prep"]["outputs"]
    mol = Chem.MolFromSmiles(out["smiles_canonical"])
    ring = indoline_ring_atom_indices(
        mol, out["spiro_carbon_idx"], out["indoline_nitrogen_idx"]
    )
    big = identify_big_gem_substituent(mol, out["gem_carbon_idx"], ring)
    # The ethyl carbon's subtree size (2 heavy atoms: itself + the terminal
    # methyl) must beat the methyl substituent (1 heavy atom).
    big_atom = mol.GetAtomWithIdx(big)
    assert big_atom.GetSymbol() == "C"
    n_heavy_neighbours = sum(
        1 for nb in big_atom.GetNeighbors() if nb.GetAtomicNum() > 1
    )
    assert n_heavy_neighbours >= 2  # bonded to gem-C and to its terminal methyl


def test_identify_big_gem_substituent_deterministic_on_bips_ties(
    bips_smiles: str,
    tmp_path: Path,
    default_config_path: Path,
    smarts_config_path: Path,
) -> None:
    manifest = _run_prep(bips_smiles, tmp_path, default_config_path, smarts_config_path)
    out = manifest["stages"]["prep"]["outputs"]
    mol = Chem.MolFromSmiles(out["smiles_canonical"])
    ring = indoline_ring_atom_indices(
        mol, out["spiro_carbon_idx"], out["indoline_nitrogen_idx"]
    )
    a = identify_big_gem_substituent(mol, out["gem_carbon_idx"], ring)
    b = identify_big_gem_substituent(mol, out["gem_carbon_idx"], ring)
    assert a == b
    assert mol.GetAtomWithIdx(a).GetSymbol() == "C"


# -- pure functions: labeller ---------------------------------------------


def test_label_conformer_flips_when_o_face_flips(
    bips_smiles: str,
    tmp_path: Path,
    default_config_path: Path,
    smarts_config_path: Path,
) -> None:
    manifest = _run_prep(bips_smiles, tmp_path, default_config_path, smarts_config_path)
    out = manifest["stages"]["prep"]["outputs"]
    mol_h = Chem.AddHs(Chem.MolFromSmiles(out["smiles_canonical"]))
    params = AllChem.ETKDGv3()
    params.randomSeed = 11
    assert AllChem.EmbedMolecule(mol_h, params) == 0

    ring = indoline_ring_atom_indices(
        mol_h, out["spiro_carbon_idx"], out["indoline_nitrogen_idx"]
    )
    big = identify_big_gem_substituent(mol_h, out["gem_carbon_idx"], ring)

    label_a = label_conformer(
        mol_h,
        conf_id=0,
        indoline_ring=ring,
        chromene_o_idx=out["chromene_oxygen_idx"],
        big_sub_idx=big,
    )

    # Reflect the chromene oxygen through the indoline plane by flipping the
    # sign of its component along the plane normal. Cheap proxy: invert all
    # coordinates of the O atom relative to the ring centroid -- this is
    # enough to flip its signed displacement.
    conf = mol_h.GetConformer(0)
    import numpy as np

    ring_xyz = np.array(
        [
            [
                conf.GetAtomPosition(i).x,
                conf.GetAtomPosition(i).y,
                conf.GetAtomPosition(i).z,
            ]
            for i in ring
        ]
    )
    centroid = ring_xyz.mean(axis=0)
    centred = ring_xyz - centroid
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    normal = vh[-1]
    o_pos = conf.GetAtomPosition(out["chromene_oxygen_idx"])
    o_xyz = np.array([o_pos.x, o_pos.y, o_pos.z])
    proj = (o_xyz - centroid).dot(normal)
    new_xyz = o_xyz - 2.0 * proj * normal
    conf.SetAtomPosition(
        out["chromene_oxygen_idx"], (float(new_xyz[0]), float(new_xyz[1]), float(new_xyz[2]))
    )

    label_b = label_conformer(
        mol_h,
        conf_id=0,
        indoline_ring=ring,
        chromene_o_idx=out["chromene_oxygen_idx"],
        big_sub_idx=big,
    )
    assert {label_a, label_b} == {"anti", "syn"}


# -- pure functions: clustering -------------------------------------------


def test_cluster_by_rmsd_keeps_lowest_energy_within_threshold() -> None:
    # Two near-duplicate low-energy conformers of butane plus a distinct one.
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    params = AllChem.ETKDGv3()
    params.randomSeed = 3
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=8, params=params)
    AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200)
    pairs = [(int(cid), float(cid)) for cid in ids]  # synthetic energies
    kept = cluster_by_rmsd(mol, pairs, rmsd_threshold_ang=1000.0, max_keep=10)
    # Threshold larger than any possible RMSD: only the lowest-energy survives.
    assert len(kept) == 1
    assert kept[0][0] == pairs[0][0]


def test_cluster_by_rmsd_caps_at_max_keep() -> None:
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCCCC"))
    params = AllChem.ETKDGv3()
    params.randomSeed = 9
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=10, params=params)
    AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200)
    # 0.0 A threshold accepts everything; max_keep caps the result.
    pairs = [(int(cid), float(i)) for i, cid in enumerate(ids)]
    kept = cluster_by_rmsd(mol, pairs, rmsd_threshold_ang=0.0, max_keep=3)
    assert len(kept) == 3


def test_cluster_by_rmsd_preserves_input_order_for_unique_confs() -> None:
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCCCC"))
    params = AllChem.ETKDGv3()
    params.randomSeed = 12
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=4, params=params)
    AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200)
    pairs = [(int(cid), float(i)) for i, cid in enumerate(ids)]
    kept = cluster_by_rmsd(mol, pairs, rmsd_threshold_ang=0.0, max_keep=10)
    # 0.0 threshold accepts everything; order matches input (energy-sorted).
    assert [p[0] for p in kept] == [p[0] for p in pairs]


# -- stage-level submit/collect/is_ready ----------------------------------


def test_is_ready_requires_prep_done(tmp_path: Path) -> None:
    assert (
        mm.is_ready({"stages": {"prep": {"status": "pending"}}}, tmp_path) is False
    )
    assert mm.is_ready({"stages": {}}, tmp_path) is False
    manifest = {
        "stages": {
            "prep": {
                "status": "done",
                "outputs": {
                    "smiles_canonical": "CCO",
                    "spiro_carbon_idx": 0,
                    "chromene_oxygen_idx": 1,
                    "indoline_nitrogen_idx": 2,
                    "gem_carbon_idx": 3,
                },
            }
        }
    }
    assert mm.is_ready(manifest, tmp_path) is True


def test_collect_is_noop(tmp_path: Path) -> None:
    assert mm.collect({}, tmp_path, {}) == {}


def test_submit_writes_xyz_files_and_sidecar(
    tmp_path: Path,
    chiral_bips_smiles: str,
    default_config_path: Path,
    smarts_config_path: Path,
) -> None:
    config = _config(default_config_path, smarts_config_path)
    manifest = _run_prep(
        chiral_bips_smiles, tmp_path, default_config_path, smarts_config_path
    )
    result = mm.submit(manifest, tmp_path, config)
    assert result["status"] == "done", result
    out = result["outputs"]
    assert out["n_conformers_anti"] >= 1
    assert out["n_conformers_syn"] >= 1
    anti_dir = tmp_path / out["anti_xyz_dir"]
    syn_dir = tmp_path / out["syn_xyz_dir"]
    assert anti_dir.is_dir()
    assert syn_dir.is_dir()
    anti_files = sorted(anti_dir.glob("conf_*.xyz"))
    syn_files = sorted(syn_dir.glob("conf_*.xyz"))
    assert len(anti_files) == out["n_conformers_anti"]
    assert len(syn_files) == out["n_conformers_syn"]

    sidecar = tmp_path / "mm" / "conformers.json"
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert "anti" in payload and "syn" in payload
    for entry in payload["anti"] + payload["syn"]:
        assert "conf_id" in entry
        assert "xyz" in entry
        assert "mmff_energy_kcal_mol" in entry
        assert "label" in entry


def test_submit_outputs_match_manifest_schema(
    tmp_path: Path,
    chiral_bips_smiles: str,
    default_config_path: Path,
    smarts_config_path: Path,
) -> None:
    config = _config(default_config_path, smarts_config_path)
    manifest = _run_prep(
        chiral_bips_smiles, tmp_path, default_config_path, smarts_config_path
    )
    result = mm.submit(manifest, tmp_path, config)
    out = result["outputs"]
    for key in (
        "n_conformers_anti",
        "n_conformers_syn",
        "anti_xyz_dir",
        "syn_xyz_dir",
        "anti",
        "syn",
    ):
        assert key in out, f"missing {key}"
    for entry in out["anti"]:
        assert entry["label"] == "anti"
        assert isinstance(entry["mmff_energy_kcal_mol"], float)
    for entry in out["syn"]:
        assert entry["label"] == "syn"


def test_submit_fails_when_only_one_label_appears(
    tmp_path: Path,
    chiral_bips_smiles: str,
    default_config_path: Path,
    smarts_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(default_config_path, smarts_config_path)
    manifest = _run_prep(
        chiral_bips_smiles, tmp_path, default_config_path, smarts_config_path
    )
    # Force every conformer to label as 'anti' to simulate a scaffold that
    # cannot produce a syn diastereomer.
    monkeypatch.setattr(mm, "label_conformer", lambda *args, **kwargs: "anti")
    result = mm.submit(manifest, tmp_path, config)
    assert result["status"] == "failed"
    assert "syn" in result["failure_reason"].lower()
