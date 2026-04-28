from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdkit import Chem

from spiropyran_dr.config_utils import load_config
from spiropyran_dr.stages import prep
from spiropyran_dr.stages.base import Stage
from spiropyran_dr.stages.prep import (
    PrepError,
    apply_smarts_filters,
    canonicalise,
    find_atom_by_smarts,
    sanity_check,
)


def test_prep_module_satisfies_stage_protocol() -> None:
    stage: Stage = prep
    assert callable(stage.is_ready)
    assert callable(stage.submit)
    assert callable(stage.collect)


# --- pure functions --------------------------------------------------------


def test_canonicalise_round_trip(bips_smiles: str) -> None:
    canonical = canonicalise(bips_smiles)
    assert canonicalise(canonical) == canonical


def test_canonicalise_invalid_raises() -> None:
    with pytest.raises(ValueError):
        canonicalise("not a smiles")


def _smarts(smarts_config_path: Path, key: str) -> str:
    import yaml

    data = yaml.safe_load(smarts_config_path.read_text(encoding="utf-8"))
    return data["atom_roles"][key]


def test_find_spiro_carbon_in_bips(bips_smiles: str, smarts_config_path: Path) -> None:
    mol = Chem.MolFromSmiles(canonicalise(bips_smiles))
    idx = find_atom_by_smarts(mol, _smarts(smarts_config_path, "spiro_carbon"), "spiro_carbon")
    atom = mol.GetAtomWithIdx(idx)
    assert atom.GetSymbol() == "C"
    assert atom.GetHybridization() == Chem.HybridizationType.SP3
    neighbour_symbols = sorted(n.GetSymbol() for n in atom.GetNeighbors())
    assert "N" in neighbour_symbols
    assert "O" in neighbour_symbols
    assert sum(1 for r in mol.GetRingInfo().AtomRings() if idx in r) == 2


def test_find_chromene_oxygen_in_bips(bips_smiles: str, smarts_config_path: Path) -> None:
    mol = Chem.MolFromSmiles(canonicalise(bips_smiles))
    idx = find_atom_by_smarts(mol, _smarts(smarts_config_path, "chromene_oxygen"), "chromene_oxygen")
    atom = mol.GetAtomWithIdx(idx)
    assert atom.GetSymbol() == "O"
    spiro_idx = find_atom_by_smarts(
        mol, _smarts(smarts_config_path, "spiro_carbon"), "spiro_carbon"
    )
    neighbour_indices = {n.GetIdx() for n in atom.GetNeighbors()}
    assert spiro_idx in neighbour_indices


def test_find_spiro_carbon_in_methyl_bips(
    methyl_bips_smiles: str, smarts_config_path: Path
) -> None:
    mol = Chem.MolFromSmiles(canonicalise(methyl_bips_smiles))
    idx = find_atom_by_smarts(mol, _smarts(smarts_config_path, "spiro_carbon"), "spiro_carbon")
    assert mol.GetAtomWithIdx(idx).GetSymbol() == "C"


def test_find_spiro_carbon_no_match_raises(
    non_spiro_smiles: str, smarts_config_path: Path
) -> None:
    mol = Chem.MolFromSmiles(canonicalise(non_spiro_smiles))
    with pytest.raises(PrepError, match="spiro_carbon"):
        find_atom_by_smarts(mol, _smarts(smarts_config_path, "spiro_carbon"), "spiro_carbon")


def test_apply_smarts_filters_required_missing(bips_smiles: str) -> None:
    mol = Chem.MolFromSmiles(canonicalise(bips_smiles))
    result = apply_smarts_filters(mol, required=["[#16]"], forbidden=[])
    assert result["passed"] is False
    assert result["required_missing"] == ["[#16]"]
    assert result["forbidden_present"] == []


def test_apply_smarts_filters_forbidden_present(bips_smiles: str) -> None:
    mol = Chem.MolFromSmiles(canonicalise(bips_smiles))
    result = apply_smarts_filters(mol, required=[], forbidden=["[#7]"])
    assert result["passed"] is False
    assert result["forbidden_present"] == ["[#7]"]
    assert result["required_missing"] == []


def test_apply_smarts_filters_empty_lists_pass(bips_smiles: str) -> None:
    mol = Chem.MolFromSmiles(canonicalise(bips_smiles))
    result = apply_smarts_filters(mol, required=[], forbidden=[])
    assert result == {"passed": True, "required_missing": [], "forbidden_present": []}


def test_sanity_check_pass_on_bips(bips_smiles: str) -> None:
    mol = Chem.MolFromSmiles(canonicalise(bips_smiles))
    assert sanity_check(mol) == []


def test_sanity_check_rejects_charged(charged_smiles: str) -> None:
    mol = Chem.MolFromSmiles(charged_smiles)
    errors = sanity_check(mol)
    assert errors
    assert any("charge" in e for e in errors)


def test_sanity_check_rejects_radical(radical_smiles: str) -> None:
    mol = Chem.MolFromSmiles(radical_smiles)
    errors = sanity_check(mol)
    assert errors
    assert any("radical" in e for e in errors)


def test_sanity_check_rejects_too_small(non_spiro_smiles: str) -> None:
    mol = Chem.MolFromSmiles(non_spiro_smiles)
    errors = sanity_check(mol)
    assert errors
    assert any("heavy atoms" in e for e in errors)


# --- stage-level submit/collect/is_ready ----------------------------------


def _config(default_config_path: Path, smarts_config_path: Path) -> dict:
    cfg = load_config(default_config_path)
    cfg["paths"] = {"smarts": str(smarts_config_path)}
    return cfg


def test_is_ready_true_when_smiles_input_present() -> None:
    assert prep.is_ready({"smiles_input": "CCO"}, Path(".")) is True


def test_is_ready_false_when_smiles_input_missing() -> None:
    assert prep.is_ready({}, Path(".")) is False
    assert prep.is_ready({"smiles_input": ""}, Path(".")) is False


def test_collect_is_noop(tmp_path: Path) -> None:
    assert prep.collect({"smiles_input": "anything"}, tmp_path, {}) == {}


def test_submit_bips_writes_sidecar_and_returns_done(
    tmp_path: Path,
    bips_smiles: str,
    default_config_path: Path,
    smarts_config_path: Path,
) -> None:
    manifest = {"smiles_input": bips_smiles}
    config = _config(default_config_path, smarts_config_path)
    result = prep.submit(manifest, tmp_path, config)
    assert result["status"] == "done", result
    out = result["outputs"]
    assert out["smiles_canonical"]
    assert out["spiro_carbon_idx"] >= 0
    assert out["chromene_oxygen_idx"] >= 0
    assert out["spiro_cip"] in ("R", "S")
    assert out["smarts_filter"]["passed"] is True
    sidecar = tmp_path / "prep" / "stereocentres.json"
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["spiro_carbon_idx"] == out["spiro_carbon_idx"]
    assert payload["spiro_cip"] == out["spiro_cip"]
    assert any(c["is_spiro_centre"] for c in payload["stereocentres"])


def test_submit_anti_syn_smiles_equal_canonical(
    tmp_path: Path,
    bips_smiles: str,
    default_config_path: Path,
    smarts_config_path: Path,
) -> None:
    config = _config(default_config_path, smarts_config_path)
    result = prep.submit({"smiles_input": bips_smiles}, tmp_path, config)
    out = result["outputs"]
    assert out["smiles_anti"] == out["smiles_canonical"]
    assert out["smiles_syn"] == out["smiles_canonical"]


def test_submit_invalid_smiles_returns_failed(
    tmp_path: Path,
    default_config_path: Path,
    smarts_config_path: Path,
) -> None:
    config = _config(default_config_path, smarts_config_path)
    result = prep.submit({"smiles_input": "not a smiles"}, tmp_path, config)
    assert result["status"] == "failed"
    assert "invalid SMILES" in result["failure_reason"]


def test_submit_non_spiro_molecule_returns_failed(
    tmp_path: Path,
    non_spiro_smiles: str,
    default_config_path: Path,
    smarts_config_path: Path,
) -> None:
    config = _config(default_config_path, smarts_config_path)
    result = prep.submit({"smiles_input": non_spiro_smiles}, tmp_path, config)
    assert result["status"] == "failed"
    # Ethanol is rejected by the heavy-atom sanity check before reaching the
    # SMARTS lookup, so the reason should mention size, not "spiro_carbon".
    assert "heavy atoms" in result["failure_reason"]


def test_submit_forbidden_smarts_blocks_before_atom_role_lookup(
    tmp_path: Path,
    bips_smiles: str,
    default_config_path: Path,
    smarts_config_path: Path,
) -> None:
    config = _config(default_config_path, smarts_config_path)
    config["filtering"]["smarts_forbidden"] = ["[#7]"]
    result = prep.submit({"smiles_input": bips_smiles}, tmp_path, config)
    assert result["status"] == "failed"
    assert "SMARTS filter failed" in result["failure_reason"]
    assert not (tmp_path / "prep" / "stereocentres.json").exists()


def test_submit_required_smarts_missing_blocks(
    tmp_path: Path,
    bips_smiles: str,
    default_config_path: Path,
    smarts_config_path: Path,
) -> None:
    config = _config(default_config_path, smarts_config_path)
    config["filtering"]["smarts_required"] = ["[#16]"]  # sulfur, not in BIPS
    result = prep.submit({"smiles_input": bips_smiles}, tmp_path, config)
    assert result["status"] == "failed"
    assert "[#16]" in result["failure_reason"]


def test_submit_charged_input_returns_failed(
    tmp_path: Path,
    charged_smiles: str,
    default_config_path: Path,
    smarts_config_path: Path,
) -> None:
    config = _config(default_config_path, smarts_config_path)
    result = prep.submit({"smiles_input": charged_smiles}, tmp_path, config)
    assert result["status"] == "failed"
    assert "charge" in result["failure_reason"]
