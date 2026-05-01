from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from spiropyran_dr.io_utils import atomic_write_json, read_xyz, write_xyz


def _embed_ethanol() -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    params = AllChem.ETKDGv3()
    params.randomSeed = 7
    assert AllChem.EmbedMolecule(mol, params) == 0
    return mol


def test_xyz_round_trip_preserves_coords_and_symbols(tmp_path: Path) -> None:
    mol = _embed_ethanol()
    out = tmp_path / "ethanol.xyz"
    write_xyz(out, mol, conf_id=0, comment="ethanol test")

    symbols, coords, comment = read_xyz(out)

    assert comment == "ethanol test"
    assert len(symbols) == mol.GetNumAtoms()
    assert symbols == [a.GetSymbol() for a in mol.GetAtoms()]

    conf = mol.GetConformer(0)
    for idx, (x, y, z) in enumerate(coords):
        pos = conf.GetAtomPosition(idx)
        assert abs(pos.x - x) < 1e-6
        assert abs(pos.y - y) < 1e-6
        assert abs(pos.z - z) < 1e-6


def test_xyz_writer_rejects_unembedded_mol(tmp_path: Path) -> None:
    mol = Chem.MolFromSmiles("CCO")  # no conformer
    with pytest.raises(ValueError, match="conformer"):
        write_xyz(tmp_path / "x.xyz", mol, conf_id=0)


def test_atomic_write_json_replaces_existing(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    atomic_write_json(target, {"a": 1})
    atomic_write_json(target, {"a": 2, "b": 3})

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload == {"a": 2, "b": 3}

    leftover = [p for p in tmp_path.iterdir() if p.name.startswith(".tmp_")]
    assert leftover == []


def test_atomic_write_json_creates_parent_dir(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deep" / "out.json"
    atomic_write_json(target, {"x": 42})
    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == {"x": 42}
