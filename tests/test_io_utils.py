from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from spiropyran_dr.io_utils import (
    atomic_write_json,
    check_orca_normal_termination,
    parse_crest_energy_from_comment,
    parse_orca_sp_energies,
    read_xyz,
    read_xyz_multiframe,
    write_xyz,
    write_xyz_from_arrays,
)


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


# -- multi-frame XYZ + CREST energies -------------------------------------


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_read_xyz_multiframe_parses_three_frames(tmp_path: Path) -> None:
    # Tiny H2 trajectory with three frames; comment line varies per frame.
    text = (
        "2\n"
        "frame 0\n"
        "H 0.000 0.000 0.000\n"
        "H 0.740 0.000 0.000\n"
        "2\n"
        "frame 1\n"
        "H 0.000 0.000 0.000\n"
        "H 0.745 0.000 0.000\n"
        "2\n"
        "frame 2\n"
        "H 0.000 0.000 0.000\n"
        "H 0.750 0.000 0.000\n"
    )
    src = tmp_path / "traj.xyz"
    _write(src, text)

    frames = read_xyz_multiframe(src)
    assert len(frames) == 3
    for i, (symbols, coords, comment) in enumerate(frames):
        assert symbols == ["H", "H"]
        assert comment == f"frame {i}"
        assert len(coords) == 2
        assert coords[0] == (0.0, 0.0, 0.0)
    assert frames[0][1][1] == (0.740, 0.0, 0.0)
    assert frames[2][1][1] == (0.750, 0.0, 0.0)


def test_read_xyz_multiframe_tolerates_trailing_blank_lines(tmp_path: Path) -> None:
    text = "1\nx\nHe 0 0 0\n\n1\ny\nHe 1 0 0\n\n"
    src = tmp_path / "blank.xyz"
    _write(src, text)
    frames = read_xyz_multiframe(src)
    assert len(frames) == 2
    assert frames[1][2] == "y"


def test_parse_crest_energy_from_comment_real_crest_format() -> None:
    # Real CREST output: a single Hartree value, indented.
    assert parse_crest_energy_from_comment("        -54.21320214") == -54.21320214


def test_parse_crest_energy_from_comment_with_trailing_tokens() -> None:
    # Synthetic fixtures append a relative energy and a label; first token wins.
    assert (
        parse_crest_energy_from_comment("   -22.10000000     0.00000000  conf 0")
        == -22.10000000
    )


def test_parse_crest_energy_from_comment_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        parse_crest_energy_from_comment("   ")


def test_write_xyz_from_arrays_round_trips(tmp_path: Path) -> None:
    symbols = ["O", "H", "H"]
    coords = [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)]
    out = tmp_path / "water.xyz"
    write_xyz_from_arrays(out, symbols, coords, comment="water")

    parsed_symbols, parsed_coords, comment = read_xyz(out)
    assert parsed_symbols == symbols
    assert comment == "water"
    for (x, y, z), (px, py, pz) in zip(coords, parsed_coords):
        assert abs(x - px) < 1e-6
        assert abs(y - py) < 1e-6
        assert abs(z - pz) < 1e-6


def test_write_xyz_from_arrays_rejects_length_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="length"):
        write_xyz_from_arrays(
            tmp_path / "bad.xyz", ["O", "H"], [(0.0, 0.0, 0.0)], comment=""
        )


# -- ORCA output parsers --------------------------------------------------

_ORCA_SUCCESS = """\
ORCA SCF DONE ...

                         FINAL SINGLE POINT ENERGY       -76.400000000

               FINAL SINGLE POINT ENERGY       -76.398000000

FINAL SINGLE POINT ENERGY       -76.395000000

****ORCA TERMINATED NORMALLY****
"""

_ORCA_FAILED = """\
ORCA SCF DONE ...

                         FINAL SINGLE POINT ENERGY       -76.400000000

Error: SCF did not converge.
"""


def test_parse_orca_sp_energies_returns_all_in_order(tmp_path: Path) -> None:
    src = tmp_path / "orca.out"
    src.write_text(_ORCA_SUCCESS, encoding="utf-8")
    energies = parse_orca_sp_energies(src)
    assert energies == [-76.4, -76.398, -76.395]


def test_parse_orca_sp_energies_raises_on_no_energy_lines(tmp_path: Path) -> None:
    src = tmp_path / "orca.out"
    src.write_text("no energy here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="FINAL SINGLE POINT ENERGY"):
        parse_orca_sp_energies(src)


def test_check_orca_normal_termination_true(tmp_path: Path) -> None:
    src = tmp_path / "orca.out"
    src.write_text(_ORCA_SUCCESS, encoding="utf-8")
    assert check_orca_normal_termination(src) is True


def test_check_orca_normal_termination_false(tmp_path: Path) -> None:
    src = tmp_path / "orca.out"
    src.write_text(_ORCA_FAILED, encoding="utf-8")
    assert check_orca_normal_termination(src) is False
