from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from spiropyran_dr.config_utils import load_config, load_smarts, load_yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "spiropyran_dr"
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "default.yaml"
SMARTS_CONFIG = PACKAGE_ROOT / "config" / "smarts.yaml"


def test_load_config_returns_filtering_block() -> None:
    cfg = load_config(DEFAULT_CONFIG)
    assert isinstance(cfg["filtering"]["smarts_required"], list)
    assert isinstance(cfg["filtering"]["smarts_forbidden"], list)


def test_load_config_missing_filtering_defaults_to_empty(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("temperature_kelvin: 298.15\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg["filtering"] == {"smarts_required": [], "smarts_forbidden": []}


def test_load_smarts_real_file_has_atom_roles() -> None:
    smarts = load_smarts(SMARTS_CONFIG)
    assert "spiro_carbon" in smarts["atom_roles"]
    assert "chromene_oxygen" in smarts["atom_roles"]


def test_load_smarts_requires_atom_roles(tmp_path: Path) -> None:
    p = tmp_path / "s.yaml"
    p.write_text("other_key: 1\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_smarts(p)


def test_load_smarts_requires_spiro_carbon(tmp_path: Path) -> None:
    p = tmp_path / "s.yaml"
    p.write_text("atom_roles:\n  chromene_oxygen: '[#8]'\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_smarts(p)


def test_load_smarts_requires_indoline_nitrogen(tmp_path: Path) -> None:
    p = tmp_path / "s.yaml"
    p.write_text(
        "atom_roles:\n"
        "  spiro_carbon: '[#6]'\n"
        "  chromene_oxygen: '[#8]'\n"
        "  gem_carbon: '[#6]'\n",
        encoding="utf-8",
    )
    with pytest.raises(KeyError, match="indoline_nitrogen"):
        load_smarts(p)


def test_load_smarts_requires_gem_carbon(tmp_path: Path) -> None:
    p = tmp_path / "s.yaml"
    p.write_text(
        "atom_roles:\n"
        "  spiro_carbon: '[#6]'\n"
        "  chromene_oxygen: '[#8]'\n"
        "  indoline_nitrogen: '[#7]'\n",
        encoding="utf-8",
    )
    with pytest.raises(KeyError, match="gem_carbon"):
        load_smarts(p)


def test_load_config_surfaces_mm_defaults() -> None:
    cfg = load_config(DEFAULT_CONFIG)
    assert cfg["mm"]["n_embed"] >= 1
    assert cfg["mm"]["rmsd_threshold_angstrom"] > 0
    assert cfg["ensemble"]["max_conformers_per_diastereomer"] >= 1
    assert cfg["ensemble"]["energy_window_kj_mol"] > 0


def test_load_config_user_mm_overrides_defaults(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("mm:\n  n_embed: 7\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg["mm"]["n_embed"] == 7
    assert cfg["mm"]["random_seed"] == 42  # default surfaces through
    assert cfg["ensemble"]["max_conformers_per_diastereomer"] == 20


def test_load_yaml_invalid_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("key: : :\n", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        load_yaml(p)


def test_load_yaml_non_mapping_raises(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_yaml(p)
