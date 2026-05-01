from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} did not parse to a mapping")
    return data


MM_DEFAULTS: dict[str, Any] = {
    "n_embed": 50,
    "mmff_max_iters": 200,
    "rmsd_threshold_angstrom": 0.5,
    "random_seed": 42,
}

ENSEMBLE_DEFAULTS: dict[str, Any] = {
    "max_conformers_per_diastereomer": 20,
}

REQUIRED_ATOM_ROLES: tuple[str, ...] = (
    "spiro_carbon",
    "chromene_oxygen",
    "indoline_nitrogen",
    "gem_carbon",
)


def load_config(path: Path) -> dict[str, Any]:
    data = load_yaml(path)
    filtering = data.get("filtering") or {}
    data["filtering"] = {
        "smarts_required": list(filtering.get("smarts_required", [])),
        "smarts_forbidden": list(filtering.get("smarts_forbidden", [])),
    }
    mm = data.get("mm") or {}
    data["mm"] = {**MM_DEFAULTS, **mm}
    ensemble = data.get("ensemble") or {}
    data["ensemble"] = {**ENSEMBLE_DEFAULTS, **ensemble}
    return data


def load_smarts(path: Path) -> dict[str, Any]:
    data = load_yaml(path)
    atom_roles = data.get("atom_roles")
    if not isinstance(atom_roles, dict):
        raise KeyError(f"{path}: missing 'atom_roles' mapping")
    for required_key in REQUIRED_ATOM_ROLES:
        if required_key not in atom_roles:
            raise KeyError(f"{path}: atom_roles.{required_key} is required")
        if not isinstance(atom_roles[required_key], str):
            raise ValueError(f"{path}: atom_roles.{required_key} must be a string SMARTS")
    return data
