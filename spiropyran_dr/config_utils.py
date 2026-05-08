from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

# Config sections that affect chemistry outcomes; included in the manifest hash.
_CHEMISTRY_SECTIONS = ("mecp", "xtb_constr", "crest", "dft")


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
    "energy_window_kj_mol": 20.0,
    "max_conformers_per_diastereomer": 20,
}

# CREST submission is delegated to a user-maintained wrapper script that
# owns NPROC, queue, and other PBS settings; we only supply walltime (in
# whole hours, the wrapper's CLI) and the absolute path to the wrapper.
CREST_DEFAULTS: dict[str, Any] = {
    "walltime_hours": 24,
    "script_path": "/storage/brno2/home/krupickm/bin/sub_crest.sh",
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
    crest = data.get("crest") or {}
    data["crest"] = {**CREST_DEFAULTS, **crest}
    return data


def compute_config_hash(config: dict[str, Any]) -> str:
    """SHA-256 over the chemistry-relevant config sections (canonical JSON)."""
    relevant = {k: config[k] for k in _CHEMISTRY_SECTIONS if k in config}
    canonical = json.dumps(relevant, sort_keys=True, ensure_ascii=True)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def load_smarts(path: Path) -> dict[str, Any]:
    data = load_yaml(path)
    atom_roles = data.get("atom_roles")
    if not isinstance(atom_roles, dict):
        raise KeyError(f"{path}: missing 'atom_roles' mapping")
    for required_key in REQUIRED_ATOM_ROLES:
        if required_key not in atom_roles:
            raise KeyError(f"{path}: atom_roles.{required_key} is required")
        if not isinstance(atom_roles[required_key], str):
            raise ValueError(
                f"{path}: atom_roles.{required_key} must be a string SMARTS"
            )
    return data
