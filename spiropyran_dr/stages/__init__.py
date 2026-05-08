from __future__ import annotations

import importlib
import types

STAGE_ORDER: tuple[str, ...] = (
    "prep",
    "mm",
    "xtb_constr",
    "crest",
    "dft_sp",
    "dft_freq",
    "aggregate",
)

# True = stage submits PBS jobs and needs qstat polling; False = runs locally.
STAGE_IS_PBS: dict[str, bool] = {
    "prep": False,
    "mm": False,
    "xtb_constr": True,
    "crest": True,
    "dft_sp": True,
    "dft_freq": True,
    "aggregate": False,
}

_MODULE_MAP: dict[str, str] = {
    "prep": "spiropyran_dr.stages.prep",
    "mm": "spiropyran_dr.stages.mm",
    "xtb_constr": "spiropyran_dr.stages.xtb_stage",
    "crest": "spiropyran_dr.stages.crest_stage",
    "dft_sp": "spiropyran_dr.stages.dft_sp_stage",
    "dft_freq": "spiropyran_dr.stages.dft_freq_stage",
    "aggregate": "spiropyran_dr.stages.aggregate",
}


def get_stage_module(name: str) -> types.ModuleType | None:
    """Return the module for a stage, or None if not yet implemented."""
    module_path = _MODULE_MAP.get(name)
    if module_path is None:
        raise ValueError(f"Unknown stage: {name!r}")
    try:
        return importlib.import_module(module_path)
    except ImportError:
        return None
