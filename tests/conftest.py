from __future__ import annotations

from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "spiropyran_dr"


@pytest.fixture
def default_config_path() -> Path:
    return PACKAGE_ROOT / "config" / "default.yaml"


@pytest.fixture
def smarts_config_path() -> Path:
    return PACKAGE_ROOT / "config" / "smarts.yaml"


@pytest.fixture
def bips_smiles() -> str:
    # 1',3',3'-trimethylspiro[chromene-2,2'-indoline] -- closed-form BIPS,
    # the canonical unsubstituted spiropyran reference structure.
    return "CC1(C)c2ccccc2N(C)C13Oc4ccccc4C=C3"


@pytest.fixture
def methyl_bips_smiles() -> str:
    # 6-methyl variant on the chromene ring, to confirm the SMARTS is not
    # accidentally tied to the unsubstituted parent.
    return "CC1(C)c2ccccc2N(C)C13Oc4cc(C)ccc4C=C3"


@pytest.fixture
def non_spiro_smiles() -> str:
    return "CCO"


@pytest.fixture
def charged_smiles() -> str:
    return "C[N+](C)(C)C.[Cl-]"


@pytest.fixture
def radical_smiles() -> str:
    return "[CH3]"
