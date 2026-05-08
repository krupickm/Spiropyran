from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from rdkit import Chem


def write_xyz(path: Path, mol: Chem.Mol, conf_id: int = 0, comment: str = "") -> None:
    """Write a single conformer of an RDKit Mol as a standard XYZ file.

    Coordinates are taken from the conformer with id ``conf_id`` (default 0).
    Element symbols come from ``atom.GetSymbol()``; explicit hydrogens are
    written if they are present on the Mol. The caller is responsible for
    AddHs/embedding before calling this.
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("Mol has no conformer; embed (and AddHs) before writing")
    conf = mol.GetConformer(conf_id)

    n = mol.GetNumAtoms()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{n}", comment]
    for idx in range(n):
        pos = conf.GetAtomPosition(idx)
        sym = mol.GetAtomWithIdx(idx).GetSymbol()
        lines.append(f"{sym} {pos.x:.8f} {pos.y:.8f} {pos.z:.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_xyz(path: Path) -> tuple[list[str], list[tuple[float, float, float]], str]:
    """Parse a single-frame XYZ file into (symbols, coords, comment)."""
    text = path.read_text(encoding="utf-8").splitlines()
    n = int(text[0].strip())
    comment = text[1] if len(text) > 1 else ""
    symbols: list[str] = []
    coords: list[tuple[float, float, float]] = []
    for raw in text[2 : 2 + n]:
        parts = raw.split()
        symbols.append(parts[0])
        coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return symbols, coords, comment


def read_xyz_multiframe(
    path: Path,
) -> list[tuple[list[str], list[tuple[float, float, float]], str]]:
    """Parse a multi-frame XYZ (CREST `crest_conformers.xyz` format).

    Each frame: count line, comment line, then `count` "symbol x y z" lines.
    Blank lines between frames are tolerated.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    frames: list[tuple[list[str], list[tuple[float, float, float]], str]] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i].strip())
        comment = lines[i + 1] if i + 1 < len(lines) else ""
        symbols: list[str] = []
        coords: list[tuple[float, float, float]] = []
        for raw in lines[i + 2 : i + 2 + n]:
            parts = raw.split()
            symbols.append(parts[0])
            coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
        if len(symbols) != n:
            raise ValueError(
                f"{path}: frame at line {i} declares {n} atoms but only "
                f"{len(symbols)} parsed"
            )
        frames.append((symbols, coords, comment))
        i += 2 + n
    return frames


def parse_crest_energy_from_comment(comment: str) -> float:
    """Extract the absolute electronic energy (Hartree) from a CREST
    `crest_conformers.xyz` frame comment line.

    CREST writes the absolute Hartree energy as the first whitespace-
    separated token of the comment line; the `crest.energies` sidecar
    holds only relative energies (kcal/mol) and is not consulted.
    """
    tokens = comment.split()
    if not tokens:
        raise ValueError("empty CREST comment line; expected absolute energy")
    return float(tokens[0])


def write_xyz_from_arrays(
    path: Path,
    symbols: list[str],
    coords: list[tuple[float, float, float]],
    comment: str = "",
) -> None:
    """Write a single XYZ frame from raw arrays (no RDKit Mol needed).

    Used to dump filtered CREST conformers, where we have only symbols and
    coordinates from a parsed ensemble file.
    """
    if len(symbols) != len(coords):
        raise ValueError(
            f"length mismatch: {len(symbols)} symbols vs {len(coords)} coords"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{len(symbols)}", comment]
    for sym, (x, y, z) in zip(symbols, coords):
        lines.append(f"{sym} {x:.8f} {y:.8f} {z:.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_xcontrol_distance_constraint(
    path: Path,
    atom_a_idx0: int,
    atom_b_idx0: int,
    distance_ang: float,
    force_constant: float,
) -> None:
    """Write an xtb/CREST $constrain block fixing one distance.

    Atom indices are accepted 0-based (RDKit / prep convention) and written
    1-based (xtb/CREST convention)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f"$constrain\n"
        f"  force constant={force_constant}\n"
        f"  distance: {atom_a_idx0 + 1},{atom_b_idx0 + 1},{distance_ang}\n"
        f"$end\n"
    )
    path.write_text(content, encoding="utf-8")


def parse_orca_sp_energies(path: Path) -> list[float]:
    """Return all final SCF energies from an ORCA single-point output file.

    ORCA writes one ``FINAL SINGLE POINT ENERGY`` line per geometry when run
    with a multi-frame XYZ input. Returns energies in the order they appear
    (i.e. conformer order). Raises ``ValueError`` if no energy lines are found.
    """
    energies: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "FINAL SINGLE POINT ENERGY" in line:
            energies.append(float(line.split()[-1]))
    if not energies:
        raise ValueError(f"{path}: no 'FINAL SINGLE POINT ENERGY' lines found")
    return energies


def parse_orca_sp_energy(path: Path) -> float:
    """Return the single SP energy from a per-conformer ORCA output file.

    Used by dft_sp where each ORCA job runs on exactly one geometry, so we
    expect exactly one ``FINAL SINGLE POINT ENERGY`` line. Zero or more than
    one is an error: more than one means a multi-frame XYZ slipped through,
    which would reuse the previous geometry's SCF guess and silently corrupt
    energies for chemically distinct conformers.
    """
    energies = parse_orca_sp_energies(path)
    if len(energies) != 1:
        raise ValueError(
            f"{path}: expected exactly one 'FINAL SINGLE POINT ENERGY' line, "
            f"found {len(energies)}"
        )
    return energies[0]


def check_orca_normal_termination(path: Path) -> bool:
    """Return True if ``ORCA TERMINATED NORMALLY`` appears in the output file."""
    return "ORCA TERMINATED NORMALLY" in path.read_text(encoding="utf-8")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON: tempfile in same dir, then os.replace.

    Same-directory tempfile guarantees the rename is on the same filesystem,
    so os.replace is atomic on POSIX and (since Python 3.3) on Windows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
