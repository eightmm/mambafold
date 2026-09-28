"""Minimal PDB/mmCIF writers for MambaFold atom-slot predictions."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from mambafold.data.constants import ID_TO_AA, RESIDUE_ATOMS

_CHAIN_IDS = tuple(
    [chr(code) for code in range(ord("A"), ord("Z") + 1)]
    + [chr(code) for code in range(ord("a"), ord("z") + 1)]
    + [chr(code) for code in range(ord("0"), ord("9") + 1)]
)
_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")
_PDB_COORD_MIN = -999.999
_PDB_COORD_MAX = 9999.999


def _chain_letter(chain: int) -> str:
    return _CHAIN_IDS[chain % len(_CHAIN_IDS)]


def _boundaries(chain_id: np.ndarray) -> list[int]:
    result = [0]
    result.extend(
        index for index in range(1, len(chain_id)) if chain_id[index] != chain_id[index - 1]
    )
    result.append(len(chain_id))
    return result


def _validated_coordinates(
    coords_A: np.ndarray,
    atom_mask: np.ndarray,
    *,
    pdb_fixed_width: bool,
) -> np.ndarray:
    coords = np.asarray(coords_A)
    selected = coords[np.asarray(atom_mask, dtype=bool)]
    if not np.isfinite(selected).all():
        raise ValueError("structure coordinates contain NaN or infinity")
    if pdb_fixed_width and selected.size:
        minimum = float(selected.min())
        maximum = float(selected.max())
        if minimum < _PDB_COORD_MIN or maximum > _PDB_COORD_MAX:
            raise ValueError(
                "coordinates do not fit PDB 8.3 fields: "
                f"range=[{minimum:.3f}, {maximum:.3f}]; use mmCIF or a "
                "converged sampling trajectory"
            )
    return coords


def write_pdb(
    coords_A: np.ndarray,
    res_type_ids: np.ndarray,
    atom_mask: np.ndarray,
    b_factors: np.ndarray,
    chain_id: np.ndarray,
    path: str | Path,
) -> None:
    """Write atom-slot coordinates to a parseable PDB file."""
    coords_A = _validated_coordinates(coords_A, atom_mask, pdb_fixed_width=True)
    lines: list[str] = []
    serial = 1
    boundaries = _boundaries(chain_id)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        letter = _chain_letter(int(chain_id[start]))
        last_residue = "UNK"
        local_index = 0
        for local_index, residue_index in enumerate(range(start, stop), start=1):
            residue = ID_TO_AA.get(int(res_type_ids[residue_index]), "UNK")
            last_residue = residue
            for slot, atom_name in enumerate(RESIDUE_ATOMS.get(residue, RESIDUE_ATOMS["UNK"])):
                if slot >= atom_mask.shape[1] or not atom_mask[residue_index, slot]:
                    continue
                x, y, z = (float(value) for value in coords_A[residue_index, slot])
                b_factor = float(np.clip(b_factors[residue_index, slot], -99.99, 999.99))
                atom_field = atom_name if len(atom_name) >= 4 else f" {atom_name:<3s}"
                lines.append(
                    f"ATOM  {serial:>5d} {atom_field:<4s} {residue:>3s} {letter}"
                    f"{local_index:>4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00{b_factor:6.2f}"
                    f"          {atom_name[0]:>2s}\n"
                )
                serial += 1
        lines.append(f"TER   {serial:>5d}      {last_residue:>3s} {letter}{local_index:>4d}\n")
        serial += 1
    lines.append("END\n")
    Path(path).write_text("".join(lines))


def write_mmcif(
    coords_A: np.ndarray,
    res_type_ids: np.ndarray,
    atom_mask: np.ndarray,
    b_factors: np.ndarray,
    chain_id: np.ndarray,
    path: str | Path,
    *,
    entry_id: str,
) -> None:
    """Write atom-slot coordinates to a parseable PDBx/mmCIF file."""
    coords_A = _validated_coordinates(coords_A, atom_mask, pdb_fixed_width=False)
    block_id = _SAFE_ID.sub("_", entry_id).strip("._") or "prediction"
    lines = [
        f"data_{block_id}\n",
        "#\n",
        f"_entry.id {block_id}\n",
        "#\n",
        "loop_\n",
        "_atom_site.group_PDB\n",
        "_atom_site.id\n",
        "_atom_site.type_symbol\n",
        "_atom_site.label_atom_id\n",
        "_atom_site.label_alt_id\n",
        "_atom_site.label_comp_id\n",
        "_atom_site.label_asym_id\n",
        "_atom_site.label_entity_id\n",
        "_atom_site.label_seq_id\n",
        "_atom_site.pdbx_PDB_ins_code\n",
        "_atom_site.Cartn_x\n",
        "_atom_site.Cartn_y\n",
        "_atom_site.Cartn_z\n",
        "_atom_site.occupancy\n",
        "_atom_site.B_iso_or_equiv\n",
        "_atom_site.pdbx_formal_charge\n",
        "_atom_site.auth_seq_id\n",
        "_atom_site.auth_comp_id\n",
        "_atom_site.auth_asym_id\n",
        "_atom_site.auth_atom_id\n",
        "_atom_site.pdbx_PDB_model_num\n",
    ]
    serial = 1
    boundaries = _boundaries(chain_id)
    for entity_index, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
        letter = _chain_letter(int(chain_id[start]))
        for local_index, residue_index in enumerate(range(start, stop), start=1):
            residue = ID_TO_AA.get(int(res_type_ids[residue_index]), "UNK")
            for slot, atom_name in enumerate(RESIDUE_ATOMS.get(residue, RESIDUE_ATOMS["UNK"])):
                if slot >= atom_mask.shape[1] or not atom_mask[residue_index, slot]:
                    continue
                x, y, z = (float(value) for value in coords_A[residue_index, slot])
                b_factor = float(np.clip(b_factors[residue_index, slot], -99.99, 999.99))
                lines.append(
                    f"ATOM {serial} {atom_name[0]} {atom_name} . {residue} {letter} "
                    f"{entity_index} {local_index} ? {x:.3f} {y:.3f} {z:.3f} 1.00 "
                    f"{b_factor:.2f} ? {local_index} {residue} {letter} {atom_name} 1\n"
                )
                serial += 1
    lines.append("#\n")
    Path(path).write_text("".join(lines))
