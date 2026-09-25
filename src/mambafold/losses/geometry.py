"""Memory-bounded stereochemical losses for atom14 protein coordinates.

Coordinates in :mod:`mambafold` are normalized by ``COORD_SCALE``.  Every
public distance in this module is expressed in Angstroms.

The clash term builds a detached residue-neighbour list from conservative
per-residue bounding spheres and differentiates only candidate atom pairs.
Gradient checkpointing bounds the retained pair workspace.  No dense
differentiable atom-pair tensor is retained.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from mambafold.data.constants import (
    ATOM_NAME_TO_ID,
    COORD_SCALE,
    ID_TO_AA,
    MAX_ATOMS_PER_RES,
    RESIDUE_ATOM_TO_SLOT,
    RESIDUE_ATOMS,
)

# Covalent heavy-atom graph for the 20 standard amino acids.  Backbone bonds
# are shared; residue-specific entries describe side chains and rings.
_BACKBONE_BONDS = (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"))
_SIDECHAIN_BONDS: dict[str, tuple[tuple[str, str], ...]] = {
    "ALA": (),
    "ARG": (("CB", "CG"), ("CG", "CD"), ("CD", "NE"), ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2")),
    "ASN": (("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")),
    "ASP": (("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")),
    "CYS": (("CB", "SG"),),
    "GLN": (("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2")),
    "GLU": (("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")),
    "GLY": (),
    "HIS": (
        ("CB", "CG"),
        ("CG", "ND1"),
        ("CG", "CD2"),
        ("ND1", "CE1"),
        ("CD2", "NE2"),
        ("CE1", "NE2"),
    ),
    "ILE": (("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1")),
    "LEU": (("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")),
    "LYS": (("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")),
    "MET": (("CB", "CG"), ("CG", "SD"), ("SD", "CE")),
    "PHE": (
        ("CB", "CG"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CD1", "CE1"),
        ("CD2", "CE2"),
        ("CE1", "CZ"),
        ("CE2", "CZ"),
    ),
    "PRO": (("CB", "CG"), ("CG", "CD"), ("CD", "N")),
    "SER": (("CB", "OG"),),
    "THR": (("CB", "OG1"), ("CB", "CG2")),
    "TRP": (
        ("CB", "CG"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CD1", "NE1"),
        ("NE1", "CE2"),
        ("CD2", "CE2"),
        ("CD2", "CE3"),
        ("CE2", "CZ2"),
        ("CE3", "CZ3"),
        ("CZ2", "CH2"),
        ("CZ3", "CH2"),
    ),
    "TYR": (
        ("CB", "CG"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CD1", "CE1"),
        ("CD2", "CE2"),
        ("CE1", "CZ"),
        ("CE2", "CZ"),
        ("CZ", "OH"),
    ),
    "VAL": (("CB", "CG1"), ("CB", "CG2")),
    "UNK": (),
}

# Bondi-like heavy-atom radii.  Atom14 contains only C/N/O/S heavy atoms.
_VDW_RADIUS_A = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}
_N, _CA, _C = 0, 1, 2
_SULFUR_FLOOR_A = 2.03 - 1.0


def _residue_bonds(residue: str) -> tuple[tuple[str, str], ...]:
    names = set(RESIDUE_ATOMS[residue])
    backbone = tuple(pair for pair in _BACKBONE_BONDS if pair[0] in names and pair[1] in names)
    return backbone + _SIDECHAIN_BONDS[residue]


def _shortest_paths(n_atoms: int, bonds: tuple[tuple[int, int], ...]) -> list[list[int]]:
    inf = 99
    distance = [[inf] * n_atoms for _ in range(n_atoms)]
    for i in range(n_atoms):
        distance[i][i] = 0
    for i, j in bonds:
        distance[i][j] = distance[j][i] = 1
    for k in range(n_atoms):
        for i in range(n_atoms):
            for j in range(n_atoms):
                distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])
    return distance


def _build_topology_tables() -> dict[str, Tensor]:
    n_res_types = max(ID_TO_AA) + 1
    residue_bonds: list[list[tuple[int, int]]] = [[] for _ in range(n_res_types)]
    residue_angles: list[list[tuple[int, int, int]]] = [[] for _ in range(n_res_types)]
    direct_bond = torch.zeros(
        n_res_types, MAX_ATOMS_PER_RES, MAX_ATOMS_PER_RES, dtype=torch.bool
    )
    graph_steps = torch.full(
        (n_res_types, MAX_ATOMS_PER_RES, MAX_ATOMS_PER_RES), 99, dtype=torch.long
    )
    for type_id, residue in ID_TO_AA.items():
        slots = RESIDUE_ATOM_TO_SLOT[residue]
        bonds = tuple((slots[a], slots[b]) for a, b in _residue_bonds(residue))
        residue_bonds[type_id] = list(bonds)
        neighbours: dict[int, list[int]] = {i: [] for i in range(len(RESIDUE_ATOMS[residue]))}
        for i, j in bonds:
            neighbours[i].append(j)
            neighbours[j].append(i)
            direct_bond[type_id, i, j] = True
            direct_bond[type_id, j, i] = True
        angles = []
        for center, adjacent in neighbours.items():
            for left_index, left in enumerate(adjacent):
                for right in adjacent[left_index + 1 :]:
                    angles.append((left, center, right))
        residue_angles[type_id] = angles

        paths = _shortest_paths(len(RESIDUE_ATOMS[residue]), bonds)
        for i, row in enumerate(paths):
            for j, steps in enumerate(row):
                graph_steps[type_id, i, j] = steps
    max_bonds = max(map(len, residue_bonds))
    max_angles = max(map(len, residue_angles))
    bond_slots = torch.zeros(n_res_types, max_bonds, 2, dtype=torch.long)
    bond_mask = torch.zeros(n_res_types, max_bonds, dtype=torch.bool)
    angle_slots = torch.zeros(n_res_types, max_angles, 3, dtype=torch.long)
    angle_mask = torch.zeros(n_res_types, max_angles, dtype=torch.bool)
    for type_id in range(n_res_types):
        if residue_bonds[type_id]:
            values = torch.tensor(residue_bonds[type_id], dtype=torch.long)
            bond_slots[type_id, : len(values)] = values
            bond_mask[type_id, : len(values)] = True
        if residue_angles[type_id]:
            values = torch.tensor(residue_angles[type_id], dtype=torch.long)
            angle_slots[type_id, : len(values)] = values
            angle_mask[type_id, : len(values)] = True

    radii = torch.zeros(len(ATOM_NAME_TO_ID), dtype=torch.float32)
    sulfur = torch.zeros(len(ATOM_NAME_TO_ID), dtype=torch.bool)
    for atom_name, atom_id in ATOM_NAME_TO_ID.items():
        if atom_name != "PAD":
            radii[atom_id] = _VDW_RADIUS_A[atom_name[0]]
            sulfur[atom_id] = atom_name[0] == "S"
    return {
        "bond_slots": bond_slots,
        "bond_mask": bond_mask,
        "angle_slots": angle_slots,
        "angle_mask": angle_mask,
        "direct_bond": direct_bond,
        "graph_distance": graph_steps,
        "radii": radii,
        "sulfur": sulfur,
    }


_CPU_TABLES = _build_topology_tables()


@lru_cache(maxsize=16)
def _tables_for(device_type: str, device_index: int | None) -> dict[str, Tensor]:
    device = torch.device(device_type, device_index)
    return {name: value.to(device) for name, value in _CPU_TABLES.items()}


def _tables(device: torch.device) -> dict[str, Tensor]:
    return _tables_for(device.type, device.index)


def _angle_cosine(a: Tensor, center: Tensor, c: Tensor) -> Tensor:
    left = F.normalize(a - center, dim=-1, eps=1e-6)
    right = F.normalize(c - center, dim=-1, eps=1e-6)
    return (left * right).sum(dim=-1).clamp(-1.0, 1.0)


def _bond_angle_loss_one(
    pred_A: Tensor,
    true_A: Tensor,
    valid_mask: Tensor,
    res_type: Tensor,
    res_seq_nums: Tensor,
    chain_id: Tensor,
    tables: dict[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return bond loss, angle loss, bond MAE (A), angle MAE (degrees)."""
    length, atoms = valid_mask.shape
    flat_pred = pred_A.reshape(length * atoms, 3)
    flat_true = true_A.reshape(length * atoms, 3)
    flat_valid = valid_mask.reshape(-1)
    residue_base = torch.arange(length, device=pred_A.device)[:, None] * atoms

    bond_slots = tables["bond_slots"][res_type]
    bond_topology_mask = tables["bond_mask"][res_type]
    bond_indices = residue_base[:, None, None] + bond_slots
    bond_valid = (
        bond_topology_mask & flat_valid[bond_indices[..., 0]] & flat_valid[bond_indices[..., 1]]
    )
    bond_pairs = bond_indices[bond_valid]

    adjacent = (
        (chain_id[:-1] == chain_id[1:])
        & (res_seq_nums[1:] == res_seq_nums[:-1] + 1)
        & flat_valid[(torch.arange(length - 1, device=pred_A.device) * atoms) + 2]
        & flat_valid[(torch.arange(1, length, device=pred_A.device) * atoms)]
    )
    peptide_left = torch.arange(length - 1, device=pred_A.device)[adjacent] * atoms + 2
    peptide_right = torch.arange(1, length, device=pred_A.device)[adjacent] * atoms
    if peptide_left.numel():
        peptide_pairs = torch.stack((peptide_left, peptide_right), dim=-1)
        bond_pairs = torch.cat((bond_pairs, peptide_pairs), dim=0)

    if bond_pairs.numel():
        pred_distance = torch.linalg.vector_norm(
            flat_pred[bond_pairs[:, 0]] - flat_pred[bond_pairs[:, 1]], dim=-1
        )
        true_distance = torch.linalg.vector_norm(
            flat_true[bond_pairs[:, 0]] - flat_true[bond_pairs[:, 1]], dim=-1
        )
        bond_error = pred_distance - true_distance
        bond_loss = F.smooth_l1_loss(pred_distance, true_distance, beta=0.1, reduction="mean")
        bond_mae = bond_error.abs().mean().detach()
    else:
        bond_loss = pred_A.sum() * 0.0
        bond_mae = bond_loss.detach()

    angle_slots = tables["angle_slots"][res_type]
    angle_topology_mask = tables["angle_mask"][res_type]
    angle_indices = residue_base[:, None, None] + angle_slots
    angle_valid = angle_topology_mask
    for position in range(3):
        angle_valid = angle_valid & flat_valid[angle_indices[..., position]]
    angle_triples = angle_indices[angle_valid]

    # Three peptide-spanning backbone angles: CA-C-N, O-C-N and C-N-CA.
    left_residue = torch.arange(length - 1, device=pred_A.device)[adjacent]
    right_residue = left_residue + 1
    if left_residue.numel():
        cross = torch.stack(
            (
                torch.stack(
                    (left_residue * atoms + 1, left_residue * atoms + 2, right_residue * atoms),
                    dim=-1,
                ),
                torch.stack(
                    (left_residue * atoms + 3, left_residue * atoms + 2, right_residue * atoms),
                    dim=-1,
                ),
                torch.stack(
                    (left_residue * atoms + 2, right_residue * atoms, right_residue * atoms + 1),
                    dim=-1,
                ),
            ),
            dim=1,
        ).reshape(-1, 3)
        cross_valid = flat_valid[cross].all(dim=-1)
        angle_triples = torch.cat((angle_triples, cross[cross_valid]), dim=0)

    if angle_triples.numel():
        pred_cos = _angle_cosine(
            flat_pred[angle_triples[:, 0]],
            flat_pred[angle_triples[:, 1]],
            flat_pred[angle_triples[:, 2]],
        )
        true_cos = _angle_cosine(
            flat_true[angle_triples[:, 0]],
            flat_true[angle_triples[:, 1]],
            flat_true[angle_triples[:, 2]],
        )
        angle_loss = (pred_cos - true_cos).square().mean()
        angle_mae = torch.rad2deg(
            (
                torch.acos(pred_cos.detach().clamp(-1 + 1e-6, 1 - 1e-6))
                - torch.acos(true_cos.detach().clamp(-1 + 1e-6, 1 - 1e-6))
            ).abs()
        ).mean()
    else:
        angle_loss = pred_A.sum() * 0.0
        angle_mae = angle_loss.detach()
    return bond_loss, angle_loss, bond_mae, angle_mae


def _safe_pair_distance(delta: Tensor) -> Tensor:
    """Euclidean distance with a finite descent direction at coincidence."""
    coincident = delta.detach().square().sum(dim=-1) < 1e-12
    fallback = delta.new_tensor((1e-3, 0.0, 0.0))
    return torch.linalg.vector_norm(delta + coincident.unsqueeze(-1) * fallback, dim=-1)


def _clash_terms(
    pair_distance_A: Tensor,
    pair_floor_A: Tensor,
    pair_valid: Tensor,
    *,
    margin_A: float,
    huber_delta_A: float,
    soft_count_tau_A: float,
) -> Tensor:
    """Return penalty, hard count, soft count and summed hard overlap."""
    penetration = pair_floor_A - pair_distance_A
    violation = F.relu(penetration + margin_A)
    penalty = torch.where(
        violation <= huber_delta_A,
        0.5 * violation.square() / huber_delta_A,
        violation - 0.5 * huber_delta_A,
    )
    valid_f = pair_valid.to(pair_distance_A.dtype)
    hard = (penetration > 0.0) & pair_valid
    soft = torch.sigmoid(penetration / soft_count_tau_A) * valid_f
    return torch.stack(
        (
            (penalty * valid_f).sum(),
            hard.sum().to(pair_distance_A.dtype),
            soft.sum().detach(),
            (F.relu(penetration) * valid_f).sum().detach(),
        )
    )


def _intra_residue_clash_terms(
    xyz_A: Tensor,
    residue_type: Tensor,
    radii_A: Tensor,
    valid: Tensor,
    direct_bond: Tensor,
    atom_upper: Tensor,
    margin_A: float,
    huber_delta_A: float,
    soft_count_tau_A: float,
) -> Tensor:
    delta = xyz_A.unsqueeze(2) - xyz_A.unsqueeze(1)
    pair_distance = _safe_pair_distance(delta)
    pair_floor = radii_A.unsqueeze(2) + radii_A.unsqueeze(1)
    pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1) & atom_upper.unsqueeze(0)
    # OpenStructure/AlphaFold-style clash validation excludes direct covalent
    # 1-2 bonds only.  1-3 and 1-4 pairs remain active; correct angles and
    # torsions put them outside the deliberately tolerant clash floor.
    pair_valid &= ~direct_bond[residue_type]
    return _clash_terms(
        pair_distance,
        pair_floor,
        pair_valid,
        margin_A=margin_A,
        huber_delta_A=huber_delta_A,
        soft_count_tau_A=soft_count_tau_A,
    )


def _inter_residue_clash_terms(
    xyz_A: Tensor,
    radii_A: Tensor,
    sulfur: Tensor,
    valid: Tensor,
    left_residue: Tensor,
    right_residue: Tensor,
    sequential: Tensor,
    peptide_bond: Tensor,
    margin_A: float,
    huber_delta_A: float,
    soft_count_tau_A: float,
) -> Tensor:
    left_xyz = xyz_A[left_residue]
    right_xyz = xyz_A[right_residue]
    pair_distance = _safe_pair_distance(left_xyz.unsqueeze(2) - right_xyz.unsqueeze(1))
    pair_floor = radii_A[left_residue].unsqueeze(2) + radii_A[right_residue].unsqueeze(1)
    pair_valid = valid[left_residue].unsqueeze(2) & valid[right_residue].unsqueeze(1)
    pair_valid &= ~(sequential[:, None, None] & peptide_bond[None])

    # OpenStructure's heavy-atom table gives every S-S pair a special
    # 2.03-1.00 = 1.03-A floor. This covers potential disulfides without
    # incorrectly exempting them and also applies to MET SD pairs.
    sulfur_pair = sulfur[left_residue].unsqueeze(2) & sulfur[right_residue].unsqueeze(1)
    pair_floor = torch.where(
        sulfur_pair,
        pair_floor.new_tensor(_SULFUR_FLOOR_A),
        pair_floor,
    )
    return _clash_terms(
        pair_distance,
        pair_floor,
        pair_valid,
        margin_A=margin_A,
        huber_delta_A=huber_delta_A,
        soft_count_tau_A=soft_count_tau_A,
    )


def _clash_loss_one(
    pred_A: Tensor,
    atom_mask: Tensor,
    res_mask: Tensor,
    res_type: Tensor,
    atom_type: Tensor,
    res_seq_nums: Tensor,
    chain_id: Tensor,
    tables: dict[str, Tensor],
    *,
    overlap_tolerance_A: float,
    margin_A: float,
    huber_delta_A: float,
    soft_count_tau_A: float,
    pair_chunk_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    valid = atom_mask & res_mask.unsqueeze(-1)
    atom_count = valid.sum().clamp(min=1).to(pred_A.dtype)
    if valid.sum() < 2:
        zero = pred_A.sum() * 0.0
        return zero, zero.detach(), zero.detach(), zero.detach()

    radius = tables["radii"][atom_type]
    sulfur = tables["sulfur"][atom_type]
    floor_radius = radius - overlap_tolerance_A / 2.0
    atom_upper = torch.triu(
        torch.ones(
            atom_mask.shape[-1], atom_mask.shape[-1], dtype=torch.bool, device=pred_A.device
        ),
        diagonal=1,
    )
    peptide_bond = torch.zeros_like(atom_upper)
    peptide_bond[_C, _N] = True
    intra_args = (
        pred_A,
        res_type,
        floor_radius,
        valid,
        tables["direct_bond"],
        atom_upper,
        margin_A,
        huber_delta_A,
        soft_count_tau_A,
    )
    if pred_A.requires_grad:
        totals = checkpoint(_intra_residue_clash_terms, *intra_args, use_reentrant=False)
    else:
        totals = _intra_residue_clash_terms(*intra_args)

    # A residue is bounded by its CA-centred maximum atom extent. Candidate
    # residue pairs therefore cannot omit a possible clash, even for a long
    # extended side chain. Only this small LxL discovery matrix is materialized.
    ca = pred_A[:, _CA]
    ca_valid = valid[:, _CA]
    with torch.no_grad():
        ca_distance = torch.cdist(ca.detach(), ca.detach())
        atom_extent = (
            torch.linalg.vector_norm(pred_A.detach() - ca.detach().unsqueeze(1), dim=-1)
            .masked_fill(~valid, float("-inf"))
            .amax(dim=-1)
        )
        max_floor_A = 2.0 * max(_VDW_RADIUS_A.values()) - overlap_tolerance_A
        candidate_cutoff = (
            atom_extent.unsqueeze(1)
            + atom_extent.unsqueeze(0)
            + max_floor_A
            + margin_A
        )
        residue_upper = torch.triu(
            torch.ones(
                pred_A.shape[0], pred_A.shape[0], dtype=torch.bool, device=pred_A.device
            ),
            diagonal=1,
        )
        candidates = (
            residue_upper
            & ca_valid.unsqueeze(1)
            & ca_valid.unsqueeze(0)
            & (ca_distance < candidate_cutoff)
        ).nonzero(as_tuple=False)

    for start in range(0, candidates.shape[0], pair_chunk_size):
        pair = candidates[start : start + pair_chunk_size]
        left_residue, right_residue = pair[:, 0], pair[:, 1]
        sequential = (
            (chain_id[left_residue] == chain_id[right_residue])
            & (res_seq_nums[right_residue] == res_seq_nums[left_residue] + 1)
        )
        args = (
            pred_A,
            floor_radius,
            sulfur,
            valid,
            left_residue,
            right_residue,
            sequential,
            peptide_bond,
            margin_A,
            huber_delta_A,
            soft_count_tau_A,
        )
        if pred_A.requires_grad:
            totals = totals + checkpoint(
                _inter_residue_clash_terms, *args, use_reentrant=False
            )
        else:
            totals = totals + _inter_residue_clash_terms(*args)

    penalty = totals[0] / atom_count
    hard_per_1000 = totals[1].detach() * (1000.0 / atom_count)
    soft_per_1000 = totals[2] * (1000.0 / atom_count)
    mean_overlap_A = totals[3] / totals[1].detach().clamp(min=1.0)
    return penalty, hard_per_1000, soft_per_1000, mean_overlap_A


def stereochemical_losses(
    pred: Tensor,
    true: Tensor,
    valid_mask: Tensor,
    atom_mask: Tensor,
    res_mask: Tensor,
    res_type: Tensor,
    atom_type: Tensor,
    res_seq_nums: Tensor,
    chain_id: Tensor,
    *,
    clash_overlap_tolerance_A: float = 1.5,
    clash_margin_A: float = 0.1,
    clash_huber_delta_A: float = 0.25,
    clash_soft_count_tau_A: float = 0.05,
    clash_pair_chunk_size: int = 256,
) -> dict[str, Tensor]:
    """Return per-example bond, angle and non-bonded clash diagnostics.

    ``pred`` and ``true`` have shape ``[B, L, A, 3]`` in normalized model
    units.  Losses are returned as ``[B]`` tensors so the caller can apply the
    same time-dependent weighting used for all-atom lDDT.
    """
    if clash_pair_chunk_size <= 0:
        raise ValueError("clash_pair_chunk_size must be positive")
    if clash_overlap_tolerance_A < 0:
        raise ValueError("clash_overlap_tolerance_A must be non-negative")
    if clash_margin_A < 0:
        raise ValueError("clash_margin_A must be non-negative")
    if clash_huber_delta_A <= 0:
        raise ValueError("clash_huber_delta_A must be positive")
    if clash_soft_count_tau_A <= 0:
        raise ValueError("clash_soft_count_tau_A must be positive")
    if pred.shape != true.shape or pred.shape[:-1] != valid_mask.shape:
        raise ValueError("coordinate and valid-mask shapes do not match")
    if atom_mask.shape != valid_mask.shape or res_mask.shape != valid_mask.shape[:2]:
        raise ValueError("atom/residue mask shapes do not match coordinates")

    tables = _tables(pred.device)
    pred_A = pred.float() * COORD_SCALE
    true_A = true.float() * COORD_SCALE
    values: dict[str, list[Tensor]] = {
        "bond": [],
        "angle": [],
        "clash": [],
        "bond_mae_A": [],
        "angle_mae_deg": [],
        "clashes_per_1000_atoms": [],
        "soft_clashes_per_1000_atoms": [],
        "mean_clash_overlap_A": [],
    }
    for batch_index in range(pred.shape[0]):
        bond, angle, bond_mae, angle_mae = _bond_angle_loss_one(
            pred_A[batch_index],
            true_A[batch_index],
            valid_mask[batch_index],
            res_type[batch_index],
            res_seq_nums[batch_index],
            chain_id[batch_index],
            tables,
        )
        clash, clash_count, soft_clash_count, mean_overlap = _clash_loss_one(
            pred_A[batch_index],
            atom_mask[batch_index],
            res_mask[batch_index],
            res_type[batch_index],
            atom_type[batch_index],
            res_seq_nums[batch_index],
            chain_id[batch_index],
            tables,
            overlap_tolerance_A=clash_overlap_tolerance_A,
            margin_A=clash_margin_A,
            huber_delta_A=clash_huber_delta_A,
            soft_count_tau_A=clash_soft_count_tau_A,
            pair_chunk_size=clash_pair_chunk_size,
        )
        values["bond"].append(bond)
        values["angle"].append(angle)
        values["clash"].append(clash)
        values["bond_mae_A"].append(bond_mae)
        values["angle_mae_deg"].append(angle_mae)
        values["clashes_per_1000_atoms"].append(clash_count)
        values["soft_clashes_per_1000_atoms"].append(soft_clash_count)
        values["mean_clash_overlap_A"].append(mean_overlap)
    return {name: torch.stack(items) for name, items in values.items()}
