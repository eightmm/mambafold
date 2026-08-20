"""Ground-truth-free stereochemical energies for inference guidance.

The model predicts Cartesian atom coordinates directly, so these terms are
deliberately conservative barriers rather than a force field.  Reference bond
and angle statistics come from the packaged OpenStructure Engh-Huber table.
All public functions accept coordinates in MambaFold's normalized units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from mambafold.data.constants import (
    AA_TO_ID,
    COORD_SCALE,
    ID_TO_AA,
    MAX_ATOMS_PER_RES,
    RESIDUE_ATOM_TO_SLOT,
    RESIDUE_ATOMS,
)

_N, _CA, _C, _O, _CB = 0, 1, 2, 3, 4
_VDW_RADIUS_A = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}
_CYS_SG = RESIDUE_ATOM_TO_SLOT["CYS"]["SG"]
_DISULFIDE_FLOOR_A = 2.03 - 1.0
_MAX_OST_CLASH_FLOOR_A = 2.0 * _VDW_RADIUS_A["S"] - 1.5


@dataclass(frozen=True)
class _Bond:
    atom1: str
    atom2: str
    mean_A: float
    std_A: float


@dataclass(frozen=True)
class _Angle:
    atom1: str
    atom2: str
    atom3: str
    mean_rad: float
    std_rad: float


@dataclass(frozen=True)
class _Topology:
    bond_i: Tensor
    bond_j: Tensor
    bond_mean_A: Tensor
    bond_std_A: Tensor
    bond_mask: Tensor
    angle_i: Tensor
    angle_j: Tensor
    angle_k: Tensor
    angle_cos: Tensor
    angle_cos_std: Tensor
    angle_mask: Tensor
    bond_exclusion: Tensor
    same_res_exclusion: Tensor
    atom_radius_A: Tensor
    atom_element_id: Tensor
    nonbonded_floor_A: Tensor


@lru_cache(maxsize=1)
def _reference_specs() -> tuple[
    dict[str, list[_Bond]],
    dict[str, list[_Angle]],
    dict[tuple[str, str], float],
]:
    text = (
        files("mambafold.resources")
        .joinpath("stereo_chemical_props.txt")
        .read_text(encoding="utf-8")
    )
    bonds: dict[str, list[_Bond]] = {}
    angles: dict[str, list[_Angle]] = {}
    nonbonded_floors: dict[tuple[str, str], float] = {}
    section = "header"
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("Bond"):
            section = "bond"
            continue
        if line.startswith("Angle"):
            section = "angle"
            continue
        if line.startswith("Non-bonded"):
            section = "nonbonded"
            continue
        if line == "-":
            if section == "nonbonded":
                break
            section = "between"
            continue
        fields = line.split()
        if section == "bond" and len(fields) == 4:
            atom1, atom2 = fields[0].split("-")
            residue = fields[1]
            bonds.setdefault(residue, []).append(
                _Bond(atom1, atom2, float(fields[2]), float(fields[3]))
            )
        elif section == "angle" and len(fields) == 4:
            atom1, atom2, atom3 = fields[0].split("-")
            residue = fields[1]
            angles.setdefault(residue, []).append(
                _Angle(
                    atom1,
                    atom2,
                    atom3,
                    math.radians(float(fields[2])),
                    math.radians(float(fields[3])),
                )
            )
        elif section == "nonbonded" and len(fields) == 3:
            first, second = fields[0].split("-")
            key = tuple(sorted((first, second)))
            nonbonded_floors[key] = float(fields[1]) - float(fields[2])
    return bonds, angles, nonbonded_floors


@lru_cache(maxsize=1)
def _cpu_topology() -> _Topology:
    bonds, angles, nonbonded_floors = _reference_specs()
    n_types = max(ID_TO_AA) + 1
    max_bonds = max(len(bonds.get(ID_TO_AA.get(i, "UNK"), [])) for i in range(n_types))
    max_angles = max(len(angles.get(ID_TO_AA.get(i, "UNK"), [])) for i in range(n_types))

    bond_i = torch.zeros(n_types, max_bonds, dtype=torch.long)
    bond_j = torch.zeros_like(bond_i)
    bond_mean = torch.zeros(n_types, max_bonds)
    bond_std = torch.ones(n_types, max_bonds)
    bond_mask = torch.zeros(n_types, max_bonds, dtype=torch.bool)
    angle_i = torch.zeros(n_types, max_angles, dtype=torch.long)
    angle_j = torch.zeros_like(angle_i)
    angle_k = torch.zeros_like(angle_i)
    angle_cos = torch.zeros(n_types, max_angles)
    angle_cos_std = torch.ones(n_types, max_angles)
    angle_mask = torch.zeros(n_types, max_angles, dtype=torch.bool)
    exclusions = torch.eye(MAX_ATOMS_PER_RES, dtype=torch.bool).expand(n_types, -1, -1).clone()
    bond_exclusion = torch.zeros_like(exclusions)
    radii = torch.zeros(n_types, MAX_ATOMS_PER_RES)
    element_names = tuple(sorted(_VDW_RADIUS_A))
    element_to_id = {name: index for index, name in enumerate(element_names)}
    element_ids = torch.zeros(n_types, MAX_ATOMS_PER_RES, dtype=torch.long)
    nonbonded_floor = torch.zeros(len(element_names), len(element_names))
    for first, first_id in element_to_id.items():
        for second, second_id in element_to_id.items():
            key = tuple(sorted((first, second)))
            if key == ("S", "S"):
                # OpenStructure applies the 1.03 A table value only to a pair
                # of CYS SG atoms. Other sulfur pairs use ordinary VDW radii.
                value = _VDW_RADIUS_A[first] + _VDW_RADIUS_A[second] - 1.5
            else:
                value = nonbonded_floors[key]
            nonbonded_floor[first_id, second_id] = value

    for res_id in range(n_types):
        residue = ID_TO_AA.get(res_id, "UNK")
        slot_map = RESIDUE_ATOM_TO_SLOT.get(residue, {})
        adjacency = torch.zeros(MAX_ATOMS_PER_RES, MAX_ATOMS_PER_RES, dtype=torch.bool)
        for index, spec in enumerate(bonds.get(residue, [])):
            if spec.atom1 not in slot_map or spec.atom2 not in slot_map:
                continue
            i, j = slot_map[spec.atom1], slot_map[spec.atom2]
            bond_i[res_id, index], bond_j[res_id, index] = i, j
            bond_mean[res_id, index], bond_std[res_id, index] = spec.mean_A, spec.std_A
            bond_mask[res_id, index] = True
            adjacency[i, j] = adjacency[j, i] = True
        for index, spec in enumerate(angles.get(residue, [])):
            if any(atom not in slot_map for atom in (spec.atom1, spec.atom2, spec.atom3)):
                continue
            i, j, k = (slot_map[atom] for atom in (spec.atom1, spec.atom2, spec.atom3))
            angle_i[res_id, index], angle_j[res_id, index], angle_k[res_id, index] = i, j, k
            angle_cos[res_id, index] = math.cos(spec.mean_rad)
            angle_cos_std[res_id, index] = max(abs(math.sin(spec.mean_rad)) * spec.std_rad, 1e-3)
            angle_mask[res_id, index] = True
        # Exclude 1-2 and 1-3 pairs from non-bonded repulsion.
        two_hop = (adjacency.float() @ adjacency.float()) > 0
        bond_exclusion[res_id] = adjacency
        exclusions[res_id] |= adjacency | two_hop
        for slot, atom in enumerate(RESIDUE_ATOMS.get(residue, [])):
            radii[res_id, slot] = _VDW_RADIUS_A.get(atom[0], 0.0)
            element_ids[res_id, slot] = element_to_id[atom[0]]

    return _Topology(
        bond_i,
        bond_j,
        bond_mean,
        bond_std,
        bond_mask,
        angle_i,
        angle_j,
        angle_k,
        angle_cos,
        angle_cos_std,
        angle_mask,
        bond_exclusion,
        exclusions,
        radii,
        element_ids,
        nonbonded_floor,
    )


_DEVICE_TOPOLOGY: dict[tuple[str, int | None], _Topology] = {}


def _topology(device: torch.device) -> _Topology:
    key = (device.type, device.index)
    if key not in _DEVICE_TOPOLOGY:
        cpu = _cpu_topology()
        _DEVICE_TOPOLOGY[key] = _Topology(
            bond_i=cpu.bond_i.to(device),
            bond_j=cpu.bond_j.to(device),
            bond_mean_A=cpu.bond_mean_A.to(device),
            bond_std_A=cpu.bond_std_A.to(device),
            bond_mask=cpu.bond_mask.to(device),
            angle_i=cpu.angle_i.to(device),
            angle_j=cpu.angle_j.to(device),
            angle_k=cpu.angle_k.to(device),
            angle_cos=cpu.angle_cos.to(device),
            angle_cos_std=cpu.angle_cos_std.to(device),
            angle_mask=cpu.angle_mask.to(device),
            bond_exclusion=cpu.bond_exclusion.to(device),
            same_res_exclusion=cpu.same_res_exclusion.to(device),
            atom_radius_A=cpu.atom_radius_A.to(device),
            atom_element_id=cpu.atom_element_id.to(device),
            nonbonded_floor_A=cpu.nonbonded_floor_A.to(device),
        )
    return _DEVICE_TOPOLOGY[key]


def _gather_atoms(coords: Tensor, slots: Tensor) -> Tensor:
    return torch.gather(
        coords,
        2,
        slots.unsqueeze(-1).expand(*slots.shape, 3),
    )


def _gather_mask(mask: Tensor, slots: Tensor) -> Tensor:
    return torch.gather(mask, 2, slots)


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp(min=1)


def _z_barrier(z: Tensor, tolerance_z: float) -> Tensor:
    excess = F.relu(z.abs() - tolerance_z)
    return F.huber_loss(excess, torch.zeros_like(excess), reduction="none", delta=2.0)


def _adjacent_mask(
    res_mask: Tensor,
    chain_id: Tensor | None,
    res_seq_nums: Tensor | None,
) -> Tensor:
    adjacent = res_mask[:, :-1] & res_mask[:, 1:]
    if chain_id is not None:
        adjacent &= chain_id[:, :-1] == chain_id[:, 1:]
    if res_seq_nums is not None:
        adjacent &= (res_seq_nums[:, 1:] - res_seq_nums[:, :-1]) == 1
    return adjacent


def covalent_geometry_loss(
    pred_coords: Tensor,
    res_type: Tensor,
    atom_mask: Tensor,
    res_mask: Tensor,
    chain_id: Tensor | None = None,
    res_seq_nums: Tensor | None = None,
    tolerance_z: float = 3.0,
) -> Tensor:
    """All standard-residue bonds/angles plus the peptide linkage geometry."""
    coords = pred_coords.float() * COORD_SCALE
    topology = _topology(coords.device)
    res_type = res_type.clamp(0, topology.bond_i.shape[0] - 1)

    bi, bj = topology.bond_i[res_type], topology.bond_j[res_type]
    bond_valid = (
        topology.bond_mask[res_type]
        & _gather_mask(atom_mask, bi)
        & _gather_mask(atom_mask, bj)
        & res_mask.unsqueeze(-1)
    )
    bond_dist = torch.linalg.vector_norm(
        _gather_atoms(coords, bi) - _gather_atoms(coords, bj), dim=-1
    )
    bond_z = (bond_dist - topology.bond_mean_A[res_type]) / topology.bond_std_A[res_type]
    bond_sum = (_z_barrier(bond_z, tolerance_z) * bond_valid).sum()
    count = bond_valid.sum()

    ai = topology.angle_i[res_type]
    aj = topology.angle_j[res_type]
    ak = topology.angle_k[res_type]
    angle_valid = (
        topology.angle_mask[res_type]
        & _gather_mask(atom_mask, ai)
        & _gather_mask(atom_mask, aj)
        & _gather_mask(atom_mask, ak)
        & res_mask.unsqueeze(-1)
    )
    va = F.normalize(_gather_atoms(coords, ai) - _gather_atoms(coords, aj), dim=-1)
    vc = F.normalize(_gather_atoms(coords, ak) - _gather_atoms(coords, aj), dim=-1)
    observed_cos = (va * vc).sum(dim=-1).clamp(-1.0, 1.0)
    angle_z = (observed_cos - topology.angle_cos[res_type]) / topology.angle_cos_std[res_type]
    total = bond_sum + (_z_barrier(angle_z, tolerance_z) * angle_valid).sum()
    count = count + angle_valid.sum()

    if coords.shape[1] >= 2:
        adjacent = _adjacent_mask(res_mask, chain_id, res_seq_nums)
        peptide_atoms = (
            atom_mask[:, :-1, _CA]
            & atom_mask[:, :-1, _C]
            & atom_mask[:, 1:, _N]
            & atom_mask[:, 1:, _CA]
        )
        peptide_valid = adjacent & peptide_atoms
        c_i = coords[:, :-1, _C]
        ca_i = coords[:, :-1, _CA]
        n_j = coords[:, 1:, _N]
        ca_j = coords[:, 1:, _CA]
        is_pro = res_type[:, 1:] == AA_TO_ID["PRO"]
        cn_mean = torch.where(is_pro, 1.341, 1.329)
        cn_std = torch.where(is_pro, 0.016, 0.014)
        cn_z = (torch.linalg.vector_norm(c_i - n_j, dim=-1) - cn_mean) / cn_std
        total = total + (_z_barrier(cn_z, tolerance_z) * peptide_valid).sum()
        count = count + peptide_valid.sum()

        # C(i)-N(i+1)-CA(i+1): cos=-0.5203 +- 0.0353.
        c_n = F.normalize(c_i - n_j, dim=-1)
        ca_n = F.normalize(ca_j - n_j, dim=-1)
        z_c_n_ca = ((c_n * ca_n).sum(-1) + 0.5203) / 0.0353
        # CA(i)-C(i)-N(i+1): cos=-0.4473 +- 0.0311.
        ca_c = F.normalize(ca_i - c_i, dim=-1)
        n_c = F.normalize(n_j - c_i, dim=-1)
        z_ca_c_n = ((ca_c * n_c).sum(-1) + 0.4473) / 0.0311
        total = (
            total
            + (
                (_z_barrier(z_c_n_ca, tolerance_z) + _z_barrier(z_ca_c_n, tolerance_z))
                * peptide_valid
            ).sum()
        )
        count = count + 2 * peptide_valid.sum()

    return total / count.clamp(min=1).to(total.dtype)


def peptide_planarity_loss(
    pred_coords: Tensor,
    atom_mask: Tensor,
    res_mask: Tensor,
    chain_id: Tensor | None = None,
    res_seq_nums: Tensor | None = None,
) -> Tensor:
    """Penalize twisted peptide units while allowing both cis and trans omega."""
    if pred_coords.shape[1] < 2:
        return pred_coords.new_zeros(())
    coords = pred_coords.float()
    valid = _adjacent_mask(res_mask, chain_id, res_seq_nums)
    valid &= (
        atom_mask[:, :-1, _CA]
        & atom_mask[:, :-1, _C]
        & atom_mask[:, :-1, _O]
        & atom_mask[:, 1:, _N]
        & atom_mask[:, 1:, _CA]
    )
    ca_i, c_i, o_i = coords[:, :-1, _CA], coords[:, :-1, _C], coords[:, :-1, _O]
    n_j, ca_j = coords[:, 1:, _N], coords[:, 1:, _CA]

    normal_c = F.normalize(torch.linalg.cross(ca_i - c_i, o_i - c_i, dim=-1), dim=-1)
    normal_n = F.normalize(torch.linalg.cross(c_i - n_j, ca_j - n_j, dim=-1), dim=-1)
    out_c = (normal_c * F.normalize(n_j - c_i, dim=-1)).sum(-1).square()
    out_n = (normal_n * F.normalize(o_i - n_j, dim=-1)).sum(-1).square()
    omega_cos = (normal_c * normal_n).sum(-1).clamp(-1.0, 1.0)
    omega_twist = 1.0 - omega_cos.square()
    return _masked_mean((out_c + out_n + omega_twist) / 3.0, valid)


def _signed_unit_volume(center: Tensor, first: Tensor, second: Tensor, third: Tensor) -> Tensor:
    a = F.normalize(first - center, dim=-1)
    b = F.normalize(second - center, dim=-1)
    c = F.normalize(third - center, dim=-1)
    return (torch.linalg.cross(a, b, dim=-1) * c).sum(-1)


def chemical_chirality_loss(
    pred_coords: Tensor,
    res_type: Tensor,
    atom_mask: Tensor,
    res_mask: Tensor,
    margin: float = 0.1,
) -> Tensor:
    """Known L-amino-acid C-alpha and Thr/Ile C-beta stereochemistry."""
    coords = pred_coords.float()
    ca_valid = (
        res_mask
        & (res_type != AA_TO_ID["GLY"])
        & atom_mask[..., _N]
        & atom_mask[..., _CA]
        & atom_mask[..., _C]
        & atom_mask[..., _CB]
    )
    ca_volume = _signed_unit_volume(
        coords[..., _CA, :],
        coords[..., _N, :],
        coords[..., _C, :],
        coords[..., _CB, :],
    )
    total = (F.relu(margin - ca_volume).square() * ca_valid).sum()
    count = ca_valid.sum()

    for residue, atom1, atom2 in (("THR", "OG1", "CG2"), ("ILE", "CG1", "CG2")):
        slots = RESIDUE_ATOM_TO_SLOT[residue]
        valid = res_mask & (res_type == AA_TO_ID[residue])
        for atom in ("CA", "CB", atom1, atom2):
            valid &= atom_mask[..., slots[atom]]
        volume = _signed_unit_volume(
            coords[..., slots["CB"], :],
            coords[..., slots["CA"], :],
            coords[..., slots[atom1], :],
            coords[..., slots[atom2], :],
        )
        total = total + (F.relu(margin - volume).square() * valid).sum()
        count = count + valid.sum()
    return total / count.clamp(min=1).to(total.dtype)


_PLANAR_GROUPS = {
    "ARG": ("CZ", "NE", "NH1", "NH2"),
    "ASN": ("CG", "CB", "OD1", "ND2"),
    "ASP": ("CG", "CB", "OD1", "OD2"),
    "GLN": ("CD", "CG", "OE1", "NE2"),
    "GLU": ("CD", "CG", "OE1", "OE2"),
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TRP": ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
}


def sidechain_planarity_loss(
    pred_coords: Tensor,
    res_type: Tensor,
    atom_mask: Tensor,
    res_mask: Tensor,
) -> Tensor:
    """Planarity barriers for aromatic, carboxylate, amide and guanidinium groups."""
    coords = pred_coords.float()
    total = coords.sum() * 0.0
    count = torch.zeros((), dtype=torch.long, device=coords.device)
    for residue, atoms in _PLANAR_GROUPS.items():
        slots = RESIDUE_ATOM_TO_SLOT[residue]
        indices = [slots[atom] for atom in atoms]
        valid = res_mask & (res_type == AA_TO_ID[residue])
        for index in indices:
            valid &= atom_mask[..., index]
        origin = coords[..., indices[0], :]
        normal = F.normalize(
            torch.linalg.cross(
                coords[..., indices[1], :] - origin,
                coords[..., indices[2], :] - origin,
                dim=-1,
            ),
            dim=-1,
        )
        for index in indices[3:]:
            direction = F.normalize(coords[..., index, :] - origin, dim=-1)
            total = total + ((normal * direction).sum(-1).square() * valid).sum()
            count = count + valid.sum()
    return total / count.clamp(min=1).to(total.dtype)


def _dihedral(a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
    b0, b1, b2 = b - a, c - b, d - c
    n0 = F.normalize(torch.linalg.cross(b0, b1, dim=-1), dim=-1)
    n1 = F.normalize(torch.linalg.cross(b1, b2, dim=-1), dim=-1)
    axis = F.normalize(b1, dim=-1)
    x = (n0 * n1).sum(-1)
    y = (torch.linalg.cross(n0, n1, dim=-1) * axis).sum(-1)
    return torch.atan2(y, x)


_RAMA_CENTERS_DEG = (
    ((-65, -40), (-120, 130), (-75, 145), (60, 40)),
    ((-65, -40), (-120, 130), (-75, 145), (60, 40), (80, 0), (80, 160), (-80, 0)),
    ((-65, -35), (-75, 145), (-120, 130)),
    ((-65, -40), (-75, 145), (-120, 130)),
)


def ramachandran_outlier_loss(
    pred_coords: Tensor,
    res_type: Tensor,
    atom_mask: Tensor,
    res_mask: Tensor,
    chain_id: Tensor | None = None,
    res_seq_nums: Tensor | None = None,
    allowed_radius_deg: float = 100.0,
    softness_deg: float = 20.0,
) -> Tensor:
    """Very broad residue-class Ramachandran barrier, not a favored-region prior."""
    if pred_coords.shape[1] < 3:
        return pred_coords.new_zeros(())
    coords = pred_coords.float()
    left = _adjacent_mask(res_mask, chain_id, res_seq_nums)[:, :-1]
    right = _adjacent_mask(res_mask, chain_id, res_seq_nums)[:, 1:]
    valid = left & right
    valid &= (
        atom_mask[:, :-2, _C]
        & atom_mask[:, 1:-1, _N]
        & atom_mask[:, 1:-1, _CA]
        & atom_mask[:, 1:-1, _C]
        & atom_mask[:, 2:, _N]
    )
    phi = _dihedral(
        coords[:, :-2, _C],
        coords[:, 1:-1, _N],
        coords[:, 1:-1, _CA],
        coords[:, 1:-1, _C],
    )
    psi = _dihedral(
        coords[:, 1:-1, _N],
        coords[:, 1:-1, _CA],
        coords[:, 1:-1, _C],
        coords[:, 2:, _N],
    )
    current = res_type[:, 1:-1]
    following = res_type[:, 2:]
    classes = torch.zeros_like(current)
    classes = torch.where(current == AA_TO_ID["GLY"], 1, classes)
    classes = torch.where(current == AA_TO_ID["PRO"], 2, classes)
    classes = torch.where(
        (following == AA_TO_ID["PRO"])
        & (current != AA_TO_ID["GLY"])
        & (current != AA_TO_ID["PRO"]),
        3,
        classes,
    )

    max_centers = max(len(group) for group in _RAMA_CENTERS_DEG)
    center_values = coords.new_zeros(4, max_centers, 2)
    center_mask = torch.zeros(4, max_centers, dtype=torch.bool, device=coords.device)
    for class_id, group in enumerate(_RAMA_CENTERS_DEG):
        center_values[class_id, : len(group)] = torch.tensor(
            group, dtype=coords.dtype, device=coords.device
        ).mul_(math.pi / 180.0)
        center_mask[class_id, : len(group)] = True
    centers = center_values[classes]
    available = center_mask[classes]
    observed = torch.stack((phi, psi), dim=-1).unsqueeze(-2)
    delta = observed - centers
    delta = torch.atan2(torch.sin(delta), torch.cos(delta))
    distance = torch.linalg.vector_norm(delta, dim=-1)
    distance = distance.masked_fill(~available, float("inf")).min(dim=-1).values
    excess = F.relu(
        (distance - math.radians(allowed_radius_deg)) / math.radians(softness_deg)
    ).square()
    return _masked_mean(excess, valid)


def _safe_vdw_pair_distance(delta: Tensor) -> Tensor:
    """Distance with a deterministic descent direction at exact coincidence."""
    near_zero = delta.detach().square().sum(dim=-1) < 1e-12
    fallback = delta.new_tensor((1e-3, 0.0, 0.0))
    safe_delta = delta + near_zero.unsqueeze(-1) * fallback
    return torch.linalg.vector_norm(safe_delta, dim=-1)


def _clash_surrogate_terms(
    pair_dist: Tensor,
    pair_floor: Tensor,
    pair_valid: Tensor,
    *,
    mode_id: int,
    margin_A: float,
    huber_delta_A: float,
    softplus_tau_A: float,
    softplus_halo: float,
) -> Tensor:
    """Return summed optimization, hard-count, and soft-count terms."""
    metric_penetration = pair_floor - pair_dist
    halo_A = max(margin_A, softplus_tau_A * softplus_halo)
    active = pair_valid & (metric_penetration > -halo_A)

    if mode_id == 0:
        # Smooth-L1 in Angstroms. At the OpenStructure boundary the configured
        # positive margin keeps a non-zero slope; deep overlaps have unit slope.
        violation = F.relu(metric_penetration + margin_A)
        penalty = torch.where(
            violation <= huber_delta_A,
            0.5 * violation.square() / huber_delta_A,
            violation - 0.5 * huber_delta_A,
        )
    else:
        # Smooth linear hinge with an exactly-zero value at the finite halo.
        # The force is sigmoid-shaped and saturates instead of exploding for
        # deeply coincident atoms.
        z = (metric_penetration + margin_A) / softplus_tau_A
        baseline = softplus_tau_A * F.softplus(pair_dist.new_tensor(-softplus_halo))
        penalty = F.relu(softplus_tau_A * F.softplus(z) - baseline)

    active_f = active.to(pair_dist.dtype)
    hard_count = ((metric_penetration > 0.0) & pair_valid).sum()
    soft_count = (torch.sigmoid(metric_penetration / softplus_tau_A) * active_f).sum().detach()
    return torch.stack(
        (
            (penalty * active_f).sum(),
            hard_count.to(pair_dist.dtype),
            soft_count,
        )
    )


def _inter_residue_clash_surrogate_chunk(
    xyz: Tensor,
    types: Tensor,
    elements: Tensor,
    valid: Tensor,
    floor_table: Tensor,
    ri: Tensor,
    rj: Tensor,
    sequential: Tensor,
    peptide_bond_exclusion: Tensor,
    disulfide_slots: Tensor,
    mode_id: int,
    margin_A: float,
    huber_delta_A: float,
    softplus_tau_A: float,
    softplus_halo: float,
) -> Tensor:
    xi, xj = xyz[ri], xyz[rj]
    pair_dist = _safe_vdw_pair_distance(xi.unsqueeze(2) - xj.unsqueeze(1))
    pair_valid = valid[ri].unsqueeze(2) & valid[rj].unsqueeze(1)
    pair_valid &= ~(sequential[:, None, None] & peptide_bond_exclusion[None])
    pair_floor = floor_table[elements[ri].unsqueeze(2), elements[rj].unsqueeze(1)]
    cys_pair = (types[ri] == AA_TO_ID["CYS"]) & (types[rj] == AA_TO_ID["CYS"])
    pair_floor = torch.where(
        cys_pair[:, None, None] & disulfide_slots[None],
        pair_floor.new_tensor(_DISULFIDE_FLOOR_A),
        pair_floor,
    )
    return _clash_surrogate_terms(
        pair_dist,
        pair_floor,
        pair_valid,
        mode_id=mode_id,
        margin_A=margin_A,
        huber_delta_A=huber_delta_A,
        softplus_tau_A=softplus_tau_A,
        softplus_halo=softplus_halo,
    )


def all_atom_clash_surrogate_loss(
    pred_coords: Tensor,
    res_type: Tensor,
    atom_mask: Tensor,
    res_mask: Tensor,
    chain_id: Tensor | None = None,
    res_seq_nums: Tensor | None = None,
    *,
    mode: str = "huber",
    margin_A: float = 0.1,
    huber_delta_A: float = 0.25,
    softplus_tau_A: float = 0.05,
    softplus_halo: float = 6.0,
    pair_chunk_size: int | None = 1024,
    reduction: str = "mean",
) -> dict[str, Tensor]:
    """OpenStructure-aligned heavy-atom clash training surrogate.

    The reference element-pair floors follow OpenStructure's protein defaults,
    including the special 1.03 Angstrom floor for a potential CYS SG-SG
    disulfide. Only direct covalent bonds are excluded: intra-residue 1-2 bonds and the
    consecutive peptide C(i)-N(i+1) bond. In particular, 1-3 pairs remain in
    the metric just as they do in OpenStructure ``GetClashes``.

    Each protein is normalized by its emitted heavy-atom count before the
    batch reduction, matching equal-target clash-rate evaluation. The returned
    hard and soft rates are per atom; multiply by 1000 for clashes/1k atoms.
    """
    if mode not in {"huber", "softplus"}:
        raise ValueError("mode must be 'huber' or 'softplus'")
    if margin_A < 0.0:
        raise ValueError("margin_A must be non-negative")
    if huber_delta_A <= 0.0:
        raise ValueError("huber_delta_A must be positive")
    if softplus_tau_A <= 0.0:
        raise ValueError("softplus_tau_A must be positive")
    if softplus_halo <= 0.0:
        raise ValueError("softplus_halo must be positive")
    if pair_chunk_size is not None and pair_chunk_size <= 0:
        raise ValueError("pair_chunk_size must be positive or None")
    if reduction not in {"mean", "none"}:
        raise ValueError("reduction must be 'mean' or 'none'")

    coords = pred_coords.float() * COORD_SCALE
    topology = _topology(coords.device)
    res_type = res_type.clamp(0, topology.atom_element_id.shape[0] - 1)
    mode_id = 0 if mode == "huber" else 1
    candidate_extra_A = max(margin_A, softplus_tau_A * softplus_halo)
    q_max_A = _MAX_OST_CLASH_FLOOR_A
    atom_upper = torch.triu(
        torch.ones(
            MAX_ATOMS_PER_RES,
            MAX_ATOMS_PER_RES,
            dtype=torch.bool,
            device=coords.device,
        ),
        diagonal=1,
    )
    peptide_bond_exclusion = torch.zeros_like(atom_upper)
    peptide_bond_exclusion[_C, _N] = True
    disulfide_slots = torch.zeros_like(atom_upper)
    disulfide_slots[_CYS_SG, _CYS_SG] = True
    per_example: list[Tensor] = []

    for batch_index in range(coords.shape[0]):
        xyz = coords[batch_index]
        types = res_type[batch_index]
        valid = atom_mask[batch_index] & res_mask[batch_index].unsqueeze(-1)
        elements = topology.atom_element_id[types]
        totals = xyz.sum().reshape(1).repeat(3) * 0.0

        intra_dist = _safe_vdw_pair_distance(xyz.unsqueeze(2) - xyz.unsqueeze(1))
        intra_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
        intra_valid &= atom_upper.unsqueeze(0)
        intra_valid &= ~topology.bond_exclusion[types]
        intra_floor = topology.nonbonded_floor_A[elements.unsqueeze(2), elements.unsqueeze(1)]
        totals = totals + _clash_surrogate_terms(
            intra_dist,
            intra_floor,
            intra_valid,
            mode_id=mode_id,
            margin_A=margin_A,
            huber_delta_A=huber_delta_A,
            softplus_tau_A=softplus_tau_A,
            softplus_halo=softplus_halo,
        )

        ca = xyz[:, _CA]
        ca_valid = valid[:, _CA]
        ca_dist = torch.cdist(ca.detach(), ca.detach())
        atom_extent = (
            torch.linalg.vector_norm(xyz.detach() - ca.detach().unsqueeze(1), dim=-1)
            .masked_fill(~valid, float("-inf"))
            .amax(dim=-1)
        )
        candidate_cutoff = (
            atom_extent.unsqueeze(1) + atom_extent.unsqueeze(0) + q_max_A + candidate_extra_A
        )
        residue_upper = torch.triu(
            torch.ones(ca.shape[0], ca.shape[0], dtype=torch.bool, device=coords.device),
            diagonal=1,
        )
        candidates = (
            residue_upper
            & ca_valid.unsqueeze(1)
            & ca_valid.unsqueeze(0)
            & (ca_dist < candidate_cutoff)
        ).nonzero(as_tuple=False)

        if candidates.numel() > 0:
            chunk_size = candidates.shape[0] if pair_chunk_size is None else pair_chunk_size
            for start in range(0, candidates.shape[0], chunk_size):
                pair_indices = candidates[start : start + chunk_size]
                ri, rj = pair_indices[:, 0], pair_indices[:, 1]
                if chain_id is not None:
                    sequential = chain_id[batch_index, ri] == chain_id[batch_index, rj]
                else:
                    sequential = torch.ones_like(ri, dtype=torch.bool)
                if res_seq_nums is not None:
                    sequential &= (
                        res_seq_nums[batch_index, rj] - res_seq_nums[batch_index, ri]
                    ) == 1
                else:
                    sequential &= (rj - ri) == 1

                args = (
                    xyz,
                    types,
                    elements,
                    valid,
                    topology.nonbonded_floor_A,
                    ri,
                    rj,
                    sequential,
                    peptide_bond_exclusion,
                    disulfide_slots,
                    mode_id,
                    margin_A,
                    huber_delta_A,
                    softplus_tau_A,
                    softplus_halo,
                )
                if xyz.requires_grad:
                    chunk_terms = checkpoint(
                        _inter_residue_clash_surrogate_chunk,
                        *args,
                        use_reentrant=False,
                    )
                else:
                    chunk_terms = _inter_residue_clash_surrogate_chunk(*args)
                totals = totals + chunk_terms

        atom_count = valid.sum().clamp(min=1).to(totals.dtype)
        per_example.append(totals / atom_count)

    if per_example:
        values = torch.stack(per_example)
    else:
        values = coords.new_zeros((0, 3))
    if reduction == "mean":
        values = values.mean(dim=0) if values.numel() else coords.new_zeros(3)
    return {
        "loss": values[..., 0],
        "hard_clashes_per_atom": values[..., 1],
        "soft_clashes_per_atom": values[..., 2],
    }


def _inter_residue_vdw_clash_chunk(
    xyz: Tensor,
    radii: Tensor,
    valid: Tensor,
    ri: Tensor,
    rj: Tensor,
    sequential: Tensor,
    adjacent_exclusion: Tensor,
    overlap_tolerance_A: float,
) -> Tensor:
    xi, xj = xyz[ri], xyz[rj]
    pair_dist = _safe_vdw_pair_distance(xi.unsqueeze(2) - xj.unsqueeze(1))
    pair_valid = valid[ri].unsqueeze(2) & valid[rj].unsqueeze(1)
    pair_valid &= ~(sequential[:, None, None] & adjacent_exclusion[None])
    pair_floor = radii[ri].unsqueeze(2) + radii[rj].unsqueeze(1) - overlap_tolerance_A
    return (F.relu(pair_floor - pair_dist).square() * pair_valid).sum()


def all_atom_vdw_clash_loss(
    pred_coords: Tensor,
    res_type: Tensor,
    atom_mask: Tensor,
    res_mask: Tensor,
    chain_id: Tensor | None = None,
    res_seq_nums: Tensor | None = None,
    overlap_tolerance_A: float = 0.6,
    residue_cutoff_A: float | None = None,
    pair_chunk_size: int | None = 2048,
) -> Tensor:
    """Topology-aware heavy-atom repulsion with bounded candidate-pair memory.

    Candidate residue pairs use a detached, conformation-dependent bounding
    sphere around each C-alpha, so the default cannot discard a possible atom
    overlap even for extended side chains. ``residue_cutoff_A`` optionally adds
    a user-requested hard cap. ``pair_chunk_size=None`` retains the dense
    candidate-pair path for numerical comparisons. Finite chunk sizes preserve
    the same pairwise energy while bounding the largest inter-residue tensor to
    ``pair_chunk_size * MAX_ATOMS_PER_RES**2`` entries.  Gradient checkpointing
    prevents all chunk-local distance tensors from being retained at once.
    """
    if pair_chunk_size is not None and pair_chunk_size <= 0:
        raise ValueError("pair_chunk_size must be positive or None")
    if residue_cutoff_A is not None and residue_cutoff_A <= 0.0:
        raise ValueError("residue_cutoff_A must be positive or None")

    coords = pred_coords.float() * COORD_SCALE
    topology = _topology(coords.device)
    res_type = res_type.clamp(0, topology.atom_radius_A.shape[0] - 1)
    total = coords.sum() * 0.0
    valid_atoms_total = (atom_mask & res_mask.unsqueeze(-1)).sum().clamp(min=1)
    atom_upper = torch.triu(
        torch.ones(MAX_ATOMS_PER_RES, MAX_ATOMS_PER_RES, dtype=torch.bool, device=coords.device),
        diagonal=1,
    )
    adjacent_exclusion = torch.zeros_like(atom_upper)
    for first, second in ((_C, _N), (_CA, _N), (_O, _N), (_C, _CA)):
        adjacent_exclusion[first, second] = True

    for batch_index in range(coords.shape[0]):
        xyz = coords[batch_index]
        types = res_type[batch_index]
        valid = atom_mask[batch_index] & res_mask[batch_index].unsqueeze(-1)
        radii = topology.atom_radius_A[types]

        # Non-bonded atom pairs within one residue.
        intra_dist = _safe_vdw_pair_distance(xyz.unsqueeze(2) - xyz.unsqueeze(1))
        intra_pair = valid.unsqueeze(2) & valid.unsqueeze(1) & atom_upper.unsqueeze(0)
        intra_pair &= ~topology.same_res_exclusion[types]
        intra_floor = radii.unsqueeze(2) + radii.unsqueeze(1) - overlap_tolerance_A
        total = total + (F.relu(intra_floor - intra_dist).square() * intra_pair).sum()

        # A residue is contained in a sphere centered at C-alpha whose radius
        # is max(atom distance to CA + atom VDW radius). If two such spheres
        # cannot overlap beyond the tolerance, every atom-pair term is zero.
        ca = xyz[:, _CA]
        ca_valid = valid[:, _CA]
        ca_dist = torch.cdist(ca.detach(), ca.detach())
        atom_extent = (
            (
                torch.linalg.vector_norm(xyz.detach() - ca.detach().unsqueeze(1), dim=-1)
                + radii.detach()
            )
            .masked_fill(~valid, float("-inf"))
            .amax(dim=-1)
        )
        candidate_cutoff = atom_extent.unsqueeze(1) + atom_extent.unsqueeze(0) - overlap_tolerance_A
        residue_upper = torch.triu(
            torch.ones(ca.shape[0], ca.shape[0], dtype=torch.bool, device=coords.device),
            diagonal=1,
        )
        candidate_pair = (
            residue_upper
            & ca_valid.unsqueeze(1)
            & ca_valid.unsqueeze(0)
            & (ca_dist < candidate_cutoff)
        )
        if residue_cutoff_A is not None:
            candidate_pair &= ca_dist < residue_cutoff_A
        candidates = candidate_pair.nonzero(as_tuple=False)
        if candidates.numel() == 0:
            continue
        chunk_size = candidates.shape[0] if pair_chunk_size is None else pair_chunk_size
        for start in range(0, candidates.shape[0], chunk_size):
            pair_indices = candidates[start : start + chunk_size]
            ri, rj = pair_indices[:, 0], pair_indices[:, 1]
            if chain_id is not None:
                sequential = chain_id[batch_index, ri] == chain_id[batch_index, rj]
            else:
                sequential = torch.ones_like(ri, dtype=torch.bool)
            if res_seq_nums is not None:
                sequential &= (res_seq_nums[batch_index, rj] - res_seq_nums[batch_index, ri]) == 1
            else:
                sequential &= (rj - ri) == 1

            if xyz.requires_grad:
                chunk_energy = checkpoint(
                    _inter_residue_vdw_clash_chunk,
                    xyz,
                    radii,
                    valid,
                    ri,
                    rj,
                    sequential,
                    adjacent_exclusion,
                    overlap_tolerance_A,
                    use_reentrant=False,
                )
            else:
                chunk_energy = _inter_residue_vdw_clash_chunk(
                    xyz,
                    radii,
                    valid,
                    ri,
                    rj,
                    sequential,
                    adjacent_exclusion,
                    overlap_tolerance_A,
                )
            total = total + chunk_energy

    return total / valid_atoms_total.to(total.dtype)


def stereochemical_energy_terms(
    pred_coords: Tensor,
    res_type: Tensor,
    atom_mask: Tensor,
    res_mask: Tensor,
    chain_id: Tensor | None = None,
    res_seq_nums: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return the complete conservative validity energy decomposition."""
    return {
        "covalent": covalent_geometry_loss(
            pred_coords, res_type, atom_mask, res_mask, chain_id, res_seq_nums
        ),
        "peptide_planarity": peptide_planarity_loss(
            pred_coords, atom_mask, res_mask, chain_id, res_seq_nums
        ),
        "chirality": chemical_chirality_loss(pred_coords, res_type, atom_mask, res_mask),
        "sidechain_planarity": sidechain_planarity_loss(pred_coords, res_type, atom_mask, res_mask),
        "ramachandran": ramachandran_outlier_loss(
            pred_coords, res_type, atom_mask, res_mask, chain_id, res_seq_nums
        ),
        "all_atom_clash": all_atom_vdw_clash_loss(
            pred_coords, res_type, atom_mask, res_mask, chain_id, res_seq_nums
        ),
    }
