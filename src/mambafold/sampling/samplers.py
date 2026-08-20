"""Direct all-atom ODE/SDE samplers."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from mambafold.data.constants import CA_ATOM_ID, COORD_SCALE
from mambafold.losses.ca_only import ca_virtual_angle_floor
from mambafold.losses.geometry import bond_length_loss, ca_clash_loss
from mambafold.losses.stereochemistry import (
    all_atom_vdw_clash_loss,
    chemical_chirality_loss,
    covalent_geometry_loss,
    peptide_planarity_loss,
    ramachandran_outlier_loss,
    sidechain_planarity_loss,
)
from mambafold.utils.geometry import masked_centroid

BatchFn = Callable[[Tensor, float], "object"]
_T_END = 0.99
_N_ATOM_ID = 0
_C_ATOM_ID = 2


@dataclass(frozen=True)
class GeometryGuidanceConfig:
    """Inference-only, ground-truth-free geometric guidance.

    The energy gradient is evaluated on the model's current clean-coordinate
    estimate and added as an external drift. ``scale=0``, ``vdw_scale=0``, and
    ``steric_scale=0`` together are an exact no-op.  The legacy fields retain
    the original joint-energy behavior; the VDW and steric fields add
    independently normalized guidance channels.
    """

    scale: float = 0.0
    start: float = 0.5
    every_n_steps: int = 1
    bond_weight: float = 1.0
    angle_weight: float = 0.25
    clash_weight: float = 0.1
    covalent_weight: float = 0.0
    peptide_planarity_weight: float = 0.0
    chirality_weight: float = 0.0
    sidechain_planarity_weight: float = 0.0
    ramachandran_weight: float = 0.0
    all_atom_clash_weight: float = 0.0
    all_atom_clash_every_n_steps: int = 4
    vdw_scale: float = 0.0
    vdw_start: float = 0.65
    vdw_every_n_steps: int = 8
    vdw_overlap_tolerance_A: float = 1.5
    vdw_max_step_A: float = 0.01
    steric_scale: float = 0.0
    steric_start: float = 0.35
    steric_ramp_end: float = 0.55
    steric_taper_start: float = 0.90
    steric_taper_final: float = 0.25
    steric_every_n_steps: int = 1
    steric_ca_min_dist_A: float = 3.6
    steric_ca_seq_sep: int = 12
    steric_segment_weight: float = 0.0
    steric_segment_min_dist_A: float = 2.5
    steric_segment_max_edge_A: float = 6.0
    steric_segment_every_n_steps: int = 4
    steric_segment_pair_chunk_size: int = 4096
    steric_smoothing_radius: int = 4
    steric_smoothing_sigma: float = 2.0
    steric_bond_projection_iters: int = 8
    steric_severity_reference_A: float = 0.5
    steric_max_step_A: float = 0.02

    @classmethod
    def stereochemical(
        cls,
        *,
        scale: float,
        start: float = 0.6,
        every_n_steps: int = 2,
    ) -> "GeometryGuidanceConfig":
        """Conservative full-validity preset; still an exact no-op at scale 0."""
        return cls(
            scale=scale,
            start=start,
            every_n_steps=every_n_steps,
            bond_weight=0.5,
            angle_weight=0.1,
            clash_weight=0.0,
            covalent_weight=1.0,
            peptide_planarity_weight=0.3,
            chirality_weight=0.5,
            sidechain_planarity_weight=0.1,
            ramachandran_weight=0.05,
            all_atom_clash_weight=0.2,
            all_atom_clash_every_n_steps=4,
        )

    @classmethod
    def self_avoidance(
        cls,
        *,
        local_scale: float,
        steric_scale: float,
        local_start: float = 0.65,
        local_every_n_steps: int = 2,
        steric_start: float = 0.35,
        steric_ramp_end: float = 0.55,
        steric_every_n_steps: int = 1,
        steric_smoothing_radius: int = 4,
    ) -> "GeometryGuidanceConfig":
        """Two-channel preset for gross nonlocal residue self-overlap.

        Local stereochemistry remains a late, conservative cleanup.  The
        independent C-alpha channel starts earlier and translates whole
        residues coherently, so steric correction does not tear side chains.
        """
        return cls(
            scale=local_scale,
            start=local_start,
            every_n_steps=local_every_n_steps,
            bond_weight=0.5,
            angle_weight=0.1,
            clash_weight=0.0,
            covalent_weight=1.0,
            peptide_planarity_weight=0.3,
            chirality_weight=0.5,
            sidechain_planarity_weight=0.1,
            ramachandran_weight=0.05,
            # Preserve the confirmed steric-1 baseline by default.  The now
            # chunked all-atom VDW term is enabled explicitly in ablations.
            all_atom_clash_weight=0.0,
            all_atom_clash_every_n_steps=4,
            steric_scale=steric_scale,
            steric_start=steric_start,
            steric_ramp_end=steric_ramp_end,
            steric_every_n_steps=steric_every_n_steps,
            steric_smoothing_radius=steric_smoothing_radius,
        )

    def validate(self) -> None:
        if self.scale < 0.0:
            raise ValueError("geometry guidance scale must be non-negative")
        if self.steric_scale < 0.0:
            raise ValueError("steric guidance scale must be non-negative")
        if not 0.0 <= self.start < 1.0:
            raise ValueError("geometry guidance start must be in [0, 1)")
        if self.every_n_steps < 1:
            raise ValueError("geometry guidance every_n_steps must be positive")
        weights = (
            self.bond_weight,
            self.angle_weight,
            self.clash_weight,
            self.covalent_weight,
            self.peptide_planarity_weight,
            self.chirality_weight,
            self.sidechain_planarity_weight,
            self.ramachandran_weight,
            self.all_atom_clash_weight,
        )
        if min(weights) < 0.0:
            raise ValueError("geometry guidance weights must be non-negative")
        if self.all_atom_clash_every_n_steps < 1:
            raise ValueError("all-atom clash interval must be positive")
        if self.vdw_scale < 0.0:
            raise ValueError("independent VDW guidance scale must be non-negative")
        if not 0.0 <= self.vdw_start < 1.0:
            raise ValueError("independent VDW guidance start must be in [0, 1)")
        if self.vdw_every_n_steps < 1:
            raise ValueError("independent VDW guidance interval must be positive")
        if self.vdw_overlap_tolerance_A <= 0.0:
            raise ValueError("independent VDW overlap tolerance must be positive")
        if self.vdw_max_step_A <= 0.0:
            raise ValueError("independent VDW maximum step must be positive")
        if not 0.0 <= self.steric_start < self.steric_ramp_end <= 1.0:
            raise ValueError("steric start/ramp end must satisfy 0 <= start < end <= 1")
        if not self.steric_ramp_end <= self.steric_taper_start < 1.0:
            raise ValueError("steric taper start must be in [ramp_end, 1)")
        if not 0.0 <= self.steric_taper_final <= 1.0:
            raise ValueError("steric final taper must be in [0, 1]")
        if self.steric_every_n_steps < 1:
            raise ValueError("steric guidance interval must be positive")
        if self.steric_ca_min_dist_A <= 0.0:
            raise ValueError("steric C-alpha distance floor must be positive")
        if self.steric_ca_seq_sep < 2:
            raise ValueError("steric C-alpha sequence separation must be at least 2")
        if self.steric_segment_weight < 0.0:
            raise ValueError("steric segment weight must be non-negative")
        if self.steric_segment_min_dist_A <= 0.0:
            raise ValueError("steric segment distance floor must be positive")
        if self.steric_segment_max_edge_A <= 0.0:
            raise ValueError("steric segment maximum edge length must be positive")
        if self.steric_segment_every_n_steps < 1:
            raise ValueError("steric segment interval must be positive")
        if self.steric_segment_pair_chunk_size < 1:
            raise ValueError("steric segment pair chunk size must be positive")
        if self.steric_smoothing_radius < 0:
            raise ValueError("steric smoothing radius must be non-negative")
        if self.steric_smoothing_sigma <= 0.0:
            raise ValueError("steric smoothing sigma must be positive")
        if self.steric_bond_projection_iters < 0:
            raise ValueError("steric bond projection iterations must be non-negative")
        if self.steric_severity_reference_A <= 0.0:
            raise ValueError("steric severity reference must be positive")
        if self.steric_max_step_A <= 0.0:
            raise ValueError("steric maximum step must be positive")


def _normalize_guidance_gradient(
    grad: Tensor,
    atom_mask: Tensor,
) -> Tensor:
    """Remove translation, unit-RMS normalize, and cap atom outliers."""
    mask = atom_mask.squeeze(0).unsqueeze(-1)
    grad = torch.nan_to_num(grad) * mask
    valid = mask.squeeze(-1).reshape(-1)
    grad = (grad - masked_centroid(grad.reshape(-1, 3), valid).view(1, 1, 3)) * mask
    denom = (mask.sum() * 3).clamp(min=1).to(grad.dtype)
    rms = torch.sqrt(grad.square().sum() / denom)
    grad = torch.nan_to_num(grad / rms.clamp(min=1e-8))
    atom_norm = torch.linalg.vector_norm(grad, dim=-1, keepdim=True)
    return grad * (3.0 / atom_norm.clamp(min=3.0))


def _smooth_residue_vectors(
    vectors: Tensor,
    res_mask: Tensor,
    chain_id: Tensor | None,
    res_seq_nums: Tensor | None,
    *,
    radius: int,
    sigma: float,
) -> Tensor:
    """Chain- and sequence-gap-aware Gaussian smoothing along residues."""
    vectors = vectors * res_mask.unsqueeze(-1)
    if radius == 0 or vectors.shape[1] < 2:
        return vectors

    length = vectors.shape[1]
    total = torch.zeros_like(vectors)
    weight_total = torch.zeros_like(res_mask, dtype=vectors.dtype)
    for offset in range(-radius, radius + 1):
        weight = math.exp(-0.5 * (offset / sigma) ** 2)
        if offset < 0:
            dst = slice(-offset, length)
            src = slice(0, length + offset)
        else:
            dst = slice(0, length - offset)
            src = slice(offset, length)
        valid = res_mask[:, dst] & res_mask[:, src]
        if chain_id is not None:
            valid &= chain_id[:, dst] == chain_id[:, src]
        if res_seq_nums is not None:
            valid &= (res_seq_nums[:, src] - res_seq_nums[:, dst]) == offset
        valid_f = valid.to(vectors.dtype)
        total[:, dst] += vectors[:, src] * valid_f.unsqueeze(-1) * weight
        weight_total[:, dst] += valid_f * weight
    return total / weight_total.clamp(min=1e-8).unsqueeze(-1)


def _project_adjacent_bond_axis(
    vectors: Tensor,
    coords: Tensor,
    atom_mask: Tensor,
    res_mask: Tensor,
    chain_id: Tensor | None,
    res_seq_nums: Tensor | None,
    *,
    iterations: int,
) -> Tensor:
    """Remove first-order adjacent-bond stretching from residue translations."""
    if iterations == 0 or vectors.shape[1] < 2:
        return vectors
    out = vectors.clone()
    length = vectors.shape[1]
    edge_index = torch.arange(length - 1, device=vectors.device)
    adjacent = res_mask[:, :-1] & res_mask[:, 1:]
    if chain_id is not None:
        adjacent &= chain_id[:, :-1] == chain_id[:, 1:]
    if res_seq_nums is not None:
        adjacent &= (res_seq_nums[:, 1:] - res_seq_nums[:, :-1]) == 1

    peptide_valid = atom_mask[:, :-1, _C_ATOM_ID] & atom_mask[:, 1:, _N_ATOM_ID]
    peptide = coords[:, 1:, _N_ATOM_ID] - coords[:, :-1, _C_ATOM_ID]
    ca_bond = coords[:, 1:, CA_ATOM_ID] - coords[:, :-1, CA_ATOM_ID]
    bond = torch.where(peptide_valid.unsqueeze(-1), peptide, ca_bond)
    unit = bond / torch.linalg.vector_norm(bond, dim=-1, keepdim=True).clamp(min=1e-6)
    for _ in range(iterations):
        for parity in (0, 1):
            select = edge_index[parity::2]
            if select.numel() == 0:
                continue
            valid = adjacent[:, select]
            relative = out[:, select + 1] - out[:, select]
            axial = (relative * unit[:, select]).sum(-1) * valid
            correction = 0.5 * axial.unsqueeze(-1) * unit[:, select]
            out[:, select] += correction
            out[:, select + 1] -= correction
    return out * res_mask.unsqueeze(-1)


def _normalize_residue_vectors(vectors: Tensor, res_mask: Tensor) -> Tensor:
    """Normalize over the nonzero residue support, independent of chain length."""
    vectors = torch.nan_to_num(vectors) * res_mask.unsqueeze(-1)
    support = res_mask & (torch.linalg.vector_norm(vectors, dim=-1) > 1e-8)
    centroid = masked_centroid(vectors, support)
    vectors = (vectors - centroid) * support.unsqueeze(-1)
    denom = (support.sum(dim=1, keepdim=True) * 3).clamp(min=1).to(vectors.dtype)
    rms = torch.sqrt(vectors.square().sum(dim=(1, 2), keepdim=True) / denom.unsqueeze(-1))
    return torch.nan_to_num(vectors / rms.clamp(min=1e-8))


def _global_vector_norm_cap(vectors: Tensor, max_norm: float = 3.0) -> Tensor:
    """Apply one scalar cap per batch so linear bond constraints stay intact."""
    largest = torch.linalg.vector_norm(vectors, dim=-1).amax(dim=1, keepdim=True)
    factor = max_norm / largest.clamp(min=max_norm)
    return vectors * factor.unsqueeze(-1)


def _nonlocal_ca_steric_energy(
    coords: Tensor,
    res_mask: Tensor,
    chain_id: Tensor | None,
    res_seq_nums: Tensor | None,
    *,
    min_dist_A: float,
    seq_sep: int,
) -> tuple[Tensor, Tensor]:
    """Upper-triangle nonlocal C-alpha barrier and RMS penetration in Angstrom."""
    ca = coords[..., CA_ATOM_ID, :]
    batch_size, length, _ = ca.shape
    delta = ca.unsqueeze(2) - ca.unsqueeze(1)
    near_zero = delta.detach().square().sum(dim=-1) < (1e-3 / COORD_SCALE) ** 2

    # torch.linalg.norm has zero gradient at exact coincidence.  Add a tiny,
    # deterministic pair direction only for those pairs so the barrier can
    # separate perfectly superposed residues without introducing randomness.
    position = torch.arange(length, device=coords.device, dtype=coords.dtype)
    jitter = torch.stack(
        (
            torch.sin(position * 1.61803398875),
            torch.cos(position * 2.41421356237),
            torch.sin(position * 3.14159265359 + 0.5),
        ),
        dim=-1,
    )
    jitter = jitter * (1e-3 / COORD_SCALE)
    fallback = jitter.unsqueeze(1) - jitter.unsqueeze(0)
    delta = delta + near_zero.unsqueeze(-1) * fallback.unsqueeze(0)
    distance = torch.linalg.vector_norm(delta, dim=-1)

    upper = torch.triu(
        torch.ones(length, length, dtype=torch.bool, device=coords.device), diagonal=1
    )
    valid = res_mask.unsqueeze(2) & res_mask.unsqueeze(1) & upper.unsqueeze(0)
    if chain_id is not None:
        same_chain = chain_id.unsqueeze(2) == chain_id.unsqueeze(1)
    else:
        same_chain = torch.ones(batch_size, length, length, dtype=torch.bool, device=coords.device)
    if res_seq_nums is not None:
        separation = (res_seq_nums.unsqueeze(2) - res_seq_nums.unsqueeze(1)).abs()
    else:
        row = torch.arange(length, device=coords.device).view(length, 1)
        col = torch.arange(length, device=coords.device).view(1, length)
        separation = (row - col).abs().unsqueeze(0).expand(batch_size, -1, -1)
    valid &= (~same_chain) | (separation > seq_sep)

    penetration = torch.relu(min_dist_A / COORD_SCALE - distance)
    violating = valid & (penetration.detach() > 0)
    count = violating.sum().clamp(min=1).to(coords.dtype)
    squared = penetration.square() * valid
    energy = squared.sum() / count
    rms_penetration_A = torch.sqrt(squared.detach().sum() / count) * COORD_SCALE
    return energy, rms_penetration_A


def _closest_segment_parameters(
    p0: Tensor,
    p1: Tensor,
    q0: Tensor,
    q1: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return detached closest-point parameters and residuals for segment pairs.

    Four endpoint-to-segment candidates plus a valid interior line-line
    solution cover the exact constrained minimum.  The selected parameters are
    treated as constants by the analytic envelope gradient below.
    """
    u = p1 - p0
    v = q1 - q0
    w = p0 - q0
    a = u.square().sum(-1)
    b = (u * v).sum(-1)
    c = v.square().sum(-1)
    d = (u * w).sum(-1)
    e = (v * w).sum(-1)

    safe_a = a.clamp(min=1e-12)
    safe_c = c.clamp(min=1e-12)
    s_q0 = (-d / safe_a).clamp(0.0, 1.0)
    s_q1 = ((b - d) / safe_a).clamp(0.0, 1.0)
    t_p0 = (e / safe_c).clamp(0.0, 1.0)
    t_p1 = ((e + b) / safe_c).clamp(0.0, 1.0)

    denominator = a * c - b.square()
    interior_valid = denominator > 1e-12
    safe_denominator = denominator.clamp(min=1e-12)
    s_interior = (b * e - c * d) / safe_denominator
    t_interior = (a * e - b * d) / safe_denominator
    interior_valid &= (
        (s_interior >= 0.0) & (s_interior <= 1.0) & (t_interior >= 0.0) & (t_interior <= 1.0)
    )

    zeros = torch.zeros_like(a)
    ones = torch.ones_like(a)
    s_candidates = torch.stack((zeros, ones, s_q0, s_q1, s_interior), dim=-1)
    t_candidates = torch.stack((t_p0, t_p1, zeros, ones, t_interior), dim=-1)
    residual = (
        p0[:, None]
        + s_candidates[..., None] * u[:, None]
        - q0[:, None]
        - t_candidates[..., None] * v[:, None]
    )
    distance_sq = residual.square().sum(-1)
    distance_sq[:, -1] = distance_sq[:, -1].masked_fill(~interior_valid, float("inf"))
    selected = distance_sq.argmin(dim=-1)
    gather = selected[:, None]
    s = s_candidates.gather(1, gather).squeeze(1)
    t = t_candidates.gather(1, gather).squeeze(1)
    q = residual.gather(1, gather[..., None].expand(-1, 1, 3)).squeeze(1)
    s = torch.where(a <= 1e-12, torch.full_like(s, 0.5), s)
    t = torch.where(c <= 1e-12, torch.full_like(t, 0.5), t)
    return s, t, q


def _segment_zero_distance_direction(
    p0: Tensor,
    p1: Tensor,
    q0: Tensor,
    q1: Tensor,
    first_index: Tensor,
    second_index: Tensor,
) -> Tensor:
    """Deterministic separating direction for exact/near-exact crossings."""
    u = p1 - p0
    v = q1 - q0
    normal = torch.linalg.cross(u, v)
    normal_norm = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)

    key = (first_index + 1).to(p0.dtype) * 1.61803398875
    key = key + (second_index + 1).to(p0.dtype) * 2.41421356237
    hashed = torch.stack(
        (torch.sin(key), torch.cos(key), torch.sin(key * 1.41421356237 + 0.5)),
        dim=-1,
    )
    tangent = torch.where(
        (torch.linalg.vector_norm(u, dim=-1, keepdim=True) > 1e-8),
        u,
        v,
    )
    tangent = tangent / torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp(min=1e-8)
    perpendicular = hashed - (hashed * tangent).sum(-1, keepdim=True) * tangent
    perpendicular_norm = torch.linalg.vector_norm(perpendicular, dim=-1, keepdim=True)
    perpendicular = torch.where(perpendicular_norm > 1e-8, perpendicular, hashed)
    fallback = perpendicular / torch.linalg.vector_norm(perpendicular, dim=-1, keepdim=True).clamp(
        min=1e-8
    )
    return torch.where(normal_norm > 1e-8, normal / normal_norm.clamp(min=1e-8), fallback)


@torch.no_grad()
def _nonlocal_ca_segment_guidance(
    coords: Tensor,
    ca_mask: Tensor,
    chain_id: Tensor | None,
    res_seq_nums: Tensor | None,
    *,
    min_dist_A: float,
    max_edge_A: float,
    seq_sep: int,
    pair_chunk_size: int,
    spatial_prefilter: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return bounded-memory analytic segment-barrier gradient, energy, and RMS.

    Segment pairs are processed independently and scattered directly into a
    residue gradient, so no O(L^2) autograd graph is retained.  Coordinates and
    gradients use the model's normalized units; reported RMS is in Angstrom.
    """
    if pair_chunk_size < 1:
        raise ValueError("pair_chunk_size must be positive")
    ca = coords[..., CA_ATOM_ID, :]
    batch_size, length, _ = ca.shape
    gradient = torch.zeros_like(ca)
    squared_sum = coords.new_zeros(())
    violating_count = torch.zeros((), dtype=torch.long, device=coords.device)
    if length < 2:
        return gradient, squared_sum, squared_sum

    if chain_id is None:
        chain_id = torch.zeros(batch_size, length, dtype=torch.long, device=coords.device)
    if res_seq_nums is None:
        res_seq_nums = torch.arange(length, device=coords.device).expand(batch_size, -1)
    segment_valid = ca_mask[:, :-1] & ca_mask[:, 1:]
    segment_valid &= chain_id[:, :-1] == chain_id[:, 1:]
    segment_valid &= (res_seq_nums[:, 1:] - res_seq_nums[:, :-1]) == 1
    segment_edge = torch.linalg.vector_norm(ca[:, 1:] - ca[:, :-1], dim=-1)
    segment_valid &= segment_edge <= max_edge_A / COORD_SCALE
    floor = min_dist_A / COORD_SCALE
    zero_jitter = 1e-3 / COORD_SCALE

    # A segment is contained by the sphere centered at its midpoint with the
    # segment half-length as radius.  If two such spheres are farther apart
    # than the barrier floor, their segments cannot violate the barrier.  Use
    # detached float32 bounds (float64 for float64 inputs) and a scale-aware
    # roundoff margin so this can only discard exact zero-contribution pairs.
    bound_dtype = torch.float64 if ca.dtype == torch.float64 else torch.float32
    bound_ca = ca.detach().to(bound_dtype)
    bound_edge = bound_ca[:, 1:] - bound_ca[:, :-1]
    segment_midpoint = bound_ca[:, :-1] + 0.5 * bound_edge
    segment_half_length = 0.5 * torch.linalg.vector_norm(bound_edge, dim=-1)
    bound_eps = torch.finfo(bound_dtype).eps

    for batch_index in range(batch_size):
        n_segments = length - 1
        row, col = torch.triu_indices(
            n_segments,
            n_segments,
            offset=1,
            device=coords.device,
        )
        valid = segment_valid[batch_index, row] & segment_valid[batch_index, col]
        first_chain = chain_id[batch_index, row]
        second_chain = chain_id[batch_index, col]
        same_chain = first_chain == second_chain
        first_start = res_seq_nums[batch_index, row]
        first_end = res_seq_nums[batch_index, row + 1]
        second_start = res_seq_nums[batch_index, col]
        second_end = res_seq_nums[batch_index, col + 1]
        separation = torch.stack(
            (
                (first_start - second_start).abs(),
                (first_start - second_end).abs(),
                (first_end - second_start).abs(),
                (first_end - second_end).abs(),
            ),
            dim=-1,
        ).amin(dim=-1)
        valid &= (~same_chain) | (separation > seq_sep)
        pairs = torch.stack((row[valid], col[valid]), dim=-1)

        if spatial_prefilter and len(pairs):
            first, second = pairs.unbind(dim=-1)
            midpoint_distance = torch.linalg.vector_norm(
                segment_midpoint[batch_index, first] - segment_midpoint[batch_index, second],
                dim=-1,
            )
            contact_bound = (
                floor
                + segment_half_length[batch_index, first]
                + segment_half_length[batch_index, second]
            )
            margin = 64.0 * bound_eps * (midpoint_distance + contact_bound.abs() + 1.0)
            finite = torch.isfinite(midpoint_distance) & torch.isfinite(contact_bound)
            potentially_violating = (~finite) | (midpoint_distance <= contact_bound + margin)
            pairs = pairs[potentially_violating]

        for offset in range(0, len(pairs), pair_chunk_size):
            chunk = pairs[offset : offset + pair_chunk_size]
            first, second = chunk[:, 0], chunk[:, 1]
            p0, p1 = ca[batch_index, first], ca[batch_index, first + 1]
            q0, q1 = ca[batch_index, second], ca[batch_index, second + 1]
            s, t, residual = _closest_segment_parameters(p0, p1, q0, q1)
            fallback = _segment_zero_distance_direction(p0, p1, q0, q1, first, second)
            near_zero = torch.linalg.vector_norm(residual, dim=-1) < zero_jitter
            safe_residual = residual + near_zero[:, None] * fallback * zero_jitter
            distance = torch.linalg.vector_norm(safe_residual, dim=-1).clamp(min=1e-12)
            penetration = torch.relu(floor - distance)
            violating = penetration > 0.0
            squared_sum += penetration.square().sum()
            violating_count += violating.sum()

            residual_gradient = -2.0 * penetration[:, None] * safe_residual / distance[:, None]
            gradient[batch_index].index_add_(0, first, (1.0 - s)[:, None] * residual_gradient)
            gradient[batch_index].index_add_(0, first + 1, s[:, None] * residual_gradient)
            gradient[batch_index].index_add_(0, second, -(1.0 - t)[:, None] * residual_gradient)
            gradient[batch_index].index_add_(0, second + 1, -t[:, None] * residual_gradient)

    count = violating_count.clamp(min=1).to(coords.dtype)
    gradient /= count
    energy = squared_sum / count
    rms_penetration_A = torch.sqrt(squared_sum / count) * COORD_SCALE
    return gradient, energy, rms_penetration_A


def _steric_guidance_gradient(
    clean_estimate: Tensor,
    batch,
    config: GeometryGuidanceConfig,
    *,
    include_segment: bool = True,
) -> tuple[Tensor, Tensor]:
    """Return a coherent unit-RMS C-alpha self-avoidance gradient and severity."""
    with torch.enable_grad():
        coords = clean_estimate.detach().float().unsqueeze(0).requires_grad_(True)
        energy, rms_penetration_A = _nonlocal_ca_steric_energy(
            coords,
            batch.res_mask.bool(),
            batch.chain_id,
            batch.res_seq_nums,
            min_dist_A=config.steric_ca_min_dist_A,
            seq_sep=config.steric_ca_seq_sep,
        )
        raw = torch.autograd.grad(energy, coords, create_graph=False)[0]

    residue_vectors = raw.sum(dim=2)
    severity = rms_penetration_A / config.steric_severity_reference_A
    if config.steric_segment_weight and include_segment:
        segment_gradient, _, segment_rms_penetration_A = _nonlocal_ca_segment_guidance(
            coords.detach(),
            batch.atom_mask.bool()[..., CA_ATOM_ID] & batch.res_mask.bool(),
            batch.chain_id,
            batch.res_seq_nums,
            min_dist_A=config.steric_segment_min_dist_A,
            max_edge_A=config.steric_segment_max_edge_A,
            seq_sep=config.steric_ca_seq_sep,
            pair_chunk_size=config.steric_segment_pair_chunk_size,
        )
        residue_vectors += config.steric_segment_weight * segment_gradient
        segment_severity = (
            config.steric_segment_weight
            * segment_rms_penetration_A
            / config.steric_severity_reference_A
        )
        severity = torch.maximum(severity, segment_severity)
    residue_vectors = _smooth_residue_vectors(
        residue_vectors,
        batch.res_mask.bool(),
        batch.chain_id,
        batch.res_seq_nums,
        radius=config.steric_smoothing_radius,
        sigma=config.steric_smoothing_sigma,
    )
    residue_vectors = _normalize_residue_vectors(
        residue_vectors,
        batch.res_mask.bool(),
    )
    residue_vectors = _project_adjacent_bond_axis(
        residue_vectors,
        clean_estimate.detach().float().unsqueeze(0),
        batch.atom_mask.bool(),
        batch.res_mask.bool(),
        batch.chain_id,
        batch.res_seq_nums,
        iterations=config.steric_bond_projection_iters,
    )
    residue_vectors = _global_vector_norm_cap(residue_vectors)
    coherent = residue_vectors.unsqueeze(2).expand_as(raw)
    coherent = coherent * batch.atom_mask.bool().unsqueeze(-1)
    severity = severity.clamp(0.0, 1.0)
    return coherent.squeeze(0).to(dtype=clean_estimate.dtype), severity


def _smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def _steric_schedule(t: float, config: GeometryGuidanceConfig) -> float:
    ramp = _smoothstep((t - config.steric_start) / (config.steric_ramp_end - config.steric_start))
    if t <= config.steric_taper_start:
        return ramp
    taper_progress = _smoothstep(
        (t - config.steric_taper_start) / (1.0 - config.steric_taper_start)
    )
    taper = 1.0 - (1.0 - config.steric_taper_final) * taper_progress
    return ramp * taper


def _cap_guidance_step(drift: Tensor, dt: float, max_step_A: float) -> Tensor:
    """Globally cap a guidance-only step without changing relative vectors."""
    max_step = max_step_A / COORD_SCALE
    displacement_norm = abs(dt) * torch.linalg.vector_norm(drift, dim=-1).amax()
    factor = max_step / displacement_norm.clamp(min=max_step)
    return drift * factor


def _inference_autocast_dtype(device: str) -> torch.dtype:
    """Use BF16 where CUDA supports it and FP16 on older CUDA devices."""
    if not str(device).startswith("cuda"):
        return torch.bfloat16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _geometry_guidance_gradient(
    clean_estimate: Tensor,
    batch,
    config: GeometryGuidanceConfig,
    *,
    include_all_atom_clash: bool = True,
) -> Tensor:
    """Return a unit-RMS gradient of a GT-free geometry energy.

    The model graph stays detached: this backward pass differentiates only the
    lightweight coordinate energy, not the folding network.
    """
    with torch.enable_grad():
        coords = clean_estimate.detach().float().unsqueeze(0).requires_grad_(True)
        atom_mask = batch.atom_mask.bool()
        res_mask = batch.res_mask.bool()
        chain_id = batch.chain_id
        res_seq_nums = batch.res_seq_nums
        ca_mask = atom_mask[..., CA_ATOM_ID] & res_mask

        energy = coords.sum() * 0.0
        if config.bond_weight:
            energy = energy + config.bond_weight * bond_length_loss(
                coords,
                batch.res_type,
                atom_mask,
                res_mask,
                chain_id=chain_id,
                res_seq_nums=res_seq_nums,
            )
        if config.angle_weight:
            energy = energy + config.angle_weight * ca_virtual_angle_floor(
                coords[..., CA_ATOM_ID, :],
                ca_mask,
                chain_id,
                res_seq_nums=res_seq_nums,
            )
        if config.clash_weight:
            energy = energy + config.clash_weight * ca_clash_loss(
                coords,
                res_mask,
                chain_id=chain_id,
            )
        if config.covalent_weight:
            energy = energy + config.covalent_weight * covalent_geometry_loss(
                coords,
                batch.res_type,
                atom_mask,
                res_mask,
                chain_id=chain_id,
                res_seq_nums=res_seq_nums,
            )
        if config.peptide_planarity_weight:
            energy = energy + config.peptide_planarity_weight * peptide_planarity_loss(
                coords,
                atom_mask,
                res_mask,
                chain_id=chain_id,
                res_seq_nums=res_seq_nums,
            )
        if config.chirality_weight:
            energy = energy + config.chirality_weight * chemical_chirality_loss(
                coords,
                batch.res_type,
                atom_mask,
                res_mask,
            )
        if config.sidechain_planarity_weight:
            energy = energy + config.sidechain_planarity_weight * sidechain_planarity_loss(
                coords,
                batch.res_type,
                atom_mask,
                res_mask,
            )
        if config.ramachandran_weight:
            energy = energy + config.ramachandran_weight * ramachandran_outlier_loss(
                coords,
                batch.res_type,
                atom_mask,
                res_mask,
                chain_id=chain_id,
                res_seq_nums=res_seq_nums,
            )
        if config.all_atom_clash_weight and include_all_atom_clash:
            energy = energy + config.all_atom_clash_weight * all_atom_vdw_clash_loss(
                coords,
                batch.res_type,
                atom_mask,
                res_mask,
                chain_id=chain_id,
                res_seq_nums=res_seq_nums,
            )
        grad = torch.autograd.grad(energy, coords, create_graph=False)[0].squeeze(0)

    grad = _normalize_guidance_gradient(grad, atom_mask)
    return grad.to(dtype=clean_estimate.dtype)


def _vdw_guidance_gradient(
    clean_estimate: Tensor,
    batch,
    config: GeometryGuidanceConfig,
) -> Tensor:
    """Return a separately normalized severe-overlap VDW gradient.

    This channel is intentionally independent from the joint local-geometry
    energy.  Its scale therefore controls displacement magnitude instead of
    only changing the direction of a subsequently normalized mixed gradient.
    The default 1.5-A overlap tolerance matches the element-radius convention
    reported by OpenStructure's ``model_clashes`` output.
    """
    with torch.enable_grad():
        coords = clean_estimate.detach().float().unsqueeze(0).requires_grad_(True)
        atom_mask = batch.atom_mask.bool()
        res_mask = batch.res_mask.bool()
        energy = all_atom_vdw_clash_loss(
            coords,
            batch.res_type,
            atom_mask,
            res_mask,
            chain_id=batch.chain_id,
            res_seq_nums=batch.res_seq_nums,
            overlap_tolerance_A=config.vdw_overlap_tolerance_A,
        )
        grad = torch.autograd.grad(energy, coords, create_graph=False)[0].squeeze(0)

    grad = _normalize_guidance_gradient(grad, atom_mask)
    return grad.to(dtype=clean_estimate.dtype)


@torch.no_grad()
def sample(
    model,
    example,
    batch_fn: BatchFn,
    *,
    n_steps: int = 50,
    seed: int = 0,
    device: str = "cuda",
    sampler: str = "ode",
    sde_tau: float = 0.01,
    sde_eps: float = 0.01,
    sde_w_cutoff: float = 0.99,
    sde_log_timesteps: bool = True,
    record_trajectory: bool = True,
    geometry_guidance: GeometryGuidanceConfig | None = None,
):
    """Sample all atom slots with one flow-matching trajectory.

    ``sampler="ode"`` is the default Euler flow path. ``sampler="sde"`` follows
    SimpleFold's Euler-Maruyama solver for the linear flow path.

    Returns:
        final_ca: [L, 3] in Angstrom
        final_aa: [L, A, 3] in Angstrom
        traj_ca: [steps, L, 3] in Angstrom, or an empty [0, L, 3] array when
            ``record_trajectory=False``
        sched: time schedule
        conf: [L] predicted per-residue confidence (pLDDT in [0, 1])
    """
    model.eval()
    if geometry_guidance is not None:
        geometry_guidance.validate()
    atom_mask_f = example.atom_mask.unsqueeze(-1).float().to(device)
    L, A = example.atom_mask.shape
    torch.manual_seed(seed)
    # Match training noise: flow_corrupt centers ε to zero-mean over valid atoms,
    # so the prior here must be centered the same way or t→0 is off-distribution.
    x = torch.randn(L, A, 3, device=device)
    valid = example.atom_mask.reshape(-1).to(device)
    x = x - masked_centroid(x.reshape(-1, 3), valid).unsqueeze(0)
    x = x * atom_mask_f

    if sampler not in {"ode", "sde"}:
        raise ValueError(f"unknown sampler: {sampler}")

    if sampler == "sde" and sde_log_timesteps:
        sched = 1.0 - torch.logspace(-2, 0, n_steps + 1, device=device).flip(0)
        sched = sched - sched.min()
        sched = (sched / sched.max()).clamp(min=1e-4, max=1.0)
    elif sampler == "sde":
        sched = torch.linspace(1e-4, 1.0, n_steps + 1, device=device)
    else:
        sched = torch.linspace(0.0, _T_END, n_steps + 1, device=device)
    # Copy the tiny schedule once. Converting CUDA scalars to Python floats in
    # every step would otherwise force two device synchronizations per step.
    sched_cpu = sched.detach().cpu()
    traj = []
    amp_enabled = str(device).startswith("cuda")
    amp_dtype = _inference_autocast_dtype(device)
    x_self_cond = None
    for i in range(n_steps):
        ti = float(sched_cpu[i].clamp(min=1e-4))
        dt = float(sched_cpu[i + 1] - sched_cpu[i])
        x = (x - masked_centroid(x.reshape(-1, 3), valid).unsqueeze(0)) * atom_mask_f
        batch = batch_fn(x, ti)
        if x_self_cond is not None:
            batch = replace(batch, x_self_cond=x_self_cond.unsqueeze(0))
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
            out = model(batch, return_aux=False)
            v_atom = out["v_atom"].squeeze(0)
        clean_estimate = (x + (1.0 - ti) * v_atom).detach() * atom_mask_f
        x_self_cond = clean_estimate
        guidance_drift = None
        if (
            geometry_guidance is not None
            and geometry_guidance.scale > 0.0
            and ti >= geometry_guidance.start
            and i % geometry_guidance.every_n_steps == 0
        ):
            progress = (ti - geometry_guidance.start) / (1.0 - geometry_guidance.start)
            guidance_index = i // geometry_guidance.every_n_steps
            include_all_atom_clash = (
                guidance_index % geometry_guidance.all_atom_clash_every_n_steps == 0
            )
            grad = _geometry_guidance_gradient(
                clean_estimate,
                batch,
                geometry_guidance,
                include_all_atom_clash=include_all_atom_clash,
            )
            guidance_drift = -geometry_guidance.scale * progress * grad
        if (
            geometry_guidance is not None
            and geometry_guidance.vdw_scale > 0.0
            and ti >= geometry_guidance.vdw_start
            and i % geometry_guidance.vdw_every_n_steps == 0
        ):
            vdw_progress = (ti - geometry_guidance.vdw_start) / (1.0 - geometry_guidance.vdw_start)
            vdw_grad = _vdw_guidance_gradient(
                clean_estimate,
                batch,
                geometry_guidance,
            )
            vdw_drift = -geometry_guidance.vdw_scale * vdw_progress * vdw_grad
            vdw_drift = _cap_guidance_step(
                vdw_drift,
                dt,
                geometry_guidance.vdw_max_step_A,
            )
            guidance_drift = vdw_drift if guidance_drift is None else guidance_drift + vdw_drift
        if (
            geometry_guidance is not None
            and geometry_guidance.steric_scale > 0.0
            and ti >= geometry_guidance.steric_start
            and i % geometry_guidance.steric_every_n_steps == 0
        ):
            steric_index = i // geometry_guidance.steric_every_n_steps
            steric_grad, severity = _steric_guidance_gradient(
                clean_estimate,
                batch,
                geometry_guidance,
                include_segment=(
                    steric_index % geometry_guidance.steric_segment_every_n_steps == 0
                ),
            )
            steric_drift = (
                -geometry_guidance.steric_scale
                * _steric_schedule(ti, geometry_guidance)
                * severity.to(dtype=steric_grad.dtype)
                * steric_grad
            )
            steric_drift = _cap_guidance_step(
                steric_drift,
                dt,
                geometry_guidance.steric_max_step_A,
            )
            guidance_drift = (
                steric_drift if guidance_drift is None else guidance_drift + steric_drift
            )
        if sampler == "sde":
            if ti >= sde_w_cutoff:
                w = 0.0
            else:
                w = (1.0 - ti) / (ti + sde_eps)
            score = ((ti * v_atom) - x) / max(1.0 - ti, 1e-6)
            drift = v_atom + w * score
            if guidance_drift is not None:
                drift = drift + guidance_drift
            x = (x + dt * drift) * atom_mask_f
            noise_scale = math.sqrt(max(2.0 * dt * w * sde_tau, 0.0))
            if noise_scale > 0.0 and i < n_steps - 1:
                noise = torch.randn_like(x) * atom_mask_f
                noise = (
                    noise - masked_centroid(noise.reshape(-1, 3), valid).unsqueeze(0)
                ) * atom_mask_f
                x = (x + noise_scale * noise) * atom_mask_f
        else:
            drift = v_atom if guidance_drift is None else v_atom + guidance_drift
            x = (x + dt * drift) * atom_mask_f
        # Re-center to zero CoM each step: training data/noise are centroid-centered,
        # and the model is not translation-equivariant, so accumulated Euler drift
        # would walk the input off-distribution.
        x = (x - masked_centroid(x.reshape(-1, 3), valid).unsqueeze(0)) * atom_mask_f
        if record_trajectory:
            traj.append(x[:, CA_ATOM_ID, :].float().cpu().numpy() * COORD_SCALE)

    batch = batch_fn(x, float(sched_cpu[-1]))
    if x_self_cond is not None:
        batch = replace(batch, x_self_cond=x_self_cond.unsqueeze(0))
    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
        out = model(batch, return_aux=False)
        v_final = out["v_atom"].squeeze(0)
        conf = out["conf"].squeeze(0).float().cpu().numpy()  # [L] predicted pLDDT
    if sampler == "sde":
        x_clean = x * atom_mask_f
    else:
        x_clean = (x + (1.0 - float(sched[-1])) * v_final) * atom_mask_f

    final_aa = x_clean.float().cpu().numpy() * COORD_SCALE
    final_ca = final_aa[:, CA_ATOM_ID, :]
    if record_trajectory:
        traj_ca = np.asarray(traj, dtype=np.float32)
    else:
        traj_ca = np.empty((0, L, 3), dtype=np.float32)
    return final_ca, final_aa, traj_ca, sched_cpu.numpy(), conf
