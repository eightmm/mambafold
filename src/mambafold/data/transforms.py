"""Data transforms for protein structure training."""

from dataclasses import replace

import torch
from torch import Tensor

from mambafold.data.constants import COORD_SCALE
from mambafold.data.types import ProteinExample
from mambafold.utils.geometry import apply_rotation, masked_centroid, random_rotation_matrix


def _with_coords(example: ProteinExample, coords: Tensor) -> ProteinExample:
    """Return a copy of `example` with coords replaced, preserving all other fields."""
    return replace(example, coords=coords)


def center_and_scale(example: ProteinExample) -> ProteinExample:
    """Center on the observed-atom centroid and scale to normalized units."""
    flat_coords = example.coords.reshape(-1, 3)   # [L*A, 3]
    flat_mask = (example.atom_mask & example.observed_mask).reshape(-1)     # [L*A]
    if not flat_mask.any():
        flat_mask = example.atom_mask.reshape(-1)
    centroid = masked_centroid(flat_coords, flat_mask)  # [1, 3]
    coords = (example.coords - centroid.unsqueeze(0)) / COORD_SCALE
    return _with_coords(example, coords)


def random_so3_augment(example: ProteinExample) -> ProteinExample:
    """Apply a single SO(3) rotation to every atom."""
    rot = random_rotation_matrix(device=example.coords.device)
    coords = apply_rotation(example.coords, rot)
    return _with_coords(example, coords)


def _sample_t(schedule: str = "uniform") -> float:
    """Sample t ∈ [0, 1] from the specified schedule.

    schedule:
      "uniform"      — t ~ U(0, 1), all noise levels equally (FM standard).
      "logit_normal" — p(t) = 0.98·LN(μ=0.8, σ=1.7) + 0.02·U(0,1),
                       SimpleFold-style oversampling near t→1 (clean).
    """
    if schedule == "logit_normal":
        if torch.rand(1).item() < 0.98:
            z = torch.randn(1).mul_(1.7).add_(0.8)
            return float(torch.sigmoid(z).item())
        return float(torch.empty(1).uniform_(0.0, 1.0).item())
    return float(torch.empty(1).uniform_(0.0, 1.0).item())


def flow_corrupt(
    coords: Tensor,
    atom_mask: Tensor,
    schedule: str = "uniform",
) -> tuple[Tensor, Tensor, Tensor]:
    """Flow-matching corruption: x_t = t·x_clean + (1-t)·ε.

    Args:
        coords: [L, A, 3] clean normalized coordinates
        atom_mask: [L, A] valid atoms
        schedule: time sampling schedule ("uniform" | "logit_normal")

    Returns:
        x_t:   [L, A, 3] interpolated coordinates
        eps:   [L, A, 3] noise (zero-mean over valid atoms)
        t:     scalar in [0, 1]
    """
    eps = torch.randn_like(coords)
    # Center eps so training and inference share the same zero-mean noise distribution.
    eps_centroid = masked_centroid(eps.reshape(-1, 3), atom_mask.reshape(-1))  # [1, 3]
    eps = eps - eps_centroid.unsqueeze(0)
    t = _sample_t(schedule)

    x_t = t * coords + (1 - t) * eps
    mask_f = atom_mask.unsqueeze(-1).to(coords.dtype)
    x_t = x_t * mask_f
    eps = eps * mask_f

    return x_t, eps, t
