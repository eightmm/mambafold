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
    """Center on the canonical-atom centroid and scale to normalized units.

    The centroid is taken over `atom_mask` — every canonical atom slot — and not
    over the resolved subset, because that is what SimpleFold does: its
    `center_random_augmentation` is called with `atom_pad_mask`, and
    `atom_resolved_mask` is reserved for the loss. Unresolved atoms carry (0,0,0)
    in the Boltz records, so this centroid is pulled toward the deposition
    frame's origin rather than sitting on the structure. That is immaterial to
    training — the FM target is rigid-aligned, which removes any translation, and
    lDDT is built from pairwise distances — but it is a real difference from
    centering on the resolved atoms, which is what this used to do.
    """
    flat_coords = example.coords.reshape(-1, 3)   # [L*A, 3]
    flat_mask = example.atom_mask.reshape(-1)     # [L*A]
    centroid = masked_centroid(flat_coords, flat_mask)  # [1, 3]
    coords = (example.coords - centroid.unsqueeze(0)) / COORD_SCALE
    return _with_coords(example, coords)


def random_so3_augment(example: ProteinExample) -> ProteinExample:
    """Apply a single SO(3) rotation to every atom."""
    rot = random_rotation_matrix(device=example.coords.device)
    coords = apply_rotation(example.coords, rot)
    return _with_coords(example, coords)


def random_se3_augment(
    example: ProteinExample,
    translation_std: float = 1.0,
) -> ProteinExample:
    """Apply SimpleFold-style random rotation and normalized translation."""
    rotated = random_so3_augment(example)
    translation = torch.randn(3, device=rotated.coords.device, dtype=rotated.coords.dtype)
    coords = rotated.coords + translation_std * translation
    return _with_coords(rotated, coords)


def _sample_t(schedule: str = "uniform", uniform_weight: float = 0.02) -> float:
    """Sample t ∈ [0, 1] from the specified schedule.

    schedule:
      "uniform"      — t ~ U(0, 1), all noise levels equally (FM standard).
      "logit_normal" — t = (1-w)·sigmoid(N(0.8, 1.7)) + w·U(0,1). SimpleFold
                       uses w = 0.02; `uniform_weight` widens the floor.
    """
    if schedule == "logit_normal":
        z = torch.randn(1).mul_(1.7).add_(0.8)
        sampled = (1.0 - uniform_weight) * torch.sigmoid(z) + uniform_weight * torch.rand(1)
    elif schedule == "uniform":
        sampled = torch.rand(1)
    else:
        raise ValueError(f"unknown time schedule: {schedule!r}")
    # SimpleFold bounds training time using its Euler-Maruyama t_start.
    sampled = sampled * (1.0 - 2.0e-4) + 1.0e-4
    return float(sampled.item())


def flow_corrupt(
    coords: Tensor,
    atom_mask: Tensor,
    schedule: str = "uniform",
    uniform_weight: float = 0.02,
) -> tuple[Tensor, Tensor, Tensor]:
    """Flow-matching corruption: x_t = t·x_clean + (1-t)·ε.

    Args:
        coords: [L, A, 3] clean normalized coordinates
        atom_mask: [L, A] valid atoms
        schedule: time sampling schedule ("uniform" | "logit_normal")

    Returns:
        x_t:   [L, A, 3] interpolated coordinates
        eps:   [L, A, 3] standard Gaussian noise on valid atoms
        t:     scalar in [0, 1]
    """
    eps = torch.randn_like(coords)
    t = _sample_t(schedule, uniform_weight)

    x_t = t * coords + (1 - t) * eps
    mask_f = atom_mask.unsqueeze(-1).to(coords.dtype)
    x_t = x_t * mask_f
    eps = eps * mask_f

    return x_t, eps, t
