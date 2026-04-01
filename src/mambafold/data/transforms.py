"""Data transforms for protein structure training."""

import torch
from torch import Tensor

from mambafold.data.constants import COORD_SCALE
from mambafold.data.types import ProteinExample
from mambafold.utils.geometry import apply_rotation, masked_centroid, random_rotation_matrix


def center_and_scale(example: ProteinExample) -> ProteinExample:
    """Center on heavy-atom centroid and scale to normalized units."""
    flat_coords = example.coords.reshape(-1, 3)  # [L*A, 3]
    flat_mask = example.atom_mask.reshape(-1)  # [L*A]
    centroid = masked_centroid(flat_coords, flat_mask)  # [1, 3]
    coords = (example.coords - centroid.unsqueeze(0)) / COORD_SCALE
    return ProteinExample(
        res_type=example.res_type,
        atom_type=example.atom_type,
        pair_type=example.pair_type,
        coords=coords,
        atom_mask=example.atom_mask,
        observed_mask=example.observed_mask,
        res_seq_nums=example.res_seq_nums,
        seq_len=example.seq_len,
    )


def random_so3_augment(example: ProteinExample) -> ProteinExample:
    """Apply random SO(3) rotation to coordinates."""
    rot = random_rotation_matrix(device=example.coords.device)
    coords = apply_rotation(example.coords, rot)
    return ProteinExample(
        res_type=example.res_type,
        atom_type=example.atom_type,
        pair_type=example.pair_type,
        coords=coords,
        atom_mask=example.atom_mask,
        observed_mask=example.observed_mask,
        res_seq_nums=example.res_seq_nums,
        seq_len=example.seq_len,
    )


def _sample_gamma(schedule: str = "logit_normal") -> float:
    """Sample γ from the specified schedule.

    schedule:
      "logit_normal" — p(γ) = 0.90·LN(μ=0.4, σ=1.5) + 0.10·U(0,1)
                       Balanced coverage for both low and high noise regions.
      "uniform"      — γ ~ U(0, 1), covers all noise levels equally.
    """
    if schedule == "uniform":
        return float(torch.empty(1).uniform_(0.0, 1.0).item())
    # logit_normal (default)
    if torch.rand(1).item() < 0.90:
        z = torch.randn(1).mul_(1.5).add_(0.4)
        gamma = torch.sigmoid(z).item()
    else:
        gamma = torch.empty(1).uniform_(0.0, 1.0).item()
    return float(gamma)


def eqm_corrupt(
    coords: Tensor,
    atom_mask: Tensor,
    schedule: str = "logit_normal",
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply EqM corruption: x_γ = γ·x + (1-γ)·ε.

    Args:
        coords: [L, A, 3] clean normalized coordinates
        atom_mask: [L, A] valid atoms
        schedule: gamma sampling schedule ("logit_normal" | "uniform")

    Returns:
        x_gamma: [L, A, 3] corrupted coordinates
        eps: [L, A, 3] noise sample
        gamma: scalar sampled from schedule
    """
    eps = torch.randn_like(coords)
    # Center eps so training and inference share the same zero-mean noise distribution
    eps_centroid = masked_centroid(eps.reshape(-1, 3), atom_mask.reshape(-1))  # [1, 3]
    eps = eps - eps_centroid.unsqueeze(0)
    gamma = _sample_gamma(schedule)

    x_gamma = gamma * coords + (1 - gamma) * eps
    mask_f = atom_mask.unsqueeze(-1).to(coords.dtype)
    x_gamma = x_gamma * mask_f
    eps = eps * mask_f

    return x_gamma, eps, gamma
