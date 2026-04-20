"""Equilibrium Matching loss functions."""

import torch
from torch import Tensor


def truncated_c(gamma: Tensor, a: float = 0.8, lam: float = 4.0) -> Tensor:
    """Compute c(γ) = λ · c_trunc(γ, a).

    c_trunc(γ) = 1          if γ ≤ a
               = (1-γ)/(1-a) if γ > a

    Args:
        gamma: [B, 1, 1, 1] interpolation factors
        a: truncation point
        lam: gradient multiplier

    Returns: [B, 1, 1, 1] gradient magnitude
    """
    c = torch.where(gamma <= a, torch.ones_like(gamma), (1 - gamma) / (1 - a))
    return lam * c


def eqm_reconstruction_scale(gamma: Tensor, a: float = 0.8, lam: float = 4.0) -> Tensor:
    """Compute scale for x̂ = x_γ - scale · f(x_γ).

    Derived from: f(x_γ) ≈ (ε - x)·c(γ), x_γ = γx + (1-γ)ε
    => x̂ = x_γ - ((1-γ)/c(γ)) · f(x_γ)

    For c_trunc:
      γ ≤ a: scale = (1-γ) / (lam·1) = (1-γ)/lam
      γ > a: scale = (1-γ) / (lam·(1-γ)/(1-a)) = (1-a)/lam

    Args:
        gamma: [B, 1, 1, 1]

    Returns: [B, 1, 1, 1] stable reconstruction scale
    """
    scale = torch.where(
        gamma <= a,
        (1 - gamma) / lam,
        torch.full_like(gamma, (1 - a) / lam),
    )
    return scale


def eqm_loss(
    pred: Tensor,
    x_clean: Tensor,
    eps: Tensor,
    gamma: Tensor,
    valid_mask: Tensor,
    a: float = 0.8,
    lam: float = 4.0,
) -> Tensor:
    """Compute EqM training loss.

    L_EqM = ||f(x_γ) - (ε - x)·c(γ)||²  (masked mean)

    Args:
        pred: [B, L, A, 3] predicted gradient
        x_clean: [B, L, A, 3] ground truth coords (normalized)
        eps: [B, L, A, 3] noise
        gamma: [B, 1, 1, 1] interpolation factor
        valid_mask: [B, L, A] valid atom mask
        a: truncation point for c(γ)
        lam: gradient multiplier

    Returns: scalar loss
    """
    c = truncated_c(gamma, a, lam)          # [B, 1, 1, 1]
    target = (eps - x_clean) * c            # [B, L, A, 3]
    diff_sq = (pred - target).pow(2).sum(dim=-1)  # [B, L, A]
    mask = valid_mask.to(diff_sq.dtype)
    loss = (diff_sq * mask).sum() / mask.sum().clamp(min=1)
    return loss
