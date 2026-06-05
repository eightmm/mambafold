"""CA-only loss helpers for Stage 1 (FM on CA coordinates).

These helpers operate on [B, L, 3] CA tensors directly, instead of the
[B, L, A, 3] all-atom shape used by Stage 2.

Coordinates are in normalized units (Å ÷ COORD_SCALE = 10).
"""

from __future__ import annotations

import torch
from torch import Tensor

from mambafold.data.constants import COORD_SCALE

_NORM_CA_CA_BOND = 3.81 / COORD_SCALE  # adjacent in-chain CA-CA ≈ 0.381


def soft_lddt_ca_only(
    pred_ca: Tensor,
    true_ca: Tensor,
    ca_mask: Tensor,
    cutoff: float = 1.5,
    thresholds: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4),
) -> Tensor:
    """Soft differentiable Cα lDDT on [B, L, 3] tensors.

    Returns `1 - mean_lddt` so smaller is better (suitable for loss minimisation).
    """
    pred_d = torch.linalg.norm(pred_ca.unsqueeze(2) - pred_ca.unsqueeze(1), dim=-1)
    true_d = torch.linalg.norm(true_ca.unsqueeze(2) - true_ca.unsqueeze(1), dim=-1)

    pair = ca_mask.unsqueeze(2) & ca_mask.unsqueeze(1) & (true_d < cutoff)
    eye = torch.eye(pred_ca.shape[1], dtype=torch.bool, device=pred_ca.device).unsqueeze(0)
    pair = pair & ~eye
    m = pair.to(pred_d.dtype)

    diff = torch.abs(pred_d - true_d)
    lddt = sum(torch.sigmoid((thr - diff) * 5.0) for thr in thresholds) / len(thresholds)
    denom = m.sum().clamp(min=1)
    return 1.0 - (lddt * m).sum() / denom


def ca_ca_bond_loss(pred_ca: Tensor, ca_mask: Tensor, chain_id: Tensor) -> Tensor:
    """Huber-style adjacent in-chain Cα-Cα bond-length deviation from 3.81 Å.

    Only penalises (i, i+1) pairs where both residues are valid AND in the
    same chain (so chain boundaries aren't punished).
    """
    if pred_ca.shape[1] < 2:
        return pred_ca.new_zeros(())
    d = torch.linalg.norm(pred_ca[:, 1:] - pred_ca[:, :-1], dim=-1)         # [B, L-1]
    adj = (ca_mask[:, :-1] & ca_mask[:, 1:]
           & (chain_id[:, :-1] == chain_id[:, 1:])).to(d.dtype)              # [B, L-1]
    err = (d - _NORM_CA_CA_BOND).abs()
    denom = adj.sum().clamp(min=1)
    return (err * adj).sum() / denom


def distogram_loss_ca_only(
    dist_logits: Tensor,         # [B, L, L, n_bins]
    true_ca: Tensor,             # [B, L, 3] normalised
    ca_mask: Tensor,             # [B, L]
    n_bins: int = 64,
    max_dist_ang: float = 22.0,  # Å
) -> Tensor:
    """Cross-entropy on binned Cα-Cα distance — Stage 1 variant.

    Same as `losses.distogram.distogram_loss` but takes CA-only true coords
    directly, no atom-slot indexing. Returns 0 if no valid non-self pairs.
    """
    B, L = ca_mask.shape
    ca_ang = true_ca * COORD_SCALE
    dist = torch.linalg.norm(ca_ang.unsqueeze(2) - ca_ang.unsqueeze(1), dim=-1)  # [B,L,L]

    bin_width = max_dist_ang / n_bins
    bin_idx = (dist / bin_width).clamp(0, n_bins - 1).long()

    pair = ca_mask.unsqueeze(2) & ca_mask.unsqueeze(1)
    eye = torch.eye(L, dtype=torch.bool, device=ca_ang.device).unsqueeze(0)
    pair = pair & ~eye

    import torch.nn.functional as F
    logits_flat = dist_logits.reshape(-1, n_bins)
    target_flat = bin_idx.reshape(-1)
    mask_flat = pair.reshape(-1)
    if not mask_flat.any():
        return dist_logits.new_zeros(())
    ce = F.cross_entropy(
        logits_flat[mask_flat], target_flat[mask_flat], reduction="mean",
    )
    return ce
