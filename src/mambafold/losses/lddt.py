"""Differentiable LDDT loss for structure quality."""

import torch
from torch import Tensor

from mambafold.data.constants import CA_ATOM_ID


def soft_lddt_ca_loss(
    pred_coords: Tensor,
    true_coords: Tensor,
    ca_mask: Tensor,
    cutoff: float = 1.5,
    thresholds: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4),
) -> Tensor:
    """Soft differentiable CA-LDDT loss.

    Computes LDDT on C-alpha atoms only (cheaper than all-atom).

    Args:
        pred_coords: [B, L, A, 3] predicted coordinates
        true_coords: [B, L, A, 3] ground truth coordinates
        ca_mask: [B, L] bool — residues with valid C-alpha
        cutoff: distance cutoff in normalized units (default 1.5 = 15Å)
        thresholds: LDDT distance thresholds in normalized units
            (default = 0.5,1,2,4 Å ÷ COORD_SCALE)

    Returns: scalar loss (1 - mean LDDT)
    """
    # Extract CA coordinates (atom14 slot order: N, CA, C, O, ...)
    pred_ca = pred_coords[:, :, CA_ATOM_ID, :]  # [B, L, 3]
    true_ca = true_coords[:, :, CA_ATOM_ID, :]  # [B, L, 3]

    # Pairwise distances
    pred_dist = torch.linalg.norm(pred_ca.unsqueeze(2) - pred_ca.unsqueeze(1), dim=-1)  # [B, L, L]
    true_dist = torch.linalg.norm(true_ca.unsqueeze(2) - true_ca.unsqueeze(1), dim=-1)  # [B, L, L]

    # Pair mask: both residues valid, within cutoff, not self
    pair_mask = ca_mask.unsqueeze(2) & ca_mask.unsqueeze(1)  # [B, L, L]
    pair_mask = pair_mask & (true_dist < cutoff)
    pair_mask = pair_mask & ~torch.eye(
        pred_ca.shape[1], dtype=torch.bool, device=pred_ca.device
    ).unsqueeze(0)

    # Soft LDDT per pair
    dist_error = torch.abs(pred_dist - true_dist)
    lddt_per_pair = sum(torch.sigmoid((thr - dist_error) * 5.0) for thr in thresholds) / len(
        thresholds
    )  # [B, L, L]

    # Masked mean
    mask_f = pair_mask.to(lddt_per_pair.dtype)
    lddt_score = (lddt_per_pair * mask_f).sum() / mask_f.sum().clamp(min=1)

    return 1.0 - lddt_score  # loss = 1 - LDDT


def soft_lddt_all_atom_loss(
    pred_coords: Tensor,
    true_coords: Tensor,
    valid_mask: Tensor,
    cutoff: float = 1.5,
    thresholds: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4),
    max_atoms: int = 2048,
) -> Tensor:
    """Sampled differentiable all-atom LDDT loss.

    Full atom-pair LDDT is O((L*A)^2), which is too expensive at L=1024.
    This computes the same local-distance objective on a deterministic subset
    of valid atom slots. The subset keeps the loss stable across ranks and
    avoids RNG inside the loss.
    """
    losses = []
    B = pred_coords.shape[0]
    for b in range(B):
        flat_mask = valid_mask[b].reshape(-1)
        idx = flat_mask.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() < 2:
            continue
        if idx.numel() > max_atoms:
            take = torch.linspace(0, idx.numel() - 1, max_atoms, device=idx.device).long()
            idx = idx[take]

        pred = pred_coords[b].reshape(-1, 3).index_select(0, idx)
        true = true_coords[b].reshape(-1, 3).index_select(0, idx)
        pred_dist = torch.cdist(pred, pred)
        true_dist = torch.cdist(true, true)

        pair_mask = true_dist < cutoff
        pair_mask = pair_mask & ~torch.eye(idx.numel(), dtype=torch.bool, device=idx.device)
        m = pair_mask.to(pred_dist.dtype)
        if m.sum() < 1:
            continue

        dist_error = (pred_dist - true_dist).abs()
        score = sum(torch.sigmoid((thr - dist_error) * 5.0) for thr in thresholds) / len(thresholds)
        losses.append(1.0 - (score * m).sum() / m.sum().clamp(min=1))

    if not losses:
        return pred_coords.new_zeros(())
    return torch.stack(losses).mean()
