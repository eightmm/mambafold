"""CA-only loss helpers for Stage 1 (FM on CA coordinates).

These helpers operate on [B, L, 3] CA tensors directly, instead of the
[B, L, A, 3] all-atom shape used by Stage 2.

Coordinates are in normalized units (Å ÷ COORD_SCALE = 10).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from mambafold.data.constants import CA_ATOM_ID, COORD_SCALE

_NORM_CA_CA_BOND = 3.81 / COORD_SCALE  # adjacent in-chain CA-CA ≈ 0.381
_CB_SLOT = 4                           # atom14 slot order: N, CA, C, O, CB, ...
_LDDT_THRESHOLDS = (0.05, 0.1, 0.2, 0.4)  # = (0.5, 1, 2, 4) Å ÷ COORD_SCALE


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

    logits_flat = dist_logits.reshape(-1, n_bins)
    target_flat = bin_idx.reshape(-1)
    mask_flat = pair.reshape(-1)
    if not mask_flat.any():
        return dist_logits.new_zeros(())
    ce = F.cross_entropy(
        logits_flat[mask_flat], target_flat[mask_flat], reduction="mean",
    )
    return ce


def _valid_ca_pairs(ca_mask: Tensor) -> Tensor:
    """[B, L, L] bool — both residues have CA, excluding the diagonal."""
    L = ca_mask.shape[1]
    pair = ca_mask.unsqueeze(2) & ca_mask.unsqueeze(1)
    eye = torch.eye(L, dtype=torch.bool, device=ca_mask.device).unsqueeze(0)
    return pair & ~eye


def drmsd_ca(pred_ca: Tensor, true_ca: Tensor, ca_mask: Tensor) -> Tensor:
    """Distance-matrix RMSD on Cα (rotation/translation invariant topology loss).

    Penalizes the whole pairwise Cα-Cα distance map, so a globally-misfolded
    topology is punished even when a rigid alignment happens to look decent.
    """
    pred_d = torch.linalg.norm(pred_ca.unsqueeze(2) - pred_ca.unsqueeze(1), dim=-1)
    true_d = torch.linalg.norm(true_ca.unsqueeze(2) - true_ca.unsqueeze(1), dim=-1)
    m = _valid_ca_pairs(ca_mask).to(pred_d.dtype)
    sq = (pred_d - true_d).pow(2)
    return ((sq * m).sum() / m.sum().clamp(min=1)).sqrt()


def pseudo_cb_loss(
    pcb_pred: Tensor, x_clean: Tensor, atom_mask: Tensor, res_mask: Tensor,
) -> Tensor:
    """Cosine loss between predicted pseudo-Cβ direction and GT (Cβ − Cα).

    Gives Stage 1 a side-chain orientation cue for Stage 2. Residues without a
    resolved Cβ (GLY or missing) are masked out.
    """
    ca = x_clean[..., CA_ATOM_ID, :]
    cb = x_clean[..., _CB_SLOT, :]
    gt_dir = F.normalize(cb - ca, dim=-1)
    pred_dir = F.normalize(pcb_pred, dim=-1)
    cos = (pred_dir * gt_dir).sum(dim=-1)                       # [B, L]
    valid = atom_mask[..., _CB_SLOT] & atom_mask[..., CA_ATOM_ID] & res_mask
    m = valid.to(cos.dtype)
    return ((1.0 - cos) * m).sum() / m.sum().clamp(min=1)


def per_residue_lddt_ca(
    pred_ca: Tensor, true_ca: Tensor, ca_mask: Tensor,
    cutoff: float = 1.5, thresholds: tuple[float, ...] = _LDDT_THRESHOLDS,
) -> Tensor:
    """Per-residue soft Cα-lDDT score in [0, 1]. Used as the confidence target."""
    pred_d = torch.linalg.norm(pred_ca.unsqueeze(2) - pred_ca.unsqueeze(1), dim=-1)
    true_d = torch.linalg.norm(true_ca.unsqueeze(2) - true_ca.unsqueeze(1), dim=-1)
    pair = _valid_ca_pairs(ca_mask) & (true_d < cutoff)
    m = pair.to(pred_d.dtype)
    diff = (pred_d - true_d).abs()
    lddt = sum(torch.sigmoid((thr - diff) * 5.0) for thr in thresholds) / len(thresholds)
    return (lddt * m).sum(dim=2) / m.sum(dim=2).clamp(min=1)    # [B, L]


def confidence_loss(
    conf_pred: Tensor, x_hat_ca: Tensor, x_clean_ca: Tensor, ca_mask: Tensor,
) -> Tensor:
    """MSE between predicted confidence and the (detached) per-residue Cα-lDDT.

    Calibrates Stage 1 confidence so Stage 2 can trust core residues and refine
    uncertain ones (loops, termini).
    """
    target = per_residue_lddt_ca(x_hat_ca, x_clean_ca, ca_mask).detach()
    m = ca_mask.to(conf_pred.dtype)
    return (((conf_pred - target) ** 2) * m).sum() / m.sum().clamp(min=1)


def contact_loss_ca(
    logits: Tensor, true_ca: Tensor, ca_mask: Tensor,
    thresh_ang: float = 8.0, longrange_sep: int = 24, lr_weight: float = 4.0,
) -> Tensor:
    """Long-range-weighted Cα contact BCE (target: Cα-Cα < `thresh_ang` Å).

    Up-weights |i−j| > `longrange_sep` pairs so the (Mamba) trunk is pushed to
    encode tertiary contacts / β-sheet pairing it cannot see by sequence scan.
    """
    logits = logits.squeeze(-1) if logits.dim() == 4 else logits   # [B, L, L]
    true_d = torch.linalg.norm(true_ca.unsqueeze(2) - true_ca.unsqueeze(1), dim=-1) * COORD_SCALE
    target = (true_d < thresh_ang).to(logits.dtype)
    L = ca_mask.shape[1]
    sep = (torch.arange(L, device=logits.device).unsqueeze(0)
           - torch.arange(L, device=logits.device).unsqueeze(1)).abs()       # [L, L]
    w = torch.where(sep > longrange_sep, lr_weight, 1.0).unsqueeze(0)         # [1, L, L]
    m = _valid_ca_pairs(ca_mask).to(logits.dtype) * w
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (bce * m).sum() / m.sum().clamp(min=1)


def ca_virtual_angle_floor(
    pred_ca: Tensor, ca_mask: Tensor, chain_id: Tensor, min_rad: float = 1.2,
) -> Tensor:
    """Penalize only physically impossible (too-sharp) Cα virtual bond angles.

    Angle at i = ∠(Cα_{i-1}, Cα_i, Cα_{i+1}). A hinge floor (default ~69°) avoids
    over-constraining helix/strand geometry while killing degenerate kinks.
    """
    if pred_ca.shape[1] < 3:
        return pred_ca.new_zeros(())
    v1 = F.normalize(pred_ca[:, :-2] - pred_ca[:, 1:-1], dim=-1)
    v2 = F.normalize(pred_ca[:, 2:] - pred_ca[:, 1:-1], dim=-1)
    cos = (v1 * v2).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
    ang = torch.arccos(cos)                                                  # [B, L-2]
    same = (chain_id[:, :-2] == chain_id[:, 1:-1]) & (chain_id[:, 1:-1] == chain_id[:, 2:])
    valid = ca_mask[:, :-2] & ca_mask[:, 1:-1] & ca_mask[:, 2:] & same
    m = valid.to(ang.dtype)
    viol = F.relu(min_rad - ang)
    return (viol * m).sum() / m.sum().clamp(min=1)


def ca_chirality_loss(
    pred_ca: Tensor, true_ca: Tensor, ca_mask: Tensor, chain_id: Tensor,
    margin: float = 0.1,
) -> Tensor:
    """Penalize mirror-image (wrong global handedness) predictions.

    The signed triple product of three consecutive unit Cα-Cα bond vectors,
    tp = (b1 × b2) · b3, flips sign under reflection. Every other Cα aux
    (lDDT, dRMSD, distogram, contact) is reflection-invariant, so a perfectly
    mirrored structure scores well on all of them — measured ~35% of holdout
    predictions are internally near-perfect mirror images. This hinge drives the
    predicted chirality sign to match the ground truth, weighted by |tp_true| so
    only genuinely chiral windows (helices/strands) contribute and locally
    planar stretches don't add noise. Per 4-residue window within one chain.
    """
    if pred_ca.shape[1] < 4:
        return pred_ca.new_zeros(())

    def _tp(ca: Tensor) -> Tensor:
        b1 = F.normalize(ca[:, 1:-2] - ca[:, :-3], dim=-1)
        b2 = F.normalize(ca[:, 2:-1] - ca[:, 1:-2], dim=-1)
        b3 = F.normalize(ca[:, 3:] - ca[:, 2:-1], dim=-1)
        return (torch.linalg.cross(b1, b2, dim=-1) * b3).sum(dim=-1)   # [B, L-3] ∈ [-1, 1]

    tp_pred = _tp(pred_ca)
    tp_true = _tp(true_ca)
    same = ((chain_id[:, :-3] == chain_id[:, 1:-2])
            & (chain_id[:, 1:-2] == chain_id[:, 2:-1])
            & (chain_id[:, 2:-1] == chain_id[:, 3:]))
    valid = (ca_mask[:, :-3] & ca_mask[:, 1:-2]
             & ca_mask[:, 2:-1] & ca_mask[:, 3:] & same)
    w = tp_true.abs() * valid.to(tp_pred.dtype)               # focus on chiral windows
    # Want tp_pred to share tp_true's sign with magnitude ≥ margin (hinge floors
    # at margin when tp_pred=0, so the model can't dodge by collapsing tp→0).
    viol = F.relu(margin - tp_pred * torch.sign(tp_true))
    return (viol * w).sum() / w.sum().clamp(min=1)


def ca_self_clash(
    pred_ca: Tensor, ca_mask: Tensor, chain_id: Tensor,
    min_dist_ang: float = 3.6, seq_sep: int = 2,
) -> Tensor:
    """Cα self-avoidance: squared-ReLU penalty on non-adjacent Cα closer than
    `min_dist_ang` Å (same chain). Stops the scaffold from self-intersecting."""
    min_d = min_dist_ang / COORD_SCALE
    d = torch.linalg.norm(pred_ca.unsqueeze(2) - pred_ca.unsqueeze(1), dim=-1)
    L = ca_mask.shape[1]
    idx = torch.arange(L, device=pred_ca.device)
    seq_far = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() > seq_sep
    same = chain_id.unsqueeze(2) == chain_id.unsqueeze(1)
    pair = ca_mask.unsqueeze(2) & ca_mask.unsqueeze(1) & seq_far.unsqueeze(0) & same
    m = pair.to(d.dtype)
    viol = torch.relu(min_d - d).pow(2)
    return (viol * m).sum() / m.sum().clamp(min=1)
