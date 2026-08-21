"""Geometry utilities for protein coordinate operations."""

import numpy as np
import torch
from torch import Tensor

# ── NumPy utilities (for inference / visualization) ──────────────────────────

def kabsch_align(mobile: np.ndarray, ref: np.ndarray):
    """Kabsch-align mobile onto ref. Both [N, 3].

    Returns:
        aligned: [N, 3] aligned mobile coordinates
        R: [3, 3] rotation matrix
        rmsd: float Kabsch RMSD
    """
    t_ref = ref.mean(0)
    t_mob = mobile.mean(0)
    r = ref - t_ref
    m = mobile - t_mob
    H = m.T @ r
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    aligned = (m @ R.T) + t_ref
    rmsd = float(np.sqrt(((aligned - ref) ** 2).sum(-1).mean()))
    return aligned, R, rmsd


def kabsch_rmsd(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    """Kabsch-aligned RMSD for masked Cα coordinates."""
    _, _, rmsd = kabsch_align(pred[mask], true[mask])
    return rmsd


def random_rotation_matrix(device: torch.device = None) -> Tensor:
    """Sample a random SO(3) rotation matrix via QR decomposition.

    Returns: [3, 3] rotation matrix
    """
    m = torch.randn(3, 3, device=device)
    q, r = torch.linalg.qr(m)
    # Ensure det(Q) = +1 (proper rotation)
    q = q * torch.sign(torch.diag(r))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def masked_centroid(coords: Tensor, mask: Tensor) -> Tensor:
    """Compute centroid of masked coordinates.

    Args:
        coords: [*, N, 3] coordinates
        mask: [*, N] bool mask

    Returns: [*, 1, 3] centroid
    """
    mask_f = mask.unsqueeze(-1).to(coords.dtype)         # [*, N, 1]
    total = mask_f.sum(dim=-2, keepdim=True).clamp(min=1)  # [*, 1, 1]
    return (coords * mask_f).sum(dim=-2, keepdim=True) / total  # [*, 1, 3]


def remove_translation(coords: Tensor, mask: Tensor) -> Tensor:
    """Center coordinates by removing masked centroid.

    Args:
        coords: [B, L, A, 3]
        mask: [B, L, A] bool

    Returns: [B, L, A, 3] centered
    """
    flat_coords = coords.reshape(coords.shape[0], -1, 3)  # [B, L*A, 3]
    flat_mask = mask.reshape(mask.shape[0], -1)            # [B, L*A]
    centroid = masked_centroid(flat_coords, flat_mask)     # [B, 1, 3]
    return coords - centroid.unsqueeze(1)                  # broadcast [B, 1, 1, 3]


def weighted_rigid_align(
    true_coords: Tensor,
    pred_coords: Tensor,
    mask: Tensor,
) -> Tensor:
    """Rigidly align true coordinates onto predicted coordinates.

    This is the masked Kabsch target alignment used by SimpleFold before its
    flow-matching MSE. Inputs may carry arbitrary point dimensions between the
    batch and xyz axes (for example ``[B, L, A, 3]``); they are flattened only
    for estimating the transform and returned in their original shape.
    """
    if true_coords.shape != pred_coords.shape:
        raise ValueError("true_coords and pred_coords must have identical shapes")
    if true_coords.ndim < 3 or true_coords.shape[-1] != 3:
        raise ValueError("coordinates must have shape [B, ..., 3]")
    if mask.shape != true_coords.shape[:-1]:
        raise ValueError("mask must match coordinate leading dimensions")

    batch_size = true_coords.shape[0]
    original_shape = true_coords.shape
    true_flat = true_coords.reshape(batch_size, -1, 3)
    pred_flat = pred_coords.reshape(batch_size, -1, 3)
    weights = mask.reshape(batch_size, -1, 1).to(true_coords.dtype)
    denominator = weights.sum(dim=1, keepdim=True).clamp(min=1)

    true_centroid = (true_flat * weights).sum(dim=1, keepdim=True) / denominator
    pred_centroid = (pred_flat * weights).sum(dim=1, keepdim=True) / denominator
    true_centered = true_flat - true_centroid
    pred_centered = pred_flat - pred_centroid

    covariance = torch.einsum(
        "bni,bnj->bij",
        weights * pred_centered,
        true_centered,
    )
    original_dtype = covariance.dtype
    covariance = covariance.float()
    u, _, vh = torch.linalg.svd(
        covariance,
        driver="gesvd" if covariance.is_cuda else None,
    )
    v = vh.mH

    # U @ diag(1, 1, det(U @ V^T)) @ V^T prevents reflections.
    provisional = u @ v.mT
    correction = torch.eye(3, dtype=covariance.dtype, device=covariance.device)
    correction = correction.unsqueeze(0).repeat(batch_size, 1, 1)
    correction[:, -1, -1] = torch.det(provisional)
    rotation = (u @ correction @ v.mT).to(original_dtype)

    aligned = true_centered @ rotation.mT + pred_centroid
    return aligned.reshape(original_shape)


def apply_rotation(coords: Tensor, rot: Tensor) -> Tensor:
    """Apply rotation matrix to coordinates.

    Args:
        coords: [*, 3]
        rot: [3, 3]

    Returns: [*, 3]
    """
    return coords @ rot.T
