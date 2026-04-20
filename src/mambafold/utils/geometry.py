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


def apply_rotation(coords: Tensor, rot: Tensor) -> Tensor:
    """Apply rotation matrix to coordinates.

    Args:
        coords: [*, 3]
        rot: [3, 3]

    Returns: [*, 3]
    """
    return coords @ rot.T


