"""Torch geometry utilities used by training and sampling."""

import torch
from torch import Tensor


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
