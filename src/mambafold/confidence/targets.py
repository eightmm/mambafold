"""Targets, loss, and score conversion for standalone pLDDT training."""

from __future__ import annotations

import math

import torch
from torch import Tensor

LDDT_CA_CUTOFF_ANGSTROM = 15.0
LDDT_CA_THRESHOLDS_ANGSTROM = (0.5, 1.0, 2.0, 4.0)
DEFAULT_N_BINS = 50


def hard_lddt_ca(
    pred_ca_angstrom: Tensor,
    true_ca_angstrom: Tensor,
    ca_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute exact per-residue hard lDDT-Cα in Angstrom units.

    A reference neighbor contributes when its true Cα distance is strictly
    below 15 Å.  Its absolute distance error is evaluated at the four strict
    thresholds ``<0.5``, ``<1``, ``<2``, and ``<4`` Å.  A residue has a valid
    label only when at least one non-self reference neighbor is available.

    Returns:
        ``(scores, label_mask)`` with shape ``[B, L]``.  Scores are in ``[0, 1]``
        and are zero where ``label_mask`` is false.
    """

    _validate_ca_inputs(pred_ca_angstrom, true_ca_angstrom, ca_mask)
    pred_dist = torch.linalg.vector_norm(
        pred_ca_angstrom.unsqueeze(2) - pred_ca_angstrom.unsqueeze(1), dim=-1
    )
    true_dist = torch.linalg.vector_norm(
        true_ca_angstrom.unsqueeze(2) - true_ca_angstrom.unsqueeze(1), dim=-1
    )

    length = ca_mask.shape[1]
    nonself = ~torch.eye(length, dtype=torch.bool, device=ca_mask.device).unsqueeze(0)
    pair_mask = (
        ca_mask.unsqueeze(2)
        & ca_mask.unsqueeze(1)
        & nonself
        & (true_dist < LDDT_CA_CUTOFF_ANGSTROM)
    )
    error = (pred_dist - true_dist).abs()
    passed = torch.stack(
        [error < threshold for threshold in LDDT_CA_THRESHOLDS_ANGSTROM], dim=-1
    )
    pair_score = passed.to(pred_ca_angstrom.dtype).mean(dim=-1)
    pair_weight = pair_mask.to(pred_ca_angstrom.dtype)
    neighbor_count = pair_weight.sum(dim=-1)
    label_mask = neighbor_count > 0
    scores = (pair_score * pair_weight).sum(dim=-1) / neighbor_count.clamp_min(1)
    return scores.masked_fill(~label_mask, 0.0), label_mask


# Descriptive alias used by callers that group several per-residue targets.
per_residue_lddt_ca = hard_lddt_ca


def lddt_bin_centers(
    n_bins: int = DEFAULT_N_BINS,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return equal-width lDDT bin centers on ``[0, 1]``.

    For the default 50 bins, centers are 0.01, 0.03, ..., 0.99.
    """

    _validate_n_bins(n_bins)
    if dtype is not None and not dtype.is_floating_point:
        raise TypeError(f"dtype must be floating, got {dtype}")
    dtype = torch.get_default_dtype() if dtype is None else dtype
    return (torch.arange(n_bins, device=device, dtype=dtype) + 0.5) / n_bins


def soft_adjacent_bin_labels(
    scores: Tensor,
    label_mask: Tensor,
    n_bins: int = DEFAULT_N_BINS,
) -> Tensor:
    """Linearly interpolate normalized lDDT scores across adjacent bin centers.

    Scores outside ``[0, 1]`` are rejected.  Values below the first center or
    above the last center are assigned fully to the edge bin.  Invalid residues
    receive an all-zero target distribution.
    """

    _validate_scores_and_mask(scores, label_mask)
    _validate_n_bins(n_bins)
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if bool(((scores < 0) | (scores > 1)).any()):
        raise ValueError("scores must lie in [0, 1]")

    position = (scores * n_bins - 0.5).clamp(0, n_bins - 1)
    lower = position.floor().to(torch.long)
    upper = position.ceil().to(torch.long)
    upper_weight = position - lower.to(position.dtype)
    lower_weight = 1.0 - upper_weight

    labels = scores.new_zeros((*scores.shape, n_bins))
    labels.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    labels.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return labels * label_mask.unsqueeze(-1).to(labels.dtype)


def macro_per_protein_cross_entropy(
    logits: Tensor,
    soft_labels: Tensor,
    label_mask: Tensor,
) -> Tensor:
    """Return soft-label CE averaged per protein and then across valid proteins.

    This macro reduction prevents long proteins from dominating the confidence
    objective.  Proteins with no valid lDDT label are excluded.  If the whole
    batch has no valid label, a differentiable scalar zero is returned.
    """

    _validate_loss_inputs(logits, soft_labels, label_mask)
    log_prob = torch.log_softmax(logits, dim=-1)
    residue_loss = -(soft_labels.to(log_prob.dtype) * log_prob).sum(dim=-1)
    weights = label_mask.to(residue_loss.dtype)
    valid_count = weights.sum(dim=-1)
    per_protein = (residue_loss * weights).sum(dim=-1) / valid_count.clamp_min(1)
    valid_protein = valid_count > 0
    if not bool(valid_protein.any()):
        return logits.sum() * 0.0
    return per_protein[valid_protein].mean()


def expected_lddt(logits: Tensor, *, temperature: float = 1.0) -> Tensor:
    """Convert bin logits to the expected normalized lDDT score in ``[0, 1]``."""

    if not isinstance(logits, Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if logits.ndim < 1 or logits.shape[-1] < 2:
        raise ValueError(f"logits must end in at least two bins, got {tuple(logits.shape)}")
    if not logits.is_floating_point():
        raise TypeError(f"logits must have a floating dtype, got {logits.dtype}")
    temperature = _validate_temperature(temperature)
    centers = lddt_bin_centers(
        logits.shape[-1],
        device=logits.device,
        dtype=logits.dtype,
    )
    probability = torch.softmax(logits / temperature, dim=-1)
    return (probability * centers).sum(dim=-1)


def expected_plddt(logits: Tensor, *, temperature: float = 1.0) -> Tensor:
    """Convert bin logits to conventional pLDDT units on ``[0, 100]``."""

    return expected_lddt(logits, temperature=temperature) * 100.0


def _validate_ca_inputs(pred_ca: Tensor, true_ca: Tensor, ca_mask: Tensor) -> None:
    for name, value in (("pred_ca_angstrom", pred_ca), ("true_ca_angstrom", true_ca)):
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError(f"{name} must have shape [B, L, 3], got {tuple(value.shape)}")
        if not value.is_floating_point():
            raise TypeError(f"{name} must have a floating dtype, got {value.dtype}")
    if pred_ca.shape != true_ca.shape:
        raise ValueError(f"predicted/true CA shapes differ: {pred_ca.shape} vs {true_ca.shape}")
    if not isinstance(ca_mask, Tensor):
        raise TypeError("ca_mask must be a torch.Tensor")
    if ca_mask.shape != pred_ca.shape[:2]:
        raise ValueError(f"ca_mask must have shape {pred_ca.shape[:2]}, got {ca_mask.shape}")
    if ca_mask.dtype is not torch.bool:
        raise TypeError(f"ca_mask must have dtype torch.bool, got {ca_mask.dtype}")
    if pred_ca.device != true_ca.device or pred_ca.device != ca_mask.device:
        raise ValueError("predicted CA, true CA, and ca_mask must share a device")
    if pred_ca.dtype != true_ca.dtype:
        raise TypeError(f"predicted/true CA dtypes differ: {pred_ca.dtype} vs {true_ca.dtype}")
    valid_xyz = ca_mask.unsqueeze(-1).expand_as(pred_ca)
    if not bool(torch.isfinite(pred_ca[valid_xyz]).all()):
        raise ValueError("predicted CA coordinates must be finite at valid mask positions")
    if not bool(torch.isfinite(true_ca[valid_xyz]).all()):
        raise ValueError("true CA coordinates must be finite at valid mask positions")


def _validate_scores_and_mask(scores: Tensor, label_mask: Tensor) -> None:
    if not isinstance(scores, Tensor) or not isinstance(label_mask, Tensor):
        raise TypeError("scores and label_mask must be torch.Tensor instances")
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape [B, L], got {tuple(scores.shape)}")
    if not scores.is_floating_point():
        raise TypeError(f"scores must have a floating dtype, got {scores.dtype}")
    if label_mask.shape != scores.shape or label_mask.dtype is not torch.bool:
        raise ValueError("label_mask must be bool and have the same [B, L] shape as scores")
    if scores.device != label_mask.device:
        raise ValueError("scores and label_mask must share a device")


def _validate_loss_inputs(logits: Tensor, soft_labels: Tensor, label_mask: Tensor) -> None:
    if not isinstance(logits, Tensor) or not isinstance(soft_labels, Tensor):
        raise TypeError("logits and soft_labels must be torch.Tensor instances")
    if logits.ndim != 3 or logits.shape[-1] < 2:
        raise ValueError(f"logits must have shape [B, L, K>=2], got {tuple(logits.shape)}")
    if soft_labels.shape != logits.shape:
        raise ValueError(f"soft_labels shape {soft_labels.shape} does not match {logits.shape}")
    if label_mask.shape != logits.shape[:2] or label_mask.dtype is not torch.bool:
        raise ValueError("label_mask must be bool and match logits [B, L]")
    if not logits.is_floating_point() or not soft_labels.is_floating_point():
        raise TypeError("logits and soft_labels must have floating dtypes")
    if logits.device != soft_labels.device or logits.device != label_mask.device:
        raise ValueError("logits, soft_labels, and label_mask must share a device")
    if not torch.isfinite(soft_labels).all():
        raise ValueError("soft_labels must be finite")
    if bool((soft_labels < 0).any()):
        raise ValueError("soft_labels must be non-negative")
    mass = soft_labels.sum(dim=-1)
    expected_mass = label_mask.to(mass.dtype)
    if not torch.allclose(mass, expected_mass, atol=1e-5, rtol=1e-5):
        raise ValueError("soft_labels must sum to one on valid residues and zero elsewhere")


def _validate_n_bins(n_bins: int) -> None:
    if type(n_bins) is not int or n_bins < 2:
        raise ValueError(f"n_bins must be an integer >= 2, got {n_bins!r}")


def _validate_temperature(temperature: float) -> float:
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError(f"temperature must be a positive finite number, got {temperature!r}")
    value = float(temperature)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"temperature must be a positive finite number, got {temperature!r}")
    return value
