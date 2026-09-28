"""Differentiable exact all-atom lDDT loss.

The public parameters use Angstrom units. Coordinates inside MambaFold are
normalized by ``COORD_SCALE``; converting the distance error back to Angstroms
keeps this surrogate numerically identical to SimpleFold's implementation.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from mambafold.data.constants import COORD_SCALE

_LDDT_THRESHOLDS_A = (0.5, 1.0, 2.0, 4.0)
_Reduction = Literal["mean", "none"]


def _reduce_losses(losses: Tensor, reduction: _Reduction) -> Tensor:
    if reduction == "none":
        return losses
    if reduction == "mean":
        return losses.mean()
    raise ValueError(f"unsupported reduction: {reduction!r}")


def _soft_pair_score(
    distance_error_normalized: Tensor,
    thresholds_A: tuple[float, ...],
) -> Tensor:
    """Return SimpleFold's smooth lDDT score for normalized distance errors."""
    distance_error_A = distance_error_normalized * COORD_SCALE
    return sum(torch.sigmoid(threshold_A - distance_error_A) for threshold_A in thresholds_A) / len(
        thresholds_A
    )


def _select_valid_atoms(
    pred_coords: Tensor,
    true_coords: Tensor,
    valid_mask: Tensor,
    batch_index: int,
) -> tuple[Tensor, Tensor]:
    flat_mask = valid_mask[batch_index].reshape(-1)
    atom_index = flat_mask.nonzero(as_tuple=False).squeeze(-1)
    pred = pred_coords[batch_index].reshape(-1, 3).index_select(0, atom_index)
    true = true_coords[batch_index].reshape(-1, 3).index_select(0, atom_index)
    return pred, true


def _exact_soft_lddt_single(
    pred: Tensor,
    true: Tensor,
    *,
    cutoff_A: float,
    thresholds_A: tuple[float, ...],
    pair_chunk_size: int,
) -> Tensor:
    """Exact all-pair soft lDDT without a dense predicted pair matrix.

    Ground-truth distances are scanned in row chunks to find every unordered
    pair inside the cutoff. Predicted distances are then evaluated only for
    those pairs. Counting each symmetric pair once leaves the lDDT ratio
    unchanged while halving the differentiable work.
    """
    num_atoms = pred.shape[0]
    if num_atoms < 2:
        return pred.sum() * 0.0 + 1.0
    if pair_chunk_size <= 0:
        raise ValueError("pair_chunk_size must be positive")

    cutoff_normalized = cutoff_A / COORD_SCALE
    column_index = torch.arange(num_atoms, device=true.device).unsqueeze(0)
    score_sum = pred.sum() * 0.0
    pair_count = 0

    def score_pairs(
        all_pred: Tensor,
        pair_row: Tensor,
        pair_column: Tensor,
        true_pair_distance: Tensor,
    ) -> Tensor:
        pred_pair_distance = torch.linalg.vector_norm(
            all_pred.index_select(0, pair_row) - all_pred.index_select(0, pair_column),
            dim=-1,
        )
        score = _soft_pair_score(
            (pred_pair_distance - true_pair_distance).abs(),
            thresholds_A,
        )
        return score.sum()

    for start in range(0, num_atoms, pair_chunk_size):
        stop = min(start + pair_chunk_size, num_atoms)
        with torch.no_grad():
            true_distance = torch.cdist(true[start:stop], true)
            row_index = torch.arange(start, stop, device=true.device).unsqueeze(1)
            # Upper triangle only: self-pairs are excluded and symmetric pairs
            # have identical score, so counting them once is exactly equivalent.
            pair_mask = (true_distance < cutoff_normalized) & (column_index > row_index)
            local_row, pair_column = pair_mask.nonzero(as_tuple=True)
            true_pair_distance = true_distance[local_row, pair_column]

        if local_row.numel() == 0:
            continue
        pair_row = local_row + start
        if torch.is_grad_enabled() and pred.requires_grad:
            block_score = checkpoint(
                score_pairs,
                pred,
                pair_row,
                pair_column,
                true_pair_distance,
                use_reentrant=False,
            )
        else:
            block_score = score_pairs(pred, pair_row, pair_column, true_pair_distance)
        score_sum = score_sum + block_score
        pair_count += pair_row.numel()

    return 1.0 - score_sum / max(pair_count, 1)


def soft_lddt_all_atom_loss(
    pred_coords: Tensor,
    true_coords: Tensor,
    valid_mask: Tensor,
    cutoff_A: float = 15.0,
    thresholds_A: tuple[float, ...] = _LDDT_THRESHOLDS_A,
    pair_chunk_size: int = 512,
    *,
    reduction: _Reduction = "mean",
) -> Tensor:
    """Exact differentiable all-atom lDDT loss.

    Every resolved atom is used. ``pair_chunk_size`` bounds the dense
    ground-truth distance workspace, but exact cutoff-neighbor discovery
    remains O(N^2) in the number of atoms.
    """
    if pred_coords.shape != true_coords.shape:
        raise ValueError(
            "pred_coords and true_coords must have identical shapes, got "
            f"{tuple(pred_coords.shape)} and {tuple(true_coords.shape)}"
        )
    if valid_mask.shape != pred_coords.shape[:-1]:
        raise ValueError(
            "valid_mask must match coordinate leading dimensions, got "
            f"{tuple(valid_mask.shape)} and {tuple(pred_coords.shape[:-1])}"
        )
    if cutoff_A <= 0:
        raise ValueError("cutoff_A must be positive")
    if not thresholds_A:
        raise ValueError("thresholds_A must not be empty")
    if pair_chunk_size <= 0:
        raise ValueError("pair_chunk_size must be positive")

    losses = []
    for batch_index in range(pred_coords.shape[0]):
        pred, true = _select_valid_atoms(
            pred_coords,
            true_coords,
            valid_mask,
            batch_index,
        )
        losses.append(
            _exact_soft_lddt_single(
                pred,
                true,
                cutoff_A=cutoff_A,
                thresholds_A=thresholds_A,
                pair_chunk_size=pair_chunk_size,
            )
        )

    if not losses:
        if reduction == "none":
            return pred_coords.new_empty((0,))
        return pred_coords.sum() * 0.0
    return _reduce_losses(torch.stack(losses), reduction)
