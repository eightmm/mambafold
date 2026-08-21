"""Direct all-atom flow-matching training and loss evaluation.

Flow matching convention::

    x_t = t * x_clean + (1 - t) * eps
    velocity target = x_clean - eps
    one-step reconstruction = x_t + (1 - t) * v

The folding baseline deliberately has only two objectives: aligned all-atom
flow matching and exact all-atom soft lDDT. Confidence is trained separately
from a frozen folding model.
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import Tensor

from mambafold.data.constants import CA_ATOM_ID
from mambafold.data.types import ProteinBatch
from mambafold.losses.lddt import soft_lddt_all_atom_loss
from mambafold.utils.geometry import weighted_rigid_align


def _alpha(t: Tensor, mode: str) -> Tensor:
    flat_t = t.reshape(t.shape[0], -1)[:, 0]
    if mode == "const":
        return torch.ones_like(flat_t)
    if mode == "ramp":
        return 1.0 + 8.0 * F.relu(flat_t - 0.5)
    raise ValueError(f"unknown alpha_mode: {mode!r}")


def _recon_atom(x_t: Tensor, t: Tensor, v_atom: Tensor) -> Tensor:
    one_minus_t = (1.0 - t.squeeze(-1).squeeze(-1)).view(-1, 1, 1, 1)
    return x_t + one_minus_t * v_atom


def _masked_mse_per_example(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Return xyz MSE per protein, giving each protein equal batch weight."""
    diff_sq = (pred - target).pow(2)
    mask_f = mask.unsqueeze(-1).to(diff_sq.dtype)
    reduce_dims = tuple(range(1, diff_sq.ndim))
    numerator = (diff_sq * mask_f).sum(dim=reduce_dims)
    atom_count = mask_f.sum(dim=tuple(range(1, mask_f.ndim)))
    denominator = (atom_count * diff_sq.shape[-1]).clamp(min=1)
    return numerator / denominator


def allatom_loss_surface(
    out: dict[str, Tensor],
    batch: ProteinBatch,
    *,
    alpha_mode: str = "const",
    use_rigid_align: bool = True,
    w_fm: float = 1.0,
    w_lddt_atom: float = 1.0,
    lddt_cutoff_A: float = 15.0,
    lddt_pair_chunk_size: int = 512,
) -> tuple[Tensor, dict[str, float]]:
    """Compute aligned all-atom FM plus exact all-atom soft lDDT."""
    v_atom = out["v_atom"].float()
    x_clean = batch.x_clean.float()
    x_t = batch.x_t.float()
    eps = batch.eps.float()
    x_hat = _recon_atom(x_t, batch.t, v_atom)

    fm_clean = x_clean
    if use_rigid_align:
        with torch.no_grad():
            fm_clean = weighted_rigid_align(
                x_clean.detach(),
                x_hat.detach(),
                batch.valid_mask,
            )

    fm_target = fm_clean - eps
    fm_per_example = _masked_mse_per_example(v_atom, fm_target, batch.valid_mask)
    loss_fm = fm_per_example.mean()

    alpha_per_example = _alpha(batch.t, alpha_mode)
    if w_lddt_atom:
        lddt_per_example = soft_lddt_all_atom_loss(
            x_hat,
            x_clean,
            batch.valid_mask,
            cutoff_A=lddt_cutoff_A,
            pair_chunk_size=lddt_pair_chunk_size,
            reduction="none",
        )
    else:
        lddt_per_example = x_hat.reshape(x_hat.shape[0], -1).sum(dim=-1) * 0.0
    loss_lddt = lddt_per_example.mean()
    loss_lddt_weighted = (alpha_per_example * lddt_per_example).mean()

    total = w_fm * loss_fm + w_lddt_atom * loss_lddt_weighted

    ca_fm = _masked_mse_per_example(
        v_atom[..., CA_ATOM_ID, :],
        fm_target[..., CA_ATOM_ID, :],
        batch.ca_mask,
    ).mean()
    metrics = {
        "fm_atom": loss_fm.item(),
        "ca_fm": ca_fm.item(),
        "lddt_atom": loss_lddt.item(),
        "lddt_atom_weighted": loss_lddt_weighted.item(),
        "alpha": alpha_per_example.mean().item(),
    }
    return total, metrics


def allatom_forward_and_loss(
    model,
    batch: ProteinBatch,
    *,
    alpha_mode: str = "const",
    use_rigid_align: bool = True,
    use_amp: bool = True,
    w_fm: float = 1.0,
    w_lddt_atom: float = 1.0,
    lddt_cutoff_A: float = 15.0,
    lddt_pair_chunk_size: int = 512,
    self_condition_prob: float = 0.0,
) -> tuple[Tensor, dict[str, float]]:
    """Run the folding model and compute its complete training objective."""
    model.train()
    amp_enabled = use_amp and batch.device.type == "cuda"
    raw_model = getattr(model, "module", model)
    used_self_cond = False

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        if self_condition_prob > 0.0 and getattr(raw_model, "self_conditioning", False):
            probability = max(0.0, min(1.0, float(self_condition_prob)))
            if torch.rand((), device=batch.device).item() < probability:
                with torch.no_grad():
                    sc_out = model(batch)
                    x_self_cond = _recon_atom(
                        batch.x_t.float(),
                        batch.t,
                        sc_out["v_atom"].float(),
                    ).detach()
                batch = replace(batch, x_self_cond=x_self_cond)
                used_self_cond = True
        out = model(batch)

    loss, metrics = allatom_loss_surface(
        out,
        batch,
        alpha_mode=alpha_mode,
        use_rigid_align=use_rigid_align,
        w_fm=w_fm,
        w_lddt_atom=w_lddt_atom,
        lddt_cutoff_A=lddt_cutoff_A,
        lddt_pair_chunk_size=lddt_pair_chunk_size,
    )
    metrics["loss"] = loss.item()
    metrics["t_mean"] = batch.t.mean().item()
    metrics["self_cond"] = float(used_self_cond)
    return loss, metrics
