"""Training and evaluation step functions."""

import torch
import torch.nn.functional as F
from torch import Tensor

from mambafold.data.types import ProteinBatch
from mambafold.losses.eqm import eqm_loss, eqm_reconstruction_scale
from mambafold.losses.lddt import soft_lddt_ca_loss


def train_step(
    model,
    batch: ProteinBatch,
    optimizer,
    grad_clip: float = 1.0,
    alpha_mode: str = "const",
    eqm_a: float = 0.8,
    eqm_lam: float = 4.0,
    lddt_cutoff: float = 1.5,
    use_amp: bool = True,
) -> dict:
    """Single training step.

    Args:
        model: MambaFoldEqM
        batch: ProteinBatch (already on device)
        optimizer: AdamW optimizer
        grad_clip: max gradient norm
        alpha_mode: "const" (pretrain) or "ramp" (finetune)
        eqm_a, eqm_lam: EqM c(γ) parameters
        lddt_cutoff: distance cutoff for LDDT (normalized units, default 1.5 = 15Å)
        use_amp: enable bfloat16 autocast (requires CUDA)

    Returns: dict of loss values
    """
    model.train()
    amp_enabled = use_amp and batch.device.type == "cuda"

    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        pred = model(batch)  # [B, L, A, 3]

        # Structure reconstruction for LDDT (computed inside autocast)
        scale = eqm_reconstruction_scale(batch.gamma, a=eqm_a, lam=eqm_lam)
        x_hat = batch.x_gamma - scale * pred

    # Cast to fp32 for numerically stable loss computation
    pred_f32 = pred.float()
    x_hat_f32 = x_hat.float()

    # EqM loss in fp32
    loss_eqm = eqm_loss(
        pred_f32, batch.x_clean.float(), batch.eps.float(), batch.gamma,
        batch.valid_mask, a=eqm_a, lam=eqm_lam,
    )

    # CA-LDDT auxiliary loss in fp32
    loss_lddt = soft_lddt_ca_loss(x_hat_f32, batch.x_clean.float(), batch.ca_mask, cutoff=lddt_cutoff)

    # Alpha weighting: ramp up LDDT weight near clean structures during finetuning
    if alpha_mode == "const":
        alpha = 1.0
    else:
        alpha = (1.0 + 8.0 * F.relu(batch.gamma - 0.5)).mean().item()

    loss = loss_eqm + alpha * loss_lddt

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
    optimizer.step()

    return {
        "loss": loss.item(),
        "eqm": loss_eqm.item(),
        "lddt": loss_lddt.item(),
        "alpha": alpha,
        "grad_norm": grad_norm,
        "gamma_mean": batch.gamma.mean().item(),
    }


@torch.no_grad()
def eval_step(
    model,
    batch: ProteinBatch,
    eqm_a: float = 0.8,
    eqm_lam: float = 4.0,
    lddt_cutoff: float = 1.5,
    use_amp: bool = True,
) -> dict:
    """Single evaluation step (no gradient).

    Returns: dict of loss values
    """
    model.eval()
    amp_enabled = use_amp and batch.device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        pred = model(batch)

        scale = eqm_reconstruction_scale(batch.gamma, a=eqm_a, lam=eqm_lam)
        x_hat = batch.x_gamma - scale * pred

        n_valid = batch.valid_mask.sum().clamp(min=1)
        grad_rms = (pred.pow(2).sum() / n_valid / 3).sqrt()

    # Cast to fp32 for numerically stable loss computation
    pred_f32 = pred.float()
    x_hat_f32 = x_hat.float()

    loss_eqm = eqm_loss(
        pred_f32, batch.x_clean.float(), batch.eps.float(), batch.gamma,
        batch.valid_mask, a=eqm_a, lam=eqm_lam,
    )
    loss_lddt = soft_lddt_ca_loss(x_hat_f32, batch.x_clean.float(), batch.ca_mask, cutoff=lddt_cutoff)

    return {
        "eqm": loss_eqm.item(),
        "lddt": loss_lddt.item(),
        "grad_rms": grad_rms.item(),
    }
