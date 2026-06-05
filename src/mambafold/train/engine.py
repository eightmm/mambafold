"""Training step functions (CA-only Stage 1, all-atom Stage 2, joint).

Single-chain training path.

Stage 1 (CA-only FM + aux):
    L = L_fm_ca + alpha(t)*L_lddt_ca + gamma*L_bond_caca + lambda*L_distogram

Stage 2 (all-atom FM + aux, CA residual-refined from Stage 1 anchor):
    L = L_fm_atom(non-CA) + alpha(t)*L_lddt_full + omega*L_bond + zeta*L_clash + eta*L_ca_anchor
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from mambafold.data.constants import CA_ATOM_ID
from mambafold.data.types import ProteinBatch
from mambafold.losses.ca_only import (
    ca_ca_bond_loss,
    distogram_loss_ca_only,
    soft_lddt_ca_only,
)
from mambafold.losses.geometry import (
    bond_length_loss,
    ca_clash_loss,
)
from mambafold.losses.lddt import soft_lddt_ca_loss


def _fm_loss_ca(v_pred: Tensor, x_clean: Tensor, eps: Tensor, mask: Tensor) -> Tensor:
    """Masked MSE for FM target (x_clean − eps) on CA positions."""
    target = x_clean - eps                            # [B, L, 3]
    diff_sq = (v_pred - target).pow(2).sum(dim=-1)    # [B, L]
    m = mask.to(diff_sq.dtype)
    return (diff_sq * m).sum() / m.sum().clamp(min=1)


def stage1_forward_and_loss(
    model,
    batch: ProteinBatch,
    *,
    alpha_mode: str = "ramp",
    use_amp: bool = True,
    w_lddt_ca: float = 1.0,
    w_bond_caca: float = 0.1,
    w_distogram: float = 0.5,
):
    """Stage 1 forward + composite loss + per-component metrics.

    Notes:
        - Distogram aux is always computed (return_aux=True) so the pair stack
          gets gradient on the binning signal from step 1.
        - The FM main loss is on raw velocity `v_ca`; reconstruction
          `x_hat_ca = x_t_ca + (1-t) · v_ca` is used for lDDT-type metrics.
    """
    model.train()
    amp_enabled = use_amp and batch.device.type == "cuda"

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        out = model(batch, return_aux=True)
        v_ca = out["v_ca"]
        dist_logits = out["distogram_logits"]

    v_ca_f32 = v_ca.float()

    x_clean_ca = batch.x_clean[..., CA_ATOM_ID, :].float()                # [B, L, 3]
    eps_ca = batch.eps[..., CA_ATOM_ID, :].float()
    x_t_ca = batch.x_t[..., CA_ATOM_ID, :].float()

    # 1) FM main loss on CA velocity
    loss_fm = _fm_loss_ca(v_ca_f32, x_clean_ca, eps_ca, batch.ca_mask)

    # 2) One-step recon for lDDT-style metrics
    one_minus_t = (1.0 - batch.t.squeeze(-1).squeeze(-1).squeeze(-1)).float()  # [B]
    x_hat_ca = x_t_ca + one_minus_t.view(-1, 1, 1) * v_ca_f32                  # [B, L, 3]

    # 3) lDDT aux losses (zero if no valid pairs)
    loss_lddt = soft_lddt_ca_only(x_hat_ca, x_clean_ca, batch.ca_mask)
    loss_bond = ca_ca_bond_loss(x_hat_ca, batch.ca_mask, batch.chain_id)
    loss_dist = distogram_loss_ca_only(
        dist_logits.float(), x_clean_ca, batch.ca_mask,
        n_bins=model.n_distogram_bins if hasattr(model, "n_distogram_bins") else 64,
    )

    # 4) Alpha schedule for lDDT (shared `ramp` policy)
    if alpha_mode == "const":
        alpha = 1.0
    else:
        alpha = (1.0 + 8.0 * F.relu(batch.t - 0.5)).mean().item()

    loss = (
        loss_fm
        + alpha * w_lddt_ca * loss_lddt
        + w_bond_caca * loss_bond
        + w_distogram * loss_dist
    )

    metrics = {
        "loss":       loss.item(),
        "fm":         loss_fm.item(),
        "lddt":       loss_lddt.item(),
        "bond_caca":  loss_bond.item(),
        "distogram":  loss_dist.item(),
        "alpha":      alpha,
        "t_mean":     batch.t.mean().item(),
    }
    return loss, metrics


# ── Stage 2 (TwoStageMambaFold with frozen S1) ────────────────────────────


def _non_ca_atom_mask(atom_mask: Tensor) -> Tensor:
    """Per-atom mask that excludes the CA slot (Stage 1 owns CA)."""
    A = atom_mask.shape[-1]
    not_ca = torch.arange(A, device=atom_mask.device) != CA_ATOM_ID
    return atom_mask & not_ca.view(1, 1, A)


def _fm_loss_atom(
    v_pred: Tensor, x_clean: Tensor, eps: Tensor, mask: Tensor,
) -> Tensor:
    """Masked MSE for FM target on per-atom velocity. mask: [B, L, A]."""
    target = x_clean - eps                                  # [B, L, A, 3]
    diff_sq = (v_pred - target).pow(2).sum(dim=-1)          # [B, L, A]
    m = mask.to(diff_sq.dtype)
    return (diff_sq * m).sum() / m.sum().clamp(min=1)


def _x_hat_atom(
    x_t: Tensor, t: Tensor, v_atom: Tensor,
) -> Tensor:
    """FM reconstruction for all atom slots, including a residual CA update."""
    one_minus_t = (1.0 - t.squeeze(-1).squeeze(-1)).view(-1, 1, 1, 1)  # [B,1,1,1]
    return x_t + one_minus_t * v_atom


def _inject_ca(x_t: Tensor, ca: Tensor) -> Tensor:
    out = x_t.clone()
    out[..., CA_ATOM_ID, :] = ca
    return out


def _ca_anchor_loss(x_hat: Tensor, s1_ca: Tensor, ca_mask: Tensor) -> Tensor:
    pred_ca = x_hat[..., CA_ATOM_ID, :]
    diff_sq = (pred_ca - s1_ca).pow(2).sum(dim=-1)
    m = ca_mask.to(diff_sq.dtype)
    return (diff_sq * m).sum() / m.sum().clamp(min=1)


def stage2_forward_and_loss(
    model,                                  # TwoStageMambaFold (freeze_stage1=True for Phase 2)
    batch: ProteinBatch,
    *,
    alpha_mode: str = "ramp",
    use_amp: bool = True,
    w_lddt_full: float = 1.0,
    w_bond: float = 0.05,
    w_clash: float = 0.01,
    w_ca_anchor: float = 2.0,
    ca_condition_noise_std: float = 0.0,
    ca_condition_noise_prob: float = 0.0,
    lddt_cutoff: float = 1.5,
):
    """Stage 2 forward (S1 frozen via TwoStage wrapper) + atom-side losses.

    Non-CA atoms receive the FM loss. CA may move through lDDT/geometry gradients,
    but `w_ca_anchor` keeps it close to Stage 1 to avoid fold drift.
    """
    model.train()
    amp_enabled = use_amp and batch.device.type == "cuda"

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        # return_aux=False — we don't need distogram_logits in Phase 2 (frozen S1).
        out = model(
            batch, return_aux=False,
            ca_condition_noise_std=ca_condition_noise_std,
            ca_condition_noise_prob=ca_condition_noise_prob,
        )
        v_atom = out["v_atom"]
        s1_ca = out["s1_ca"]
        s1_ca_cond = out.get("s1_ca_cond", s1_ca)

    v_atom_f32 = v_atom.float()
    s1_ca_f32 = s1_ca.float()
    s1_ca_cond_f32 = s1_ca_cond.float()

    # FM main loss on non-CA atoms.
    non_ca_mask = _non_ca_atom_mask(batch.atom_mask)               # [B, L, A]
    loss_fm = _fm_loss_atom(
        v_atom_f32, batch.x_clean.float(), batch.eps.float(),
        non_ca_mask & batch.valid_mask,
    )

    x_t_s2 = _inject_ca(batch.x_t.float(), s1_ca_cond_f32)
    x_hat = _x_hat_atom(x_t_s2, batch.t.float(), v_atom_f32)

    # Aux losses; CA residual is allowed but anchored to Stage 1.
    loss_lddt = soft_lddt_ca_loss(x_hat, batch.x_clean.float(),
                                  batch.ca_mask, cutoff=lddt_cutoff)
    loss_bond = bond_length_loss(x_hat, batch.res_type, batch.atom_mask, batch.res_mask)
    loss_clash = ca_clash_loss(x_hat, batch.res_mask, chain_id=batch.chain_id)
    loss_ca_anchor = _ca_anchor_loss(x_hat, s1_ca_f32, batch.ca_mask)
    # alpha schedule
    if alpha_mode == "const":
        alpha = 1.0
    else:
        alpha = (1.0 + 8.0 * F.relu(batch.t - 0.5)).mean().item()

    loss = (
        loss_fm
        + alpha * w_lddt_full * loss_lddt
        + w_bond * loss_bond
        + w_clash * loss_clash
        + w_ca_anchor * loss_ca_anchor
    )
    metrics = {
        "loss":        loss.item(),
        "fm_atom":     loss_fm.item(),
        "lddt_full":   loss_lddt.item(),
        "bond":        loss_bond.item(),
        "clash":       loss_clash.item(),
        "ca_anchor":   loss_ca_anchor.item(),
        "alpha":       alpha,
        "t_mean":      batch.t.mean().item(),
    }
    return loss, metrics


@torch.no_grad()
def stage2_eval_step(
    model,
    batch: ProteinBatch,
    lddt_cutoff: float = 1.5,
    use_amp: bool = True,
) -> dict:
    """No-grad eval step for Phase 2 / Phase 3 (TwoStage model)."""
    model.eval()
    amp_enabled = use_amp and batch.device.type == "cuda"

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        out = model(batch, return_aux=False)
        v_atom = out["v_atom"]
        s1_ca = out["s1_ca"]
        s1_ca_cond = out.get("s1_ca_cond", s1_ca)
        n_valid = batch.valid_mask.sum().clamp(min=1)
        v_rms = (v_atom.pow(2).sum() / n_valid / 3).sqrt()

    v_atom_f32 = v_atom.float()
    s1_ca_f32 = s1_ca.float()
    s1_ca_cond_f32 = s1_ca_cond.float()
    non_ca_mask = _non_ca_atom_mask(batch.atom_mask)
    loss_fm = _fm_loss_atom(
        v_atom_f32, batch.x_clean.float(), batch.eps.float(),
        non_ca_mask & batch.valid_mask,
    )
    x_t_s2 = _inject_ca(batch.x_t.float(), s1_ca_cond_f32)
    x_hat = _x_hat_atom(x_t_s2, batch.t.float(), v_atom_f32)
    loss_lddt = soft_lddt_ca_loss(x_hat, batch.x_clean.float(),
                                  batch.ca_mask, cutoff=lddt_cutoff)
    loss_bond = bond_length_loss(x_hat, batch.res_type, batch.atom_mask, batch.res_mask)
    loss_clash = ca_clash_loss(x_hat, batch.res_mask, chain_id=batch.chain_id)
    loss_ca_anchor = _ca_anchor_loss(x_hat, s1_ca_f32, batch.ca_mask)
    return {
        "fm_atom":     loss_fm.item(),
        "lddt_full":   loss_lddt.item(),
        "bond":        loss_bond.item(),
        "clash":       loss_clash.item(),
        "ca_anchor":   loss_ca_anchor.item(),
        "v_rms":       v_rms.item(),
    }


# ── joint (Phase 3) — both stages backprop ─────────────────────────────


def joint_forward_and_loss(
    model,
    batch: ProteinBatch,
    *,
    alpha_mode: str = "ramp",
    use_amp: bool = True,
    # Stage 1 weights
    w_lddt_ca: float = 1.0,
    w_bond_caca: float = 0.1,
    w_distogram: float = 0.5,
    # Stage 2 weights
    w_lddt_full: float = 1.0,
    w_bond: float = 0.05,
    w_clash: float = 0.01,
    w_ca_anchor: float = 2.0,
    ca_condition_noise_std: float = 0.0,
    ca_condition_noise_prob: float = 0.0,
    # Stage balance
    w_stage1: float = 1.0,
    lddt_cutoff: float = 1.5,
):
    """Joint Phase 3 loss — Stage 1 + residual-refining Stage 2."""
    model.train()
    amp_enabled = use_amp and batch.device.type == "cuda"

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        out = model(
            batch, return_aux=True,
            ca_condition_noise_std=ca_condition_noise_std,
            ca_condition_noise_prob=ca_condition_noise_prob,
        )
        v_ca = out["v_ca"]
        v_atom = out["v_atom"]
        s1_ca = out["s1_ca"]
        s1_ca_cond = out.get("s1_ca_cond", s1_ca)
        dist_logits = out["distogram_logits"]

    # Stage 1 loss surface (CA-only)
    v_ca_f32 = v_ca.float()
    x_clean_ca = batch.x_clean[..., CA_ATOM_ID, :].float()
    eps_ca = batch.eps[..., CA_ATOM_ID, :].float()
    x_t_ca = batch.x_t[..., CA_ATOM_ID, :].float()

    loss_s1_fm = _fm_loss_ca(v_ca_f32, x_clean_ca, eps_ca, batch.ca_mask)
    one_minus_t = (1.0 - batch.t.squeeze(-1).squeeze(-1).squeeze(-1)).float()
    x_hat_ca = x_t_ca + one_minus_t.view(-1, 1, 1) * v_ca_f32
    loss_s1_lddt = soft_lddt_ca_only(x_hat_ca, x_clean_ca, batch.ca_mask)
    loss_s1_bond = ca_ca_bond_loss(x_hat_ca, batch.ca_mask, batch.chain_id)
    loss_s1_dist = distogram_loss_ca_only(
        dist_logits.float(), x_clean_ca, batch.ca_mask,
    )

    # Stage 2 loss surface (all-atom, CA residual-refined from Stage 1 anchor)
    v_atom_f32 = v_atom.float()
    s1_ca_f32 = s1_ca.float()
    s1_ca_cond_f32 = s1_ca_cond.float()
    non_ca_mask = _non_ca_atom_mask(batch.atom_mask)
    loss_s2_fm = _fm_loss_atom(
        v_atom_f32, batch.x_clean.float(), batch.eps.float(),
        non_ca_mask & batch.valid_mask,
    )
    x_t_s2 = _inject_ca(batch.x_t.float(), s1_ca_cond_f32)
    x_hat_atom = _x_hat_atom(x_t_s2, batch.t.float(), v_atom_f32)
    loss_s2_lddt = soft_lddt_ca_loss(x_hat_atom, batch.x_clean.float(),
                                     batch.ca_mask, cutoff=lddt_cutoff)
    loss_s2_bond = bond_length_loss(x_hat_atom, batch.res_type,
                                    batch.atom_mask, batch.res_mask)
    loss_s2_clash = ca_clash_loss(x_hat_atom, batch.res_mask,
                                  chain_id=batch.chain_id)
    loss_s2_ca_anchor = _ca_anchor_loss(x_hat_atom, s1_ca_f32, batch.ca_mask)

    if alpha_mode == "const":
        alpha = 1.0
    else:
        alpha = (1.0 + 8.0 * F.relu(batch.t - 0.5)).mean().item()

    loss_s1_total = (
        loss_s1_fm
        + alpha * w_lddt_ca * loss_s1_lddt
        + w_bond_caca * loss_s1_bond
        + w_distogram * loss_s1_dist
    )
    loss_s2_total = (
        loss_s2_fm
        + alpha * w_lddt_full * loss_s2_lddt
        + w_bond * loss_s2_bond
        + w_clash * loss_s2_clash
        + w_ca_anchor * loss_s2_ca_anchor
    )
    loss = w_stage1 * loss_s1_total + loss_s2_total

    metrics = {
        "loss":              loss.item(),
        "s1_fm":             loss_s1_fm.item(),
        "s1_lddt":           loss_s1_lddt.item(),
        "s1_bond_caca":      loss_s1_bond.item(),
        "s1_distogram":      loss_s1_dist.item(),
        "s2_fm_atom":        loss_s2_fm.item(),
        "s2_lddt_full":      loss_s2_lddt.item(),
        "s2_bond":           loss_s2_bond.item(),
        "s2_clash":          loss_s2_clash.item(),
        "s2_ca_anchor":      loss_s2_ca_anchor.item(),
        "alpha":             alpha,
        "t_mean":            batch.t.mean().item(),
    }
    return loss, metrics


@torch.no_grad()
def joint_eval_step(
    model,
    batch: ProteinBatch,
    use_amp: bool = True,
    lddt_cutoff: float = 1.5,
) -> dict:
    """No-grad eval for joint Phase 3 — both stages' key metrics."""
    model.eval()
    amp_enabled = use_amp and batch.device.type == "cuda"

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        out = model(batch, return_aux=False)
        v_ca = out["v_ca"]
        v_atom = out["v_atom"]
        s1_ca = out["s1_ca"]
        s1_ca_cond = out.get("s1_ca_cond", s1_ca)

    v_ca_f32 = v_ca.float()
    v_atom_f32 = v_atom.float()
    s1_ca_f32 = s1_ca.float()
    s1_ca_cond_f32 = s1_ca_cond.float()

    x_clean_ca = batch.x_clean[..., CA_ATOM_ID, :].float()
    eps_ca = batch.eps[..., CA_ATOM_ID, :].float()
    x_t_ca = batch.x_t[..., CA_ATOM_ID, :].float()
    loss_s1_fm = _fm_loss_ca(v_ca_f32, x_clean_ca, eps_ca, batch.ca_mask)
    one_minus_t = (1.0 - batch.t.squeeze(-1).squeeze(-1).squeeze(-1)).float()
    x_hat_ca = x_t_ca + one_minus_t.view(-1, 1, 1) * v_ca_f32
    loss_s1_lddt = soft_lddt_ca_only(x_hat_ca, x_clean_ca, batch.ca_mask)

    non_ca_mask = _non_ca_atom_mask(batch.atom_mask)
    loss_s2_fm = _fm_loss_atom(
        v_atom_f32, batch.x_clean.float(), batch.eps.float(),
        non_ca_mask & batch.valid_mask,
    )
    x_t_s2 = _inject_ca(batch.x_t.float(), s1_ca_cond_f32)
    x_hat_atom = _x_hat_atom(x_t_s2, batch.t.float(), v_atom_f32)
    loss_s2_lddt = soft_lddt_ca_loss(x_hat_atom, batch.x_clean.float(),
                                     batch.ca_mask, cutoff=lddt_cutoff)
    loss_s2_clash = ca_clash_loss(x_hat_atom, batch.res_mask,
                                  chain_id=batch.chain_id)
    loss_s2_ca_anchor = _ca_anchor_loss(x_hat_atom, s1_ca_f32, batch.ca_mask)
    return {
        "s1_fm":          loss_s1_fm.item(),
        "s1_lddt":        loss_s1_lddt.item(),
        "s2_fm_atom":     loss_s2_fm.item(),
        "s2_lddt_full":   loss_s2_lddt.item(),
        "s2_clash":       loss_s2_clash.item(),
        "s2_ca_anchor":   loss_s2_ca_anchor.item(),
    }


# ── Stage 1 eval (kept below new Stage 2 / joint blocks for readability) ──


@torch.no_grad()
def stage1_eval_step(
    model,
    batch: ProteinBatch,
    use_amp: bool = True,
) -> dict:
    """No-grad eval for Stage 1."""
    model.eval()
    amp_enabled = use_amp and batch.device.type == "cuda"

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        v_ca, _ = model(batch)
        n_valid = batch.ca_mask.sum().clamp(min=1)
        v_rms = (v_ca.pow(2).sum() / n_valid / 3).sqrt()

    v_ca_f32 = v_ca.float()
    x_clean_ca = batch.x_clean[..., CA_ATOM_ID, :].float()
    eps_ca = batch.eps[..., CA_ATOM_ID, :].float()
    x_t_ca = batch.x_t[..., CA_ATOM_ID, :].float()

    loss_fm = _fm_loss_ca(v_ca_f32, x_clean_ca, eps_ca, batch.ca_mask)
    one_minus_t = (1.0 - batch.t.squeeze(-1).squeeze(-1).squeeze(-1)).float()
    x_hat_ca = x_t_ca + one_minus_t.view(-1, 1, 1) * v_ca_f32
    loss_lddt = soft_lddt_ca_only(x_hat_ca, x_clean_ca, batch.ca_mask)
    loss_bond = ca_ca_bond_loss(x_hat_ca, batch.ca_mask, batch.chain_id)
    return {
        "fm":         loss_fm.item(),
        "lddt":       loss_lddt.item(),
        "bond_caca":  loss_bond.item(),
        "v_rms":      v_rms.item(),
    }
