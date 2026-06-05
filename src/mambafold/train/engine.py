"""Training/eval step functions for the 2-stage coarse-to-fine pipeline.

Single-chain training path. The three training stages share two reusable
"loss surfaces" so the forward/loss math lives in exactly one place:

    Stage 1 (CA-only FM + aux):
        L = L_fm_ca + α(t)·w_lddt·L_lddt_ca + w_bond·L_bond_caca + w_dist·L_distogram

    Stage 2 (all-atom FM + aux, CA residual-refined from the Stage 1 anchor):
        L = L_fm_atom(non-CA) + α(t)·w_lddt·L_lddt_full
            + w_bond·L_bond + w_clash·L_clash + w_anchor·L_ca_anchor

    joint (Phase 3):
        L = w_stage1·L_stage1 + L_stage2   (both surfaces, both backprop)

Conventions (flow matching, normalized units):
    x_t = t·x_clean + (1-t)·ε         (built by the collator)
    velocity target = x_clean - ε
    one-step recon  = x_t + (1-t)·v   (used for lDDT/geometry metrics)
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


# ── shared low-level helpers ──────────────────────────────────────────────


def _alpha(t: Tensor, mode: str) -> float:
    """lDDT weight schedule. `const` → 1; `ramp` → 1 + 8·ReLU(t-0.5).mean."""
    if mode == "const":
        return 1.0
    return (1.0 + 8.0 * F.relu(t - 0.5)).mean().item()


def _recon_ca(x_t_ca: Tensor, t: Tensor, v_ca: Tensor) -> Tensor:
    """One-step FM reconstruction of CA positions. [B, L, 3]."""
    one_minus_t = (1.0 - t.squeeze(-1).squeeze(-1).squeeze(-1)).float()   # [B]
    return x_t_ca + one_minus_t.view(-1, 1, 1) * v_ca


def _recon_atom(x_t: Tensor, t: Tensor, v_atom: Tensor) -> Tensor:
    """One-step FM reconstruction for all atom slots. [B, L, A, 3]."""
    one_minus_t = (1.0 - t.squeeze(-1).squeeze(-1)).view(-1, 1, 1, 1)     # [B,1,1,1]
    return x_t + one_minus_t * v_atom


def _fm_loss_ca(v_pred: Tensor, x_clean: Tensor, eps: Tensor, mask: Tensor) -> Tensor:
    """Masked MSE for the FM target (x_clean − eps) on CA positions."""
    target = x_clean - eps                            # [B, L, 3]
    diff_sq = (v_pred - target).pow(2).sum(dim=-1)    # [B, L]
    m = mask.to(diff_sq.dtype)
    return (diff_sq * m).sum() / m.sum().clamp(min=1)


def _fm_loss_atom(v_pred: Tensor, x_clean: Tensor, eps: Tensor, mask: Tensor) -> Tensor:
    """Masked MSE for the FM target on per-atom velocity. mask: [B, L, A]."""
    target = x_clean - eps                                  # [B, L, A, 3]
    diff_sq = (v_pred - target).pow(2).sum(dim=-1)          # [B, L, A]
    m = mask.to(diff_sq.dtype)
    return (diff_sq * m).sum() / m.sum().clamp(min=1)


def _non_ca_atom_mask(atom_mask: Tensor) -> Tensor:
    """Per-atom mask that excludes the CA slot (Stage 1 owns CA)."""
    A = atom_mask.shape[-1]
    not_ca = torch.arange(A, device=atom_mask.device) != CA_ATOM_ID
    return atom_mask & not_ca.view(1, 1, A)


def _inject_ca(x_t: Tensor, ca: Tensor) -> Tensor:
    """Return a copy of x_t with the CA slot overwritten by `ca`."""
    out = x_t.clone()
    out[..., CA_ATOM_ID, :] = ca
    return out


def _ca_anchor_loss(x_hat: Tensor, s1_ca: Tensor, ca_mask: Tensor) -> Tensor:
    """Pull the Stage-2 refined CA toward the Stage-1 scaffold.

    The anchor is a fixed *target*, so it is detached: gradients refine the
    Stage-2 CA only and never drag Stage 1 toward Stage 2 (matters in joint,
    where Stage 1 still has grad).
    """
    pred_ca = x_hat[..., CA_ATOM_ID, :]
    diff_sq = (pred_ca - s1_ca.detach()).pow(2).sum(dim=-1)
    m = ca_mask.to(diff_sq.dtype)
    return (diff_sq * m).sum() / m.sum().clamp(min=1)


# ── reusable loss surfaces (shared by forward + joint) ─────────────────────


def _stage1_loss_surface(
    out: dict,
    batch: ProteinBatch,
    *,
    alpha_mode: str,
    w_lddt_ca: float,
    w_bond_caca: float,
    w_distogram: float,
) -> tuple[Tensor, dict]:
    """Stage 1 composite loss + per-component metrics from a model output dict.

    `out` must carry `v_ca` and `distogram_logits` (model called with
    return_aux=True). The FM main loss is on raw velocity; `x_hat_ca` is the
    one-step recon used by the lDDT-style auxiliaries.
    """
    v_ca = out["v_ca"].float()
    dist_logits = out["distogram_logits"].float()

    x_clean_ca = batch.x_clean[..., CA_ATOM_ID, :].float()    # [B, L, 3]
    eps_ca = batch.eps[..., CA_ATOM_ID, :].float()
    x_t_ca = batch.x_t[..., CA_ATOM_ID, :].float()

    loss_fm = _fm_loss_ca(v_ca, x_clean_ca, eps_ca, batch.ca_mask)
    x_hat_ca = _recon_ca(x_t_ca, batch.t, v_ca)
    loss_lddt = soft_lddt_ca_only(x_hat_ca, x_clean_ca, batch.ca_mask)
    loss_bond = ca_ca_bond_loss(x_hat_ca, batch.ca_mask, batch.chain_id)
    loss_dist = distogram_loss_ca_only(dist_logits, x_clean_ca, batch.ca_mask)

    alpha = _alpha(batch.t, alpha_mode)
    total = (
        loss_fm
        + alpha * w_lddt_ca * loss_lddt
        + w_bond_caca * loss_bond
        + w_distogram * loss_dist
    )
    metrics = {
        "fm":        loss_fm.item(),
        "lddt":      loss_lddt.item(),
        "bond_caca": loss_bond.item(),
        "distogram": loss_dist.item(),
        "alpha":     alpha,
    }
    return total, metrics


def _stage2_loss_surface(
    out: dict,
    batch: ProteinBatch,
    *,
    alpha_mode: str,
    w_lddt_full: float,
    w_bond: float,
    w_clash: float,
    w_ca_anchor: float,
    lddt_cutoff: float,
) -> tuple[Tensor, dict]:
    """Stage 2 composite loss + metrics from a model output dict.

    Non-CA atoms get the FM loss. CA is initialized from Stage 1 (`s1_ca_cond`)
    and may move via lDDT/geometry gradients, but the anchor loss keeps it near
    the Stage 1 scaffold.
    """
    v_atom = out["v_atom"].float()
    s1_ca = out["s1_ca"].float()
    s1_ca_cond = out.get("s1_ca_cond", out["s1_ca"]).float()

    non_ca_mask = _non_ca_atom_mask(batch.atom_mask)               # [B, L, A]
    loss_fm = _fm_loss_atom(
        v_atom, batch.x_clean.float(), batch.eps.float(),
        non_ca_mask & batch.valid_mask,
    )

    x_t_s2 = _inject_ca(batch.x_t.float(), s1_ca_cond)
    x_hat = _recon_atom(x_t_s2, batch.t, v_atom)

    loss_lddt = soft_lddt_ca_loss(x_hat, batch.x_clean.float(),
                                  batch.ca_mask, cutoff=lddt_cutoff)
    loss_bond = bond_length_loss(x_hat, batch.res_type, batch.atom_mask, batch.res_mask)
    loss_clash = ca_clash_loss(x_hat, batch.res_mask, chain_id=batch.chain_id)
    loss_anchor = _ca_anchor_loss(x_hat, s1_ca, batch.ca_mask)

    alpha = _alpha(batch.t, alpha_mode)
    total = (
        loss_fm
        + alpha * w_lddt_full * loss_lddt
        + w_bond * loss_bond
        + w_clash * loss_clash
        + w_ca_anchor * loss_anchor
    )
    metrics = {
        "fm_atom":   loss_fm.item(),
        "lddt_full": loss_lddt.item(),
        "bond":      loss_bond.item(),
        "clash":     loss_clash.item(),
        "ca_anchor": loss_anchor.item(),
        "alpha":     alpha,
    }
    return total, metrics


# ── Stage 1 (CA-only) ──────────────────────────────────────────────────────


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

    Distogram aux is always computed (return_aux=True) so the pair stack gets
    gradient on the binning signal from step 1.
    """
    model.train()
    amp_enabled = use_amp and batch.device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        out = model(batch, return_aux=True)

    loss, metrics = _stage1_loss_surface(
        out, batch, alpha_mode=alpha_mode,
        w_lddt_ca=w_lddt_ca, w_bond_caca=w_bond_caca, w_distogram=w_distogram,
    )
    metrics["loss"] = loss.item()
    metrics["t_mean"] = batch.t.mean().item()
    return loss, metrics


@torch.no_grad()
def stage1_eval_step(model, batch: ProteinBatch, use_amp: bool = True) -> dict:
    """No-grad eval for Stage 1."""
    model.eval()
    amp_enabled = use_amp and batch.device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        v_ca, _ = model(batch)
        n_valid = batch.ca_mask.sum().clamp(min=1)
        v_rms = (v_ca.pow(2).sum() / n_valid / 3).sqrt()

    v_ca = v_ca.float()
    x_clean_ca = batch.x_clean[..., CA_ATOM_ID, :].float()
    eps_ca = batch.eps[..., CA_ATOM_ID, :].float()
    x_t_ca = batch.x_t[..., CA_ATOM_ID, :].float()

    x_hat_ca = _recon_ca(x_t_ca, batch.t, v_ca)
    return {
        "fm":        _fm_loss_ca(v_ca, x_clean_ca, eps_ca, batch.ca_mask).item(),
        "lddt":      soft_lddt_ca_only(x_hat_ca, x_clean_ca, batch.ca_mask).item(),
        "bond_caca": ca_ca_bond_loss(x_hat_ca, batch.ca_mask, batch.chain_id).item(),
        "v_rms":     v_rms.item(),
    }


# ── Stage 2 (TwoStageMambaFold with frozen S1) ─────────────────────────────


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
    """Stage 2 forward (S1 frozen via TwoStage wrapper) + atom-side losses."""
    model.train()
    amp_enabled = use_amp and batch.device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        # return_aux=False — no distogram needed in Phase 2 (frozen S1).
        out = model(
            batch, return_aux=False,
            ca_condition_noise_std=ca_condition_noise_std,
            ca_condition_noise_prob=ca_condition_noise_prob,
        )

    loss, metrics = _stage2_loss_surface(
        out, batch, alpha_mode=alpha_mode,
        w_lddt_full=w_lddt_full, w_bond=w_bond, w_clash=w_clash,
        w_ca_anchor=w_ca_anchor, lddt_cutoff=lddt_cutoff,
    )
    metrics["loss"] = loss.item()
    metrics["t_mean"] = batch.t.mean().item()
    return loss, metrics


@torch.no_grad()
def stage2_eval_step(
    model, batch: ProteinBatch, lddt_cutoff: float = 1.5, use_amp: bool = True,
) -> dict:
    """No-grad eval step for Phase 2 / Phase 3 (TwoStage model)."""
    model.eval()
    amp_enabled = use_amp and batch.device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        out = model(batch, return_aux=False)
        n_valid = batch.valid_mask.sum().clamp(min=1)
        v_rms = (out["v_atom"].pow(2).sum() / n_valid / 3).sqrt()

    _, metrics = _stage2_loss_surface(
        out, batch, alpha_mode="const",
        w_lddt_full=1.0, w_bond=0.0, w_clash=0.0, w_ca_anchor=0.0,
        lddt_cutoff=lddt_cutoff,
    )
    metrics.pop("alpha", None)
    metrics["v_rms"] = v_rms.item()
    return metrics


# ── joint (Phase 3) — both stages backprop ─────────────────────────────────


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
    """Joint Phase 3 loss — Stage 1 + residual-refining Stage 2 (both backprop)."""
    model.train()
    amp_enabled = use_amp and batch.device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        out = model(
            batch, return_aux=True,
            ca_condition_noise_std=ca_condition_noise_std,
            ca_condition_noise_prob=ca_condition_noise_prob,
        )

    loss_s1, m_s1 = _stage1_loss_surface(
        out, batch, alpha_mode=alpha_mode,
        w_lddt_ca=w_lddt_ca, w_bond_caca=w_bond_caca, w_distogram=w_distogram,
    )
    loss_s2, m_s2 = _stage2_loss_surface(
        out, batch, alpha_mode=alpha_mode,
        w_lddt_full=w_lddt_full, w_bond=w_bond, w_clash=w_clash,
        w_ca_anchor=w_ca_anchor, lddt_cutoff=lddt_cutoff,
    )
    loss = w_stage1 * loss_s1 + loss_s2

    metrics = {
        "loss":         loss.item(),
        "s1_fm":        m_s1["fm"],
        "s1_lddt":      m_s1["lddt"],
        "s1_bond_caca": m_s1["bond_caca"],
        "s1_distogram": m_s1["distogram"],
        "s2_fm_atom":   m_s2["fm_atom"],
        "s2_lddt_full": m_s2["lddt_full"],
        "s2_bond":      m_s2["bond"],
        "s2_clash":     m_s2["clash"],
        "s2_ca_anchor": m_s2["ca_anchor"],
        "alpha":        m_s1["alpha"],
        "t_mean":       batch.t.mean().item(),
    }
    return loss, metrics


@torch.no_grad()
def joint_eval_step(
    model, batch: ProteinBatch, use_amp: bool = True, lddt_cutoff: float = 1.5,
) -> dict:
    """No-grad eval for joint Phase 3 — both stages' key metrics."""
    model.eval()
    amp_enabled = use_amp and batch.device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        out = model(batch, return_aux=False)

    v_ca = out["v_ca"].float()
    x_clean_ca = batch.x_clean[..., CA_ATOM_ID, :].float()
    eps_ca = batch.eps[..., CA_ATOM_ID, :].float()
    x_t_ca = batch.x_t[..., CA_ATOM_ID, :].float()
    x_hat_ca = _recon_ca(x_t_ca, batch.t, v_ca)

    _, m_s2 = _stage2_loss_surface(
        out, batch, alpha_mode="const",
        w_lddt_full=1.0, w_bond=0.0, w_clash=0.0, w_ca_anchor=0.0,
        lddt_cutoff=lddt_cutoff,
    )
    return {
        "s1_fm":        _fm_loss_ca(v_ca, x_clean_ca, eps_ca, batch.ca_mask).item(),
        "s1_lddt":      soft_lddt_ca_only(x_hat_ca, x_clean_ca, batch.ca_mask).item(),
        "s2_fm_atom":   m_s2["fm_atom"],
        "s2_lddt_full": m_s2["lddt_full"],
        "s2_clash":     m_s2["clash"],
        "s2_ca_anchor": m_s2["ca_anchor"],
    }
