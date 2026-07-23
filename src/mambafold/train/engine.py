"""Direct all-atom training/eval steps.

Flow matching convention:
    x_t = t * x_clean + (1 - t) * eps
    velocity target = x_clean - eps
    one-step reconstruction = x_t + (1 - t) * v
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import Tensor

from mambafold.data.constants import CA_ATOM_ID
from mambafold.data.types import ProteinBatch
from mambafold.losses.ca_only import (
    allatom_chirality_loss,
    ca_chirality_loss,
    ca_self_clash,
    ca_virtual_angle_floor,
    confidence_loss_allatom,
    contact_loss_ca,
    distogram_loss_ca_only,
    drmsd_ca,
    pseudo_cb_loss,
)
from mambafold.losses.geometry import (
    all_atom_clash_loss,
    bond_length_loss,
    ca_clash_loss,
)
from mambafold.losses.lddt import soft_lddt_all_atom_loss, soft_lddt_ca_loss


def _zero_like_loss(ref: Tensor) -> Tensor:
    return ref.sum() * 0.0


def _alpha(t: Tensor, mode: str) -> float:
    if mode == "const":
        return 1.0
    return (1.0 + 8.0 * F.relu(t - 0.5)).mean().item()


def _recon_atom(x_t: Tensor, t: Tensor, v_atom: Tensor) -> Tensor:
    one_minus_t = (1.0 - t.squeeze(-1).squeeze(-1)).view(-1, 1, 1, 1)
    return x_t + one_minus_t * v_atom


def _fm_loss_atom(v_pred: Tensor, x_clean: Tensor, eps: Tensor, mask: Tensor) -> Tensor:
    target = x_clean - eps
    diff_sq = (v_pred - target).pow(2).sum(dim=-1)
    m = mask.to(diff_sq.dtype)
    return (diff_sq * m).sum() / m.sum().clamp(min=1)


def _fm_loss_ca(v_pred: Tensor, x_clean: Tensor, eps: Tensor, mask: Tensor) -> Tensor:
    target = x_clean - eps
    diff_sq = (v_pred - target).pow(2).sum(dim=-1)
    m = mask.to(diff_sq.dtype)
    return (diff_sq * m).sum() / m.sum().clamp(min=1)


def allatom_loss_surface(
    out: dict,
    batch: ProteinBatch,
    *,
    alpha_mode: str,
    w_fm: float,
    w_lddt_atom: float,
    w_lddt_ca: float,
    w_bond: float,
    w_clash: float,
    w_ca_clash: float,
    w_distogram: float,
    w_drmsd: float,
    w_contact: float,
    w_pcb: float,
    w_conf: float,
    w_ca_angle: float,
    w_ca_self_clash: float,
    w_chirality: float,
    w_chirality_atom: float,
    lddt_cutoff: float,
    max_lddt_atoms: int,
    max_clash_atoms: int,
) -> tuple[Tensor, dict]:
    """Direct all-atom composite loss.

    Main supervision is all-atom FM. Geometry follows the AF/SimpleFold spirit:
    sampled all-atom lDDT for local atom neighborhoods, backbone/CB bond terms,
    non-bonded atom clash, plus CA topology auxiliaries for global fold quality.
    """
    v_atom = out["v_atom"].float()
    v_ca = out["v_ca"].float()
    x_clean = batch.x_clean.float()
    x_t = batch.x_t.float()
    eps = batch.eps.float()
    x_clean_ca = x_clean[..., CA_ATOM_ID, :]
    eps_ca = eps[..., CA_ATOM_ID, :]

    x_hat = _recon_atom(x_t, batch.t, v_atom)
    x_hat_ca = x_hat[..., CA_ATOM_ID, :]

    loss_fm = _fm_loss_atom(v_atom, x_clean, eps, batch.valid_mask)
    loss_lddt_atom = soft_lddt_all_atom_loss(
        x_hat,
        x_clean,
        batch.valid_mask,
        cutoff=lddt_cutoff,
        max_atoms=max_lddt_atoms,
    )
    loss_lddt_ca = soft_lddt_ca_loss(x_hat, x_clean, batch.ca_mask, cutoff=lddt_cutoff)
    loss_bond = (
        bond_length_loss(
            x_hat,
            batch.res_type,
            batch.atom_mask,
            batch.res_mask,
            chain_id=batch.chain_id,
            res_seq_nums=batch.res_seq_nums,
            true_coords=x_clean,
        )
        if w_bond
        else _zero_like_loss(x_hat)
    )
    loss_clash = (
        all_atom_clash_loss(
            x_hat,
            batch.valid_mask,
            batch.res_mask,
            chain_id=batch.chain_id,
            max_atoms=max_clash_atoms,
        )
        if w_clash
        else _zero_like_loss(x_hat)
    )
    loss_ca_clash = (
        ca_clash_loss(x_hat, batch.res_mask, chain_id=batch.chain_id)
        if w_ca_clash
        else _zero_like_loss(x_hat)
    )

    loss_dist = (
        distogram_loss_ca_only(out["distogram_logits"].float(), x_clean_ca, batch.ca_mask)
        if w_distogram and "distogram_logits" in out
        else _zero_like_loss(x_hat)
    )
    loss_drmsd = drmsd_ca(x_hat_ca, x_clean_ca, batch.ca_mask)
    loss_contact = (
        contact_loss_ca(out["contact_logits"].float(), x_clean_ca, batch.ca_mask)
        if w_contact and "contact_logits" in out
        else _zero_like_loss(x_hat)
    )
    loss_pcb = (
        pseudo_cb_loss(out["pcb_dir"].float(), x_clean, batch.atom_mask, batch.res_mask)
        if w_pcb
        else _zero_like_loss(x_hat)
    )
    loss_conf = (
        confidence_loss_allatom(
            out["conf"].float(), x_hat, x_clean, batch.valid_mask, batch.res_mask
        )
        if w_conf
        else _zero_like_loss(x_hat)
    )
    loss_angle = (
        ca_virtual_angle_floor(
            x_hat_ca,
            batch.ca_mask,
            batch.chain_id,
            batch.res_seq_nums,
            true_ca=x_clean_ca,
        )
        if w_ca_angle
        else _zero_like_loss(x_hat)
    )
    loss_selfclash = (
        ca_self_clash(x_hat_ca, batch.ca_mask, batch.chain_id)
        if w_ca_self_clash
        else _zero_like_loss(x_hat)
    )
    loss_chir = (
        ca_chirality_loss(
            x_hat_ca,
            x_clean_ca,
            batch.ca_mask,
            batch.chain_id,
            batch.res_seq_nums,
        )
        if w_chirality
        else _zero_like_loss(x_hat)
    )
    loss_chir_atom = (
        allatom_chirality_loss(
            x_hat,
            x_clean,
            batch.atom_mask,
            batch.res_mask,
        )
        if w_chirality_atom
        else _zero_like_loss(x_hat)
    )

    alpha = _alpha(batch.t, alpha_mode)
    total = (
        w_fm * loss_fm
        + alpha * w_lddt_atom * loss_lddt_atom
        + alpha * w_lddt_ca * loss_lddt_ca
        + w_bond * loss_bond
        + w_clash * loss_clash
        + w_ca_clash * loss_ca_clash
        + w_distogram * loss_dist
        + w_drmsd * loss_drmsd
        + w_contact * loss_contact
        + w_pcb * loss_pcb
        + w_conf * loss_conf
        + w_ca_angle * loss_angle
        + w_ca_self_clash * loss_selfclash
        + w_chirality * loss_chir
        + w_chirality_atom * loss_chir_atom
    )
    metrics = {
        "fm_atom": loss_fm.item(),
        "ca_fm": _fm_loss_ca(v_ca, x_clean_ca, eps_ca, batch.ca_mask).item(),
        "lddt_atom": loss_lddt_atom.item(),
        "lddt_ca": loss_lddt_ca.item(),
        "bond": loss_bond.item(),
        "clash": loss_clash.item(),
        "ca_clash": loss_ca_clash.item(),
        "distogram": loss_dist.item(),
        "drmsd": loss_drmsd.item(),
        "contact": loss_contact.item(),
        "pcb": loss_pcb.item(),
        "conf": loss_conf.item(),
        "ca_angle": loss_angle.item(),
        "ca_self_clash": loss_selfclash.item(),
        "chirality": loss_chir.item(),
        "chirality_atom": loss_chir_atom.item(),
        "alpha": alpha,
    }
    return total, metrics


def allatom_forward_and_loss(
    model,
    batch: ProteinBatch,
    *,
    alpha_mode: str = "ramp",
    use_amp: bool = True,
    w_fm: float = 1.0,
    w_lddt_atom: float = 1.0,
    w_lddt_ca: float = 0.5,
    w_bond: float = 0.05,
    w_clash: float = 0.02,
    w_ca_clash: float = 0.01,
    w_distogram: float = 0.5,
    w_drmsd: float = 0.75,
    w_contact: float = 0.5,
    w_pcb: float = 0.2,
    w_conf: float = 0.05,
    w_ca_angle: float = 0.1,
    w_ca_self_clash: float = 0.1,
    w_chirality: float = 1.0,
    w_chirality_atom: float = 0.5,
    lddt_cutoff: float = 1.5,
    max_lddt_atoms: int = 2048,
    max_clash_atoms: int = 2048,
    self_condition_prob: float = 0.0,
):
    model.train()
    amp_enabled = use_amp and batch.device.type == "cuda"
    raw_model = getattr(model, "module", model)
    used_self_cond = False
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        if self_condition_prob > 0.0 and getattr(raw_model, "self_conditioning", False):
            p = max(0.0, min(1.0, float(self_condition_prob)))
            if torch.rand((), device=batch.device).item() < p:
                with torch.no_grad():
                    sc_out = model(batch, return_aux=False)
                    x_self_cond = _recon_atom(
                        batch.x_t.float(), batch.t, sc_out["v_atom"].float()
                    ).detach()
                batch = replace(batch, x_self_cond=x_self_cond)
                used_self_cond = True
        out = model(batch, return_aux=True)

    loss, metrics = allatom_loss_surface(
        out,
        batch,
        alpha_mode=alpha_mode,
        w_fm=w_fm,
        w_lddt_atom=w_lddt_atom,
        w_lddt_ca=w_lddt_ca,
        w_bond=w_bond,
        w_clash=w_clash,
        w_ca_clash=w_ca_clash,
        w_distogram=w_distogram,
        w_drmsd=w_drmsd,
        w_contact=w_contact,
        w_pcb=w_pcb,
        w_conf=w_conf,
        w_ca_angle=w_ca_angle,
        w_ca_self_clash=w_ca_self_clash,
        w_chirality=w_chirality,
        w_chirality_atom=w_chirality_atom,
        lddt_cutoff=lddt_cutoff,
        max_lddt_atoms=max_lddt_atoms,
        max_clash_atoms=max_clash_atoms,
    )
    metrics["loss"] = loss.item()
    metrics["t_mean"] = batch.t.mean().item()
    metrics["self_cond"] = 1.0 if used_self_cond else 0.0
    return loss, metrics


@torch.no_grad()
def allatom_eval_step(
    model,
    batch: ProteinBatch,
    use_amp: bool = True,
    lddt_cutoff: float = 1.5,
    max_lddt_atoms: int = 2048,
    max_clash_atoms: int = 2048,
) -> dict:
    model.eval()
    amp_enabled = use_amp and batch.device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        out = model(batch, return_aux=True)
        n_valid = batch.valid_mask.sum().clamp(min=1)
        v_rms = (out["v_atom"].pow(2).sum() / n_valid / 3).sqrt()

    _, metrics = allatom_loss_surface(
        out,
        batch,
        alpha_mode="const",
        w_fm=1.0,
        w_lddt_atom=1.0,
        w_lddt_ca=1.0,
        w_bond=0.0,
        w_clash=0.0,
        w_ca_clash=0.0,
        w_distogram=0.0,
        w_drmsd=0.0,
        w_contact=0.0,
        w_pcb=0.0,
        w_conf=0.0,
        w_ca_angle=0.0,
        w_ca_self_clash=0.0,
        w_chirality=0.0,
        w_chirality_atom=0.0,
        lddt_cutoff=lddt_cutoff,
        max_lddt_atoms=max_lddt_atoms,
        max_clash_atoms=max_clash_atoms,
    )
    return {
        "fm_atom": metrics["fm_atom"],
        "ca_fm": metrics["ca_fm"],
        "lddt_atom": metrics["lddt_atom"],
        "lddt_ca": metrics["lddt_ca"],
        "bond": metrics["bond"],
        "clash": metrics["clash"],
        "v_rms": v_rms.item(),
    }
