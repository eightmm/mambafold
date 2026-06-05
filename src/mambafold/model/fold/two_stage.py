"""TwoStageMambaFold — joint wrapper running Stage 1 then Stage 2.

Used during Phase 2 (Stage 1 frozen, only Stage 2 trains) and Phase 3
(joint backprop, both stages update). The `freeze_stage1` flag decides
whether Stage 1's outputs are detached before flowing into Stage 2.

Phase-2 forward:
    with torch.no_grad():
        v_ca, s1_latent = stage1(batch)
    s1_ca = batch.x_t_ca + (1-t) * v_ca
    optional noisy condition: s1_ca_cond = s1_ca + noise
    batch2 = batch with x_t's CA slot <- s1_ca_cond
    v_atom = stage2(batch2, s1_ca_cond, s1_latent)

Phase-3 forward: same but no torch.no_grad and no detach.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
from torch import Tensor

from mambafold.data.constants import CA_ATOM_ID
from mambafold.data.types import ProteinBatch
from mambafold.model.fold.stage1_ca import MambaFoldStage1
from mambafold.model.fold.stage2_atom import MambaFoldStage2


class TwoStageMambaFold(nn.Module):
    """Combined Stage 1 + Stage 2 wrapper for Phase 2/3 training.

    Args:
        stage1: pre-built `MambaFoldStage1`.
        stage2: pre-built `MambaFoldStage2`.
        freeze_stage1: if True, Stage 1 runs under `torch.no_grad()` and its
            outputs are detached. Stage 2 gradients cannot reach Stage 1.
            Phase 2 = True, Phase 3 = False.
    """

    def __init__(
        self,
        stage1: MambaFoldStage1,
        stage2: MambaFoldStage2,
        freeze_stage1: bool = True,
    ):
        super().__init__()
        self.stage1 = stage1
        self.stage2 = stage2
        self.freeze_stage1 = False                       # set via setter for consistency
        self.set_stage1_frozen(freeze_stage1)

    def set_stage1_frozen(self, frozen: bool):
        """Toggle Stage 1 freeze on the fly (Phase 2 ↔ Phase 3).

        Updates BOTH `freeze_stage1` (controls the runtime no_grad context)
        AND `requires_grad` on every Stage 1 parameter (so optimizers built
        with `filter(lambda p: p.requires_grad, ...)` skip them cleanly).
        """
        self.freeze_stage1 = bool(frozen)
        for p in self.stage1.parameters():
            p.requires_grad = not frozen

    @staticmethod
    def _x_hat_ca(x_t: Tensor, t: Tensor, v_ca: Tensor) -> Tensor:
        """One-step recon of the CA position from Stage 1's velocity."""
        x_t_ca = x_t[..., CA_ATOM_ID, :]                               # [B, L, 3]
        one_minus_t = (1.0 - t.squeeze(-1).squeeze(-1).squeeze(-1)).view(-1, 1, 1)
        return x_t_ca + one_minus_t * v_ca

    def forward(
        self,
        batch: ProteinBatch,
        return_aux: bool = False,
        ca_condition_noise_std: float = 0.0,
        ca_condition_noise_prob: float = 0.0,
    ) -> dict:
        """
        Args:
            return_aux: If True, runs Stage 1 with `return_aux=True` so the
                output dict also carries `distogram_logits`.
            ca_condition_noise_std/prob: optional training-time perturbation
                applied to the Stage 1 C-alpha condition before Stage 2 sees it.

        Returns dict with keys:
            v_ca               [B, L, 3]
            v_atom             [B, L, A, 3]
            s1_ca              [B, L, 3]    — Stage 1 C-alpha anchor
            s1_ca_cond         [B, L, 3]    — possibly noised Stage 2 condition
            s1_pcb             [B, L, 3]    — Stage 1 pseudo-Cβ direction
            s1_conf            [B, L]       — Stage 1 per-residue confidence
            s1_latent          [B, L, d_res]
            distogram_logits   [B, L, L, n_bins] — only if return_aux=True
        """
        # Stage 1 runs under no_grad when frozen (Phase 2); its outputs are then
        # detached so Stage 2 gradients cannot reach Stage 1.
        with torch.no_grad() if self.freeze_stage1 else nullcontext():
            s1 = self.stage1(batch, return_aux=return_aux)

        def _maybe_detach(x):
            return x.detach() if self.freeze_stage1 else x

        v_ca = _maybe_detach(s1["v_ca"])
        s1_latent = _maybe_detach(s1["trunk_latent"])
        s1_pcb = _maybe_detach(s1["pcb_dir"])
        s1_conf = _maybe_detach(s1["conf"])
        dist_logits = _maybe_detach(s1["distogram_logits"]) if return_aux else None
        contact_logits = _maybe_detach(s1["contact_logits"]) if return_aux else None

        s1_ca = self._x_hat_ca(batch.x_t, batch.t, v_ca)                   # [B, L, 3]
        s1_ca_cond = s1_ca
        if self.training and ca_condition_noise_std > 0.0 and ca_condition_noise_prob > 0.0:
            if torch.rand((), device=s1_ca.device) < ca_condition_noise_prob:
                mask = batch.ca_mask.unsqueeze(-1).to(s1_ca.dtype)
                s1_ca_cond = s1_ca + torch.randn_like(s1_ca) * ca_condition_noise_std * mask

        x_t_s2 = MambaFoldStage2.inject_ca_slot(batch.x_t, s1_ca_cond)
        batch2 = batch.with_coords(x_t_s2)

        v_atom = self.stage2(
            batch2, s1_ca=s1_ca_cond, s1_latent=s1_latent,
            s1_pcb=s1_pcb, s1_conf=s1_conf,
        )

        # Canonical Stage-1 keys (pcb_dir/conf/distogram_logits/contact_logits)
        # so the shared engine loss surfaces read the same names in every stage.
        out = {
            "v_ca": v_ca,
            "v_atom": v_atom,
            "s1_ca": s1_ca,
            "s1_ca_cond": s1_ca_cond,
            "pcb_dir": s1_pcb,
            "conf": s1_conf,
            "s1_latent": s1_latent,
        }
        if return_aux:
            out["distogram_logits"] = dist_logits
            out["contact_logits"] = contact_logits
        return out
