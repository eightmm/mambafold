"""Stage 1 — CA-only flow matching model with Linear Triangle pair stack.

Inputs (residue-level only, sliced from full ProteinBatch):
    res_type, ESM3, chain_id/entity_id/sym_id, res_seq_nums,
    is_nterm/is_cterm, x_t^CA (= batch.x_t[..., CA_ATOM_ID, :]), t

Outputs:
    v_ca          [B, L, 3]      — FM velocity for CA
    trunk_latent  [B, L, d_res]  — passed to Stage 2 as conditioning

See `docs/architecture.md` §3 for the design rationale.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mambafold.data.constants import AA_TO_ID, CA_ATOM_ID, COORD_SCALE
from mambafold.data.types import ProteinBatch
from mambafold.model.bimamba3 import MambaStack
from mambafold.model.embeddings import (
    CoordinateFourierEmbedder,
    SequenceFourierEmbedder,
)
from mambafold.model.fold.pair_blocks import PairBlock, PairToSingleAttention

NUM_RES_TYPES = len(AA_TO_ID)  # 21 (20 AAs + UNK)


class MambaFoldStage1(nn.Module):
    """Coarse-stage CA-only flow matching model.

    Param target: ~225M (most of it in the Mamba trunk; pair stack ~3-5M).

    Args:
        d_res: Residue token dim. Default 1024.
        n_trunk: Number of BiMamba3 layers in the trunk. Default 12.
        d_res_type: Residue-type embedding dim. Default 32.
        d_res_pos: Sequence position (chain/entity/sym + Fourier) embed dim.
        d_plm: PLM (ESM3) embedding dim. Default 1536.
        d_plm_proj: Internal PLM projection width. Default 256.
        d_ca_emb: Fourier embed dim of x_t^CA scalar coords. Default 128.
        use_plm: If True, ESM3 features are required (loud error otherwise).
        # Pair stack
        d_pair: Pair tensor dim. Default 192.
        n_pair_blocks: Number of PairBlocks. Default 4 (after I1 memory profile).
        n_pair_heads: Heads inside LinearTriangleAttention. Default 4.
        d_pair_head: Per-head dim in LinearTriangleAttention. Default 48.
        pair_mult_c: Hidden width inside TriangleMultiplicativeUpdate. Default 128.
        # SSM
        mimo_rank: Mamba MIMO rank. Default 4.
        d_state: Mamba state dim. Default 64.
        expand: Mamba inner-dim multiplier. Default 2.
        headdim: Mamba head dim. Default 64.
        bidirectional: BiMamba (default True) vs causal Mamba.
        # Pair init
        relpos_max: AF3 relpos clip range. Default 32 (gives 65 + OUT_OF_CHAIN bins).
    """

    def __init__(
        self,
        d_res: int = 1024,
        n_trunk: int = 12,
        d_res_type: int = 32,
        d_res_pos: int = 64,
        d_plm: int = 1536,
        d_plm_proj: int = 256,
        d_ca_emb: int = 128,
        use_plm: bool = True,
        d_pair: int = 192,
        n_pair_blocks: int = 4,
        n_pair_heads: int = 4,
        d_pair_head: int = 48,
        pair_mult_c: int = 128,
        mimo_rank: int = 4,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 64,
        bidirectional: bool = True,
        relpos_max: int = 32,
        n_cycles: int = 1,
        n_recycle_bins: int = 32,
        recycle_max_dist: float = 22.0,
        pair_use_mult_update: bool = True,
        pair_use_tri_attn: bool = True,
        trunk_attn_layers: list[int] | None = None,
        trunk_attn_every: int | None = None,
        n_attn_heads: int = 16,
    ):
        super().__init__()
        self.d_res = d_res
        self.d_pair = d_pair
        self.use_plm = use_plm
        self.d_plm = d_plm
        self.relpos_max = relpos_max
        self.n_relpos_bins = 2 * relpos_max + 2  # in-chain bins + OUT_OF_CHAIN
        # Recycling: previous-cycle Cα distance map fed back into the pair init.
        self.n_cycles = n_cycles
        self.n_recycle_bins = n_recycle_bins
        self.recycle_max_dist = recycle_max_dist

        # ── Residue-side embedders ──────────────────────────────────────
        self.res_type_embed = nn.Embedding(NUM_RES_TYPES, d_res_type)
        self.seq_pos_embed = SequenceFourierEmbedder(d_out=d_res_pos) if d_res_pos > 0 else None
        self.ca_coord_embed = CoordinateFourierEmbedder(d_out=d_ca_emb)

        if use_plm:
            self.plm_norm = nn.LayerNorm(d_plm)
            self.plm_proj = nn.Linear(d_plm, d_plm_proj)
        else:
            self.plm_norm = None
            self.plm_proj = None
            d_plm_proj = 0  # contributes 0 to trunk input width

        # Trunk input = ca_emb + t(1) + termini(2) + chain_break(1)
        #             + res_type_emb + seq_pos_emb + plm_proj
        trunk_in_dim = d_ca_emb + 1 + 2 + 1 + d_res_type + d_res_pos + d_plm_proj
        self.trunk_input_norm = nn.LayerNorm(trunk_in_dim)
        self.trunk_proj = nn.Linear(trunk_in_dim, d_res)

        # ── Sequence trunk ──────────────────────────────────────────────
        self.residue_trunk = MambaStack(
            d_res, n_trunk,
            d_state=d_state, mimo_rank=mimo_rank, expand=expand, headdim=headdim,
            bidirectional=bidirectional,
            attn_layers=trunk_attn_layers, attn_every=trunk_attn_every,
            n_attn_heads=n_attn_heads, attn_relpos_max=relpos_max,
        )

        # ── Pair representation ─────────────────────────────────────────
        self.pair_init_single = nn.Linear(d_res, d_pair)
        if use_plm:
            self.pair_init_esm = nn.Linear(d_plm_proj, d_pair)
        else:
            self.pair_init_esm = None
        self.relpos_embed = nn.Embedding(self.n_relpos_bins, d_pair)
        # Recycled Cα distance → pair feature (zero-init so cycle 1 is unchanged).
        self.recycle_dist_embed = nn.Embedding(n_recycle_bins, d_pair)
        nn.init.zeros_(self.recycle_dist_embed.weight)

        self.pair_blocks = nn.ModuleList([
            PairBlock(
                d_pair=d_pair,
                n_heads=n_pair_heads,
                d_head=d_pair_head,
                mult_c=pair_mult_c,
                use_mult_update=pair_use_mult_update,
                use_tri_attn=pair_use_tri_attn,
            )
            for _ in range(n_pair_blocks)
        ])

        # pair → single bias: attention pooling over each row (keeps which j
        # matters) projected back to d_res.
        self.pair_to_single = PairToSingleAttention(d_pair, d_res, n_heads=n_pair_heads)

        # ── CA output head ──────────────────────────────────────────────
        self.ca_head = nn.Sequential(
            nn.LayerNorm(d_res),
            nn.Linear(d_res, d_res // 2),
            nn.GELU(),
            nn.Linear(d_res // 2, 3),
        )

        # ── pseudo-Cβ direction head (side-chain orientation cue for Stage 2)
        self.pcb_head = nn.Sequential(
            nn.LayerNorm(d_res),
            nn.Linear(d_res, d_res // 2),
            nn.GELU(),
            nn.Linear(d_res // 2, 3),
        )

        # ── per-residue confidence head (predicted Cα-lDDT in [0, 1]) ────
        self.conf_head = nn.Sequential(
            nn.LayerNorm(d_res),
            nn.Linear(d_res, d_res // 2),
            nn.GELU(),
            nn.Linear(d_res // 2, 1),
        )

        # ── Distogram + contact aux heads (Stage 1 only) ─────────────────
        # Symmetrise the pair tensor internally; train the pair stack directly
        # on Cα-Cα distance / contact prediction — the strongest early signal
        # and the main way the (Mamba) trunk is pushed to encode long-range
        # tertiary contacts it cannot see by sequence scan alone.
        self.n_distogram_bins = 64
        self.distogram_head = nn.Sequential(
            nn.LayerNorm(d_pair),
            nn.Linear(d_pair, self.n_distogram_bins),
        )
        self.contact_head = nn.Sequential(
            nn.LayerNorm(d_pair),
            nn.Linear(d_pair, 1),
        )

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _chain_break(chain_id: Tensor, res_mask: Tensor) -> Tensor:
        """Per-residue flag — True at first residue of each chain visible in the crop."""
        prev = torch.cat(
            [torch.full_like(chain_id[:, :1], -1), chain_id[:, :-1]], dim=1,
        )
        return ((chain_id != prev) & res_mask).unsqueeze(-1)  # [B, L, 1]

    def _relpos_bin(self, batch: ProteinBatch) -> Tensor:
        diff = batch.res_seq_nums.unsqueeze(2) - batch.res_seq_nums.unsqueeze(1)
        same_chain = batch.chain_id.unsqueeze(2) == batch.chain_id.unsqueeze(1)
        return torch.where(
            same_chain,
            diff.clamp(-self.relpos_max, self.relpos_max) + self.relpos_max,
            torch.full_like(diff, self.n_relpos_bins - 1),
        )

    def _compose_residue_input(
        self, batch: ProteinBatch, x_t_ca: Tensor, plm: Tensor | None,
    ) -> Tensor:
        B, L = batch.res_type.shape
        dtype = x_t_ca.dtype

        ca_feat = self.ca_coord_embed(x_t_ca)                              # [B, L, d_ca_emb]
        t_feat = batch.t.squeeze(-1).squeeze(-1).expand(-1, L).unsqueeze(-1).to(dtype)
        terminus_feat = torch.stack(
            [batch.is_nterm.to(dtype), batch.is_cterm.to(dtype)], dim=-1,
        )                                                                  # [B, L, 2]
        chain_break_feat = self._chain_break(batch.chain_id, batch.res_mask).to(dtype)
        rt_feat = self.res_type_embed(batch.res_type)                      # [B, L, d_res_type]
        parts = [ca_feat, t_feat, terminus_feat, chain_break_feat, rt_feat]
        if self.seq_pos_embed is not None:
            pos_feat = self.seq_pos_embed(
                batch.res_seq_nums, batch.res_mask,
                chain_id=batch.chain_id, entity_id=batch.entity_id, sym_id=batch.sym_id,
            )
            parts.append(pos_feat)
        if plm is not None:
            parts.append(plm)
        trunk_in = torch.cat(parts, dim=-1)
        trunk_in = self.trunk_input_norm(trunk_in)
        return self.trunk_proj(trunk_in)

    def _recycle_dist_bin(self, ca: Tensor) -> Tensor:
        """Bin a Cα distance map (Å) for the recycle pair embedding. [B, L, L]."""
        d = torch.linalg.norm(ca.unsqueeze(2) - ca.unsqueeze(1), dim=-1) * COORD_SCALE
        bin_w = self.recycle_max_dist / self.n_recycle_bins
        return (d / bin_w).clamp(0, self.n_recycle_bins - 1).long()

    def _embed_plm(self, batch: ProteinBatch, dtype: torch.dtype) -> Tensor | None:
        """Project ESM3 features (loud failure if expected but missing)."""
        if not self.use_plm:
            return None
        if batch.esm is None:
            raise RuntimeError(
                "MambaFoldStage1 built with use_plm=True but batch.esm is None. "
                "Pre-compute ESM3 features (scripts/precompute_esm.py)."
            )
        if batch.esm.shape[-1] != self.d_plm:
            raise RuntimeError(f"Expected ESM dim {self.d_plm}, got {batch.esm.shape[-1]}.")
        return self.plm_proj(self.plm_norm(batch.esm.to(dtype=dtype)))       # [B, L, d_plm_proj]

    def _pair_and_heads(
        self, batch: ProteinBatch, res0: Tensor, plm: Tensor | None,
        recycle_ca: Tensor | None, return_aux: bool,
    ) -> dict:
        """One cycle: build pair from the (cached) trunk latent + optional
        recycled Cα distance, run the pair stack, then all output heads."""
        mask_f = batch.res_mask.unsqueeze(-1).to(res0.dtype)

        s_p = self.pair_init_single(res0)
        pair = s_p.unsqueeze(2) + s_p.unsqueeze(1)                           # [B, L, L, d_pair]
        if self.pair_init_esm is not None and plm is not None:
            e_p = self.pair_init_esm(plm)
            pair = pair + e_p.unsqueeze(2) + e_p.unsqueeze(1)
        pair = pair + self.relpos_embed(self._relpos_bin(batch))
        if recycle_ca is not None:
            pair = pair + self.recycle_dist_embed(self._recycle_dist_bin(recycle_ca))

        pair_mask = batch.res_mask.unsqueeze(2) & batch.res_mask.unsqueeze(1)  # [B, L, L]
        for blk in self.pair_blocks:
            pair = blk(pair, pair_mask)

        res = res0 + self.pair_to_single(pair, batch.res_mask)
        out = {
            "v_ca":         self.ca_head(res) * mask_f,
            "trunk_latent": res,
            "pcb_dir":      F.normalize(self.pcb_head(res), dim=-1) * mask_f,
            "conf":         torch.sigmoid(self.conf_head(res)).squeeze(-1) * batch.res_mask.to(res.dtype),
        }
        if return_aux:
            pair_sym = (pair + pair.transpose(1, 2)) / 2
            out["distogram_logits"] = self.distogram_head(pair_sym)         # [B,L,L,n_bins]
            out["contact_logits"] = self.contact_head(pair_sym).squeeze(-1)  # [B,L,L]
        return out

    # ── forward ─────────────────────────────────────────────────────────

    def forward(
        self, batch: ProteinBatch, return_aux: bool = False,
        n_cycles: int | None = None,
    ) -> dict:
        """
        Always returns a dict so the scaffold interface (Cα + orientation +
        confidence) is uniform across train and inference.

        Args:
            return_aux: also emit `distogram_logits`/`contact_logits` (final cycle).
            n_cycles: recycling iterations (defaults to `self.n_cycles`). Earlier
                cycles run under no_grad and only feed their predicted Cα distance
                map back into the pair init; the final cycle carries gradient.

        Returns keys:
            v_ca         [B, L, 3]
            trunk_latent [B, L, d_res]
            pcb_dir      [B, L, 3]  — unit pseudo-Cβ direction
            conf         [B, L]     — predicted Cα-lDDT in [0, 1]
            distogram_logits/contact_logits — only if return_aux
        """
        n_cycles = n_cycles if n_cycles is not None else self.n_cycles
        x_t_ca = batch.x_t[..., CA_ATOM_ID, :]                              # [B, L, 3]
        plm = self._embed_plm(batch, x_t_ca.dtype)

        # Residue trunk is recycle-independent → compute once and reuse.
        res0 = self._compose_residue_input(batch, x_t_ca, plm)
        res0 = self.residue_trunk(res0, batch.res_mask)

        one_minus_t = (1.0 - batch.t.squeeze(-1).squeeze(-1).squeeze(-1)).view(-1, 1, 1)
        recycle_ca = None
        for cycle in range(n_cycles):
            is_last = cycle == n_cycles - 1
            ctx = nullcontext() if is_last else torch.no_grad()
            with ctx:
                out = self._pair_and_heads(
                    batch, res0, plm, recycle_ca, return_aux and is_last,
                )
            if not is_last:
                # One-step Cα recon → recycle feature (stop-grad into next cycle).
                recycle_ca = (x_t_ca + one_minus_t * out["v_ca"]).detach()
        return out
