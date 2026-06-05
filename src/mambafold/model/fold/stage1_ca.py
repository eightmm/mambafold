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

import torch
import torch.nn as nn
from torch import Tensor

from mambafold.data.constants import AA_TO_ID, CA_ATOM_ID
from mambafold.data.types import ProteinBatch
from mambafold.model.bimamba3 import MambaStack
from mambafold.model.embeddings import (
    CoordinateFourierEmbedder,
    SequenceFourierEmbedder,
)
from mambafold.model.fold.pair_blocks import PairBlock, pair_to_single

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
    ):
        super().__init__()
        self.d_res = d_res
        self.d_pair = d_pair
        self.use_plm = use_plm
        self.d_plm = d_plm
        self.relpos_max = relpos_max
        self.n_relpos_bins = 2 * relpos_max + 2  # in-chain bins + OUT_OF_CHAIN

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
        )

        # ── Pair representation ─────────────────────────────────────────
        self.pair_init_single = nn.Linear(d_res, d_pair)
        if use_plm:
            self.pair_init_esm = nn.Linear(d_plm_proj, d_pair)
        else:
            self.pair_init_esm = None
        self.relpos_embed = nn.Embedding(self.n_relpos_bins, d_pair)

        self.pair_blocks = nn.ModuleList([
            PairBlock(
                d_pair=d_pair,
                n_heads=n_pair_heads,
                d_head=d_pair_head,
                mult_c=pair_mult_c,
            )
            for _ in range(n_pair_blocks)
        ])

        # pair → single bias (masked row mean projected back to d_res)
        self.pair_to_single_proj = nn.Linear(d_pair, d_res)

        # ── CA output head ──────────────────────────────────────────────
        self.ca_head = nn.Sequential(
            nn.LayerNorm(d_res),
            nn.Linear(d_res, d_res // 2),
            nn.GELU(),
            nn.Linear(d_res // 2, 3),
        )

        # ── Distogram aux head (Stage 1 only) ───────────────────────────
        # Symmetrises pair tensor internally; trains the pair stack directly
        # on Cα-Cα distance prediction, which is the strongest early signal.
        self.n_distogram_bins = 64
        self.distogram_head = nn.Sequential(
            nn.LayerNorm(d_pair),
            nn.Linear(d_pair, self.n_distogram_bins),
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

    # ── forward ─────────────────────────────────────────────────────────

    def forward(
        self, batch: ProteinBatch, return_aux: bool = False,
    ) -> tuple[Tensor, Tensor] | dict:
        """
        Args:
            return_aux: If True, returns a dict including distogram logits
                for auxiliary supervision during training. Otherwise returns
                the lean (v_ca, trunk_latent) tuple used at inference time.

        Returns (default):
            v_ca:         [B, L, 3]
            trunk_latent: [B, L, d_res]
        Returns (return_aux=True):
            dict with keys: v_ca, trunk_latent, distogram_logits ([B,L,L,n_bins])
        """
        # 1. Slice x_t^CA from the full batch
        x_t_ca = batch.x_t[..., CA_ATOM_ID, :]                              # [B, L, 3]

        # 2. PLM features (loud failure if expected but missing)
        plm = None
        if self.use_plm:
            if batch.esm is None:
                raise RuntimeError(
                    "MambaFoldStage1 built with use_plm=True but batch.esm is None. "
                    "Pre-compute ESM3 features (scripts/precompute_esm.py)."
                )
            if batch.esm.shape[-1] != self.d_plm:
                raise RuntimeError(
                    f"Expected ESM dim {self.d_plm}, got {batch.esm.shape[-1]}.",
                )
            esm = batch.esm.to(dtype=x_t_ca.dtype)
            plm = self.plm_proj(self.plm_norm(esm))                          # [B, L, d_plm_proj]

        # 3. Residue trunk
        res = self._compose_residue_input(batch, x_t_ca, plm)                # [B, L, d_res]
        res = self.residue_trunk(res, batch.res_mask)                        # [B, L, d_res]

        # 4. Pair construction (initial seed from residue trunk + ESM + relpos)
        s_p = self.pair_init_single(res)
        pair = s_p.unsqueeze(2) + s_p.unsqueeze(1)                           # [B, L, L, d_pair]
        if self.pair_init_esm is not None and plm is not None:
            e_p = self.pair_init_esm(plm)
            pair = pair + e_p.unsqueeze(2) + e_p.unsqueeze(1)
        pair = pair + self.relpos_embed(self._relpos_bin(batch))

        # 5. Pair stack
        pair_mask = batch.res_mask.unsqueeze(2) & batch.res_mask.unsqueeze(1)  # [B, L, L]
        for blk in self.pair_blocks:
            pair = blk(pair, pair_mask)

        # 6. Pair → single bias, then CA head
        res = res + pair_to_single(pair, batch.res_mask, self.pair_to_single_proj)
        v_ca = self.ca_head(res) * batch.res_mask.unsqueeze(-1).to(res.dtype)

        if not return_aux:
            return v_ca, res

        # Distogram aux logits — symmetrise pair tensor first.
        pair_sym = (pair + pair.transpose(1, 2)) / 2
        dist_logits = self.distogram_head(pair_sym)                             # [B,L,L,n_bins]
        return {
            "v_ca": v_ca,
            "trunk_latent": res,
            "distogram_logits": dist_logits,
        }
