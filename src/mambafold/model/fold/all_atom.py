"""Direct all-atom flow matching model.

Inputs:
    res_type, PLM, chain_id/entity_id/sym_id, res_seq_nums,
    is_nterm/is_cterm, noised atom-slot coordinates, t

Outputs:
    v_atom        [B, L, A, 3]   — all-atom FM velocity
    v_ca          [B, L, 3]      — CA velocity view from v_atom
    trunk_latent  [B, L, d_res]  — residue representation for aux heads

The model is intentionally single-path: no separate coarse path and no
recycling loop. The mainline reasons at three SSM levels — atom → token → atom:
an atom-level BiMamba encoder pools each residue's atoms into a token, a
pair-free BiMamba residue trunk does global inter-residue reasoning, and an
atom-level BiMamba decoder reads the residue latent back out into per-atom
velocities (see `atom_mamba.py`).

There is no attention and no pair stack, and neither is reachable by config.
Both were removed rather than switched off: an O(L²) triangle-update path that
merely defaults to disabled still has to be carried, sized and reasoned about,
and a trunk that *could* be given attention cannot support the claim that a
pure SSM trunk suffices. An all-attention control arm is a separate trunk,
written for the purpose.

A lightweight pair-free aux head can still emit distogram/contact logits — it
builds an outer sum from the residue latent and never feeds anything back into
the trunk, so it is an output head, not a pair representation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mambafold.data.constants import AA_TO_ID, CA_ATOM_ID, MAX_ATOMS_PER_RES
from mambafold.data.types import ProteinBatch
from mambafold.model.bimamba3 import MambaStack
from mambafold.model.embeddings import (
    CoordinateFourierEmbedder,
    SequenceFourierEmbedder,
)
from mambafold.model.fold.atom_mamba import AtomDecoder, AtomEncoder, FiLM

NUM_RES_TYPES = len(AA_TO_ID)  # 21 (20 AAs + UNK)


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding of the FM time t ∈ [0, 1] → MLP → [B, d_out].

    Replaces the previous single-scalar `t` feature so the noise level is a rich
    vector the trunk and atom blocks can be modulated by (see FiLM).
    """

    def __init__(self, d_out: int, n_freqs: int = 64):
        super().__init__()
        self.register_buffer(
            "freqs",
            torch.exp(torch.linspace(0.0, math.log(1000.0), n_freqs)),
            persistent=False,
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_freqs, d_out),
            nn.SiLU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, t: Tensor) -> Tensor:
        t = t.reshape(t.shape[0], 1).to(self.freqs.dtype)  # [B, 1]
        ang = t * self.freqs  # [B, n_freqs]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.mlp(emb)  # [B, d_out]


class MambaFoldAllAtom(nn.Module):
    """Direct all-atom flow matching model.

    Args:
        d_res: Residue token dim. Default 1024.
        n_trunk: Number of BiMamba3 layers in the trunk. Default 12.
        d_res_type: Residue-type embedding dim. Default 32.
        d_res_pos: Sequence position (chain/entity/sym + Fourier) embed dim.
        d_plm: External PLM embedding dim. Default 1536.
        d_plm_proj: Internal PLM projection width. Default 256.
        d_ca_emb: Fourier embed dim of per-atom x_t scalar coords. Default 128.
        use_plm: If True, external PLM features are required (loud error otherwise).
        # SSM
        mimo_rank: Mamba MIMO rank. Default 4.
        d_state: Mamba state dim. Default 64.
        expand: Mamba inner-dim multiplier. Default 2.
        headdim: Mamba head dim. Default 64.
        bidirectional: BiMamba (default True) vs causal Mamba.
        # Pair-free aux heads (off by default; nothing feeds back into the trunk)
        pairfree_aux_heads: Build distogram/contact logits from the residue latent.
        d_aux_pair: Width of that outer sum. Default 192.
        relpos_max: AF3 relpos clip range. Default 32 (gives 65 + OUT_OF_CHAIN bins).
            Only used by the aux heads.
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
        mimo_rank: int = 4,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 64,
        bidirectional: bool = True,
        relpos_max: int = 32,
        trunk_time_film: bool = False,
        trunk_adaln_zero: bool = False,
        bimamba_share: bool = False,
        d_atom: int = 128,
        n_atom_layers: int = 4,
        self_conditioning: bool = False,
        pairfree_aux_heads: bool = False,
        d_aux_pair: int = 192,
    ):
        super().__init__()
        self.d_res = d_res
        self.use_plm = use_plm
        self.d_plm = d_plm
        self.pairfree_aux_heads = pairfree_aux_heads
        self.self_conditioning = self_conditioning
        self.relpos_max = relpos_max
        self.n_relpos_bins = 2 * relpos_max + 2  # in-chain bins + OUT_OF_CHAIN

        # ── Residue-side embedders ──────────────────────────────────────
        self.res_type_embed = nn.Embedding(NUM_RES_TYPES, d_res_type)
        self.seq_pos_embed = SequenceFourierEmbedder(d_out=d_res_pos) if d_res_pos > 0 else None
        self.ca_coord_embed = CoordinateFourierEmbedder(d_out=d_ca_emb)
        if self_conditioning:
            self.self_cond_proj = nn.Linear(d_ca_emb, d_ca_emb, bias=False)
            nn.init.zeros_(self.self_cond_proj.weight)
        else:
            self.self_cond_proj = None

        # FM time/noise-level conditioning. A sinusoidal+MLP embedding of t is
        # broadcast into the trunk (FiLM on the trunk input) and the atom
        # encoder/decoder (FiLM inside), so every level knows the noise level —
        # the atom blocks previously saw no time signal at all.
        d_temb = 128
        self.time_embed = TimeEmbedding(d_temb)
        self.film_trunk = FiLM(d_res, d_temb)

        # Atom-level encoder: BiMamba over each residue's atom slots, then pool to
        # a residue token. Replaces a flat atom-coordinate projection so the trunk
        # token already carries intra-residue (side-chain) structure. Reuses the
        # trunk's SSM hyperparameters.
        self.atom_encoder = AtomEncoder(
            d_atom,
            d_ca_emb,
            n_layers=n_atom_layers,
            d_temb=d_temb,
            d_state=d_state,
            mimo_rank=mimo_rank,
            expand=expand,
            headdim=headdim,
            bimamba_share=bimamba_share,
        )

        if use_plm:
            self.plm_norm = nn.LayerNorm(d_plm)
            self.plm_proj = nn.Linear(d_plm, d_plm_proj)
        else:
            self.plm_norm = None
            self.plm_proj = None
            d_plm_proj = 0  # contributes 0 to trunk input width

        # Trunk input = atom_token + termini(2) + chain_break(1)
        #             + res_type_emb + seq_pos_emb + plm_proj
        # (time enters via FiLM on the trunk input, not as a concat feature)
        trunk_in_dim = d_atom + 2 + 1 + d_res_type + d_res_pos + d_plm_proj
        self.trunk_input_norm = nn.LayerNorm(trunk_in_dim)
        self.trunk_proj = nn.Linear(trunk_in_dim, d_res)

        # ── Sequence trunk ──────────────────────────────────────────────
        self.residue_trunk = MambaStack(
            d_res,
            n_trunk,
            d_state=d_state,
            mimo_rank=mimo_rank,
            expand=expand,
            headdim=headdim,
            bidirectional=bidirectional,
            layerwise_time_film=trunk_time_film,
            d_temb=d_temb,
            adaln_zero=trunk_adaln_zero,
            bimamba_share=bimamba_share,
        )

        # ── Pair-free aux heads ─────────────────────────────────────────
        # An outer sum of two projections of the residue latent, plus a relpos
        # embedding. Materializes [B, L, L, d_aux_pair] only when asked for, and
        # only on the way out — the trunk never sees it.
        if pairfree_aux_heads:
            self.pairfree_left = nn.Linear(d_res, d_aux_pair)
            self.pairfree_right = nn.Linear(d_res, d_aux_pair)
            self.pairfree_relpos = nn.Embedding(self.n_relpos_bins, d_aux_pair)
        else:
            self.pairfree_left = None
            self.pairfree_right = None
            self.pairfree_relpos = None

        # ── all-atom output decoder ─────────────────────────────────────
        # BiMamba over atom slots, conditioned on the trunk residue latent and
        # skipping in the encoder's per-atom representation, → per-atom velocity.
        self.atom_decoder = AtomDecoder(
            d_res,
            d_atom,
            n_layers=n_atom_layers,
            d_temb=d_temb,
            d_state=d_state,
            mimo_rank=mimo_rank,
            expand=expand,
            headdim=headdim,
            bimamba_share=bimamba_share,
        )

        # ── pseudo-Cβ direction head (side-chain orientation aux)
        self.pcb_head = nn.Sequential(
            nn.LayerNorm(d_res),
            nn.Linear(d_res, d_res // 2),
            nn.GELU(),
            nn.Linear(d_res // 2, 3),
        )

        # Legacy scalar confidence auxiliary retained for checkpoint compatibility.
        # The active ESMC-6B run uses w_conf=0, so this output is untrained and
        # uncalibrated; it must not be reported as pLDDT. The supported pLDDT path
        # uses a separately trained frozen-folding PLDDTHead.
        self.conf_head = nn.Sequential(
            nn.LayerNorm(d_res),
            nn.Linear(d_res, d_res // 2),
            nn.GELU(),
            nn.Linear(d_res // 2, 1),
        )

        # ── Distogram + contact aux heads ────────────────────────────────
        self.n_distogram_bins = 64
        if pairfree_aux_heads:
            self.distogram_head = nn.Sequential(
                nn.LayerNorm(d_aux_pair),
                nn.Linear(d_aux_pair, self.n_distogram_bins),
            )
            self.contact_head = nn.Sequential(
                nn.LayerNorm(d_aux_pair),
                nn.Linear(d_aux_pair, 1),
            )
        else:
            self.distogram_head = None
            self.contact_head = None

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _chain_break(chain_id: Tensor, res_mask: Tensor) -> Tensor:
        """Per-residue flag — True at first residue of each chain visible in the crop."""
        prev = torch.cat(
            [torch.full_like(chain_id[:, :1], -1), chain_id[:, :-1]],
            dim=1,
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
        self,
        batch: ProteinBatch,
        coord_feat: Tensor,
        plm: Tensor | None,
    ) -> Tensor:
        B, L = batch.res_type.shape
        dtype = coord_feat.dtype

        terminus_feat = torch.stack(
            [batch.is_nterm.to(dtype), batch.is_cterm.to(dtype)],
            dim=-1,
        )  # [B, L, 2]
        chain_break_feat = self._chain_break(batch.chain_id, batch.res_mask).to(dtype)
        rt_feat = self.res_type_embed(batch.res_type)  # [B, L, d_res_type]
        parts = [coord_feat, terminus_feat, chain_break_feat, rt_feat]
        if self.seq_pos_embed is not None:
            pos_feat = self.seq_pos_embed(
                batch.res_seq_nums,
                batch.res_mask,
                chain_id=batch.chain_id,
                entity_id=batch.entity_id,
                sym_id=batch.sym_id,
            )
            parts.append(pos_feat)
        if plm is not None:
            parts.append(plm)
        trunk_in = torch.cat(parts, dim=-1)
        trunk_in = self.trunk_input_norm(trunk_in)
        return self.trunk_proj(trunk_in)

    def _atom_coord_embed(self, batch: ProteinBatch) -> Tensor:
        """Fourier-embed the noised per-atom coordinates: [B, L, A, d_ca_emb].

        Fed to the atom encoder (the trunk token is its pooled output).
        """
        x_t = batch.x_t
        if x_t.shape[-2] != MAX_ATOMS_PER_RES:
            raise RuntimeError(
                f"Direct all-atom model expects {MAX_ATOMS_PER_RES} atom slots, "
                f"got {x_t.shape[-2]}."
            )
        atom_mask = batch.atom_mask.unsqueeze(-1).to(x_t.dtype)
        coord_emb = self.ca_coord_embed(x_t * atom_mask)
        if self.self_cond_proj is not None:
            x_sc = batch.x_self_cond
            if x_sc is None:
                sc_emb = torch.zeros_like(coord_emb)
            elif x_sc.shape != x_t.shape:
                raise RuntimeError(
                    f"x_self_cond shape {tuple(x_sc.shape)} != x_t shape {tuple(x_t.shape)}"
                )
            else:
                sc_emb = self.self_cond_proj(
                    self.ca_coord_embed(x_sc.to(dtype=x_t.dtype) * atom_mask)
                )
            coord_emb = coord_emb + sc_emb
        return coord_emb

    def _embed_plm(self, batch: ProteinBatch, dtype: torch.dtype) -> Tensor | None:
        """Project external PLM features (loud failure if expected but missing)."""
        if not self.use_plm:
            return None
        if batch.esm is None:
            raise RuntimeError(
                "MambaFoldAllAtom built with use_plm=True but batch.esm is None. "
                "Pre-compute PLM features (scripts/precompute_esm.py)."
            )
        if batch.esm.shape[-1] != self.d_plm:
            raise RuntimeError(f"Expected ESM dim {self.d_plm}, got {batch.esm.shape[-1]}.")
        return self.plm_proj(self.plm_norm(batch.esm.to(dtype=dtype)))  # [B, L, d_plm_proj]

    def _heads(
        self,
        batch: ProteinBatch,
        res: Tensor,
        atom_repr: Tensor,
        temb: Tensor,
        return_aux: bool,
    ) -> dict:
        """Run every output head on the trunk latent.

        The trunk latent goes straight to the heads: there is no pair stack to
        fold back into it, so `res` is exactly what the trunk produced.
        """
        mask_f = batch.res_mask.unsqueeze(-1).to(res.dtype)
        mask_atom = batch.atom_mask.to(res.dtype).unsqueeze(-1)
        v_atom = self.atom_decoder(res, atom_repr, batch, temb) * mask_atom
        v_ca = v_atom[:, :, CA_ATOM_ID, :] * mask_f

        out = {
            "v_ca": v_ca,
            "v_atom": v_atom,
            "trunk_latent": res,
            "pcb_dir": F.normalize(self.pcb_head(res), dim=-1) * mask_f,
            "conf": torch.sigmoid(self.conf_head(res)).squeeze(-1) * batch.res_mask.to(res.dtype),
        }
        if return_aux and self.pairfree_left is not None:
            # [B, L, L, d_aux_pair] is built here and nowhere else, so the O(L²)
            # allocation is confined to runs that ask for the aux logits.
            aux_pair = (
                self.pairfree_left(res).unsqueeze(2)
                + self.pairfree_right(res).unsqueeze(1)
                + self.pairfree_relpos(self._relpos_bin(batch))
            )
            pair_sym = (aux_pair + aux_pair.transpose(1, 2)) / 2
            out["distogram_logits"] = self.distogram_head(pair_sym)  # [B,L,L,n_bins]
            out["contact_logits"] = self.contact_head(pair_sym).squeeze(-1)  # [B,L,L]
        return out

    # ── forward ─────────────────────────────────────────────────────────

    def forward(
        self,
        batch: ProteinBatch,
        return_aux: bool = False,
    ) -> dict:
        """
        Always returns a dict so train/eval/inference share one interface.

        Args:
            return_aux: also emit `distogram_logits`/`contact_logits`.

        Returns keys:
            v_atom       [B, L, A, 3]
            v_ca         [B, L, 3]
            trunk_latent [B, L, d_res]
            pcb_dir      [B, L, 3]  — unit pseudo-Cβ direction
            conf         [B, L]     — legacy, uncalibrated scalar in [0, 1]
            distogram_logits/contact_logits — only if return_aux
        """
        temb = self.time_embed(batch.t)  # [B, d_temb]
        coord_emb = self._atom_coord_embed(batch)  # [B, L, A, d_ca_emb]
        atom_token, atom_repr = self.atom_encoder(coord_emb, batch, temb)  # token [B,L,d_atom]
        plm = self._embed_plm(batch, atom_token.dtype)

        res = self._compose_residue_input(batch, atom_token, plm)
        res = self.film_trunk(res, temb)  # inject noise level
        res = self.residue_trunk(res, batch.res_mask, temb)
        return self._heads(batch, res, atom_repr, temb, return_aux)
