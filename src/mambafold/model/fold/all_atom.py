"""Direct all-atom flow matching model.

Inputs:
    res_type, PLM, chain_id/entity_id/sym_id, res_seq_nums,
    is_nterm/is_cterm, noised atom-slot coordinates, t

Outputs:
    v_atom        [B, L, A, 3]   — all-atom FM velocity
    trunk_latent  [B, L, d_res]  — residue representation for downstream heads

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
pure SSM trunk suffices. The all-attention control trunk this once promised
was cancelled on cost, so nothing here is a controlled comparison against
attention. Confidence is trained as a separate phase against a frozen folding
model and is not part of this model.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
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
    """

    def __init__(
        self,
        d_res: int = 1024,
        n_trunk: int = 12,
        d_res_type: int = 32,
        d_res_pos: int = 64,
        d_plm: int = 1536,
        d_plm_proj: int = 256,
        # Width the FM time embedding is carried at. SimpleFold's DiTBlock
        # projects adaLN modulation from the full trunk width
        # (`Linear(hidden_size, 6 * hidden_size)`), so matching means d_temb =
        # d_res. It multiplies through every block, so it is a first-class part
        # of the parameter budget, not a side channel.
        d_temb: int = 128,
        d_ca_emb: int = 128,
        use_plm: bool = True,
        mimo_rank: int = 4,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 64,
        bidirectional: bool = True,
        bimamba_share: bool = False,
        d_atom: int = 128,
        n_atom_layers: int = 4,
        # The atom levels get their own SSM settings. Inheriting the trunk's is
        # what made `mimo_rank: 1` — chosen so the trunk can reach d_state 128
        # on an RTX 6000 Ada — set chunk_size 64 on a 14-slot sequence, so every
        # atom-level layer processed 64 positions to use 14.
        atom_mixer: str = "mamba",
        atom_d_state: int = 64,
        atom_mimo_rank: int = 2,
        # Cross-residue mixing on the backbone atom streams. 0 disables it and
        # leaves the atom levels strictly intra-residue.
        n_atom_cross_layers: int = 1,
        n_backbone_streams: int = 5,
        self_conditioning: bool = False,
    ):
        super().__init__()
        self.d_res = d_res
        self.use_plm = use_plm
        self.d_plm = d_plm
        self.self_conditioning = self_conditioning

        # ── Residue-side embedders ──────────────────────────────────────
        self.res_type_embed = nn.Embedding(NUM_RES_TYPES, d_res_type)
        self.seq_pos_embed = SequenceFourierEmbedder(d_out=d_res_pos) if d_res_pos > 0 else None
        # The noised structure enters through two embedders on two bands rather
        # than one embedder on one. A single absolute-coordinate embedding has
        # to cover both the whole-crop extent (a 1024-residue chain spans about
        # +-35 A about its centroid) and bond-scale detail (1.2-1.5 A), a range
        # of more than four octaves that no 16-band embedding spans at useful
        # resolution. Splitting them costs nothing and lets each band sit where
        # its quantity actually lives:
        #   ca_abs_embed   the residue's own CA, absolute        128 A -> 4 A
        #   atom_rel_embed the atom minus that CA, relative       16 A -> 1 A
        # The relative half is what makes side-chain geometry expressible at
        # all: as a difference of two absolute embeddings whose finest period
        # was 6.28 A, it was not.
        self.ca_abs_embed = CoordinateFourierEmbedder(
            d_out=d_ca_emb, min_period_A=4.0, max_period_A=128.0
        )
        self.atom_rel_embed = CoordinateFourierEmbedder(
            d_out=d_ca_emb, min_period_A=1.0, max_period_A=16.0, include_norm=True
        )
        if self_conditioning:
            self.self_cond_proj = nn.Linear(d_ca_emb, d_ca_emb, bias=False)
            nn.init.zeros_(self.self_cond_proj.weight)
        else:
            self.self_cond_proj = None

        # FM time/noise-level conditioning. A sinusoidal+MLP embedding of t is
        # broadcast into the trunk (FiLM on the trunk input) and the atom
        # encoder/decoder (FiLM inside), so every level knows the noise level —
        # the atom blocks previously saw no time signal at all.
        self.d_temb = d_temb
        self.time_embed = TimeEmbedding(d_temb)
        # Chain-length conditioning, added into the same embedding the AdaLN
        # blocks read. SimpleFold folds log(num_tokens) into its timestep
        # condition. The length must come from `res_mask.sum()`, never from the
        # padded axis: DDP pads every rank up to one global length, so a padded
        # length would make a protein's conditioning depend on which other
        # proteins its rank-mates happened to draw — and no such length exists
        # at inference.
        self.length_embed = nn.Sequential(
            nn.Linear(1, d_temb), nn.SiLU(), nn.Linear(d_temb, d_temb)
        )
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
            d_state=atom_d_state,
            mimo_rank=atom_mimo_rank,
            expand=expand,
            headdim=headdim,
            bimamba_share=bimamba_share,
            mixer=atom_mixer,
            n_cross_layers=n_atom_cross_layers,
            n_backbone_streams=n_backbone_streams,
        )

        if use_plm:
            self.plm_norm = nn.LayerNorm(d_plm)
            self.plm_proj = nn.Linear(d_plm, d_plm_proj)
            # Re-injected into the trunk's residual stream part-way down. The
            # PLM is the only evolutionary signal this model has, and a scan
            # cannot re-select it the way attention re-reads every position at
            # every layer: supplied once at the input, it has to survive every
            # residual block to still be usable at layer 10. Zero-init so the
            # trunk starts exactly as it would without it.
            self.plm_reinject = nn.Linear(d_plm_proj, d_res)
            nn.init.zeros_(self.plm_reinject.weight)
            nn.init.zeros_(self.plm_reinject.bias)
        else:
            self.plm_norm = None
            self.plm_proj = None
            self.plm_reinject = None
            d_plm_proj = 0  # contributes 0 to trunk input width

        # Trunk input = atom_token + ca_abs + termini(2) + chain_break(1)
        #             + res_type_emb + seq_pos_emb + plm_proj
        # (time enters via FiLM on the trunk input, not as a concat feature)
        #
        # `ca_abs` is the same embedding the atom encoder already computed, fed
        # to the trunk directly. Without it the trunk's only view of the noised
        # structure is the d_atom-wide token pooled over a residue's atoms —
        # 128 of 1251 input channels, against 1024 for the PLM, in a model
        # whose entire output is a coordinate update. It is a slice of an
        # existing tensor, so it costs one more block of `trunk_proj` and no
        # extra compute.
        trunk_in_dim = d_atom + d_ca_emb + 2 + 1 + d_res_type + d_res_pos + d_plm_proj
        self.trunk_input_norm = nn.LayerNorm(trunk_in_dim)
        self.trunk_proj = nn.Linear(trunk_in_dim, d_res)

        # ── Sequence trunk ──────────────────────────────────────────────
        # Roughly the one-third and two-thirds marks.
        self.plm_inject_at = tuple(sorted({n_trunk // 3, (2 * n_trunk) // 3})) if n_trunk else ()
        self.residue_trunk = MambaStack(
            d_res,
            n_trunk,
            d_state=d_state,
            mimo_rank=mimo_rank,
            expand=expand,
            headdim=headdim,
            bidirectional=bidirectional,
            d_temb=d_temb,
            bimamba_share=bimamba_share,
        )

        # ── all-atom output decoder ─────────────────────────────────────
        # BiMamba over atom slots, conditioned on the trunk residue latent and
        # skipping in the encoder's per-atom representation, → per-atom velocity.
        self.atom_decoder = AtomDecoder(
            d_res,
            d_atom,
            n_layers=n_atom_layers,
            d_temb=d_temb,
            d_state=atom_d_state,
            mimo_rank=atom_mimo_rank,
            expand=expand,
            headdim=headdim,
            bimamba_share=bimamba_share,
            mixer=atom_mixer,
            n_cross_layers=n_atom_cross_layers,
            n_backbone_streams=n_backbone_streams,
        )

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _chain_break(chain_id: Tensor, res_mask: Tensor) -> Tensor:
        """Per-residue flag — True at first residue of each chain visible in the crop."""
        prev = torch.cat(
            [torch.full_like(chain_id[:, :1], -1), chain_id[:, :-1]],
            dim=1,
        )
        return ((chain_id != prev) & res_mask).unsqueeze(-1)  # [B, L, 1]

    def _compose_residue_input(
        self,
        batch: ProteinBatch,
        coord_feat: Tensor,
        ca_feat: Tensor,
        plm: Tensor | None,
    ) -> Tensor:
        dtype = coord_feat.dtype

        terminus_feat = torch.stack(
            [batch.is_nterm.to(dtype), batch.is_cterm.to(dtype)],
            dim=-1,
        )  # [B, L, 2]
        chain_break_feat = self._chain_break(batch.chain_id, batch.res_mask).to(dtype)
        rt_feat = self.res_type_embed(batch.res_type)  # [B, L, d_res_type]
        parts = [coord_feat, ca_feat.to(dtype), terminus_feat, chain_break_feat, rt_feat]
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

    def _split_scale_embed(self, coords: Tensor, atom_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Two-band embedding of one coordinate set.

        Returns (per-atom [B, L, A, d_ca_emb], per-residue CA [B, L, d_ca_emb]).

        The per-atom feature is the sum of the coarse CA-absolute embedding and
        the fine CA-relative one. Summing two `Linear`-terminated embedders is
        exactly a single `Linear` over their concatenated raw features, so this
        is the concatenation without materialising a duplicate copy of the CA
        feature in every atom slot.
        """
        mask_f = atom_mask.unsqueeze(-1).to(coords.dtype)
        coords = coords * mask_f
        ca = coords[:, :, CA_ATOM_ID : CA_ATOM_ID + 1, :]  # [B, L, 1, 3]
        ca_feat = self.ca_abs_embed(ca.squeeze(2))  # [B, L, d_ca_emb]
        rel = (coords - ca) * mask_f  # [B, L, A, 3]
        return self.atom_rel_embed(rel) + ca_feat.unsqueeze(2), ca_feat

    def _atom_coord_embed(self, batch: ProteinBatch) -> tuple[Tensor, Tensor]:
        """Embed the noised coordinates.

        Returns (per-atom [B, L, A, d_ca_emb], per-residue CA [B, L, d_ca_emb]).
        The first feeds the atom encoder, the second also goes straight to the
        trunk input.
        """
        x_t = batch.x_t
        if x_t.shape[-2] != MAX_ATOMS_PER_RES:
            raise RuntimeError(
                f"Direct all-atom model expects {MAX_ATOMS_PER_RES} atom slots, "
                f"got {x_t.shape[-2]}."
            )
        coord_emb, ca_feat = self._split_scale_embed(x_t, batch.atom_mask)
        if self.self_cond_proj is not None:
            x_sc = batch.x_self_cond
            if x_sc is None:
                x_sc = torch.zeros_like(x_t)
                present = x_t.new_zeros((x_t.shape[0], 1, 1, 1))
            elif x_sc.shape != x_t.shape:
                raise RuntimeError(
                    f"x_self_cond shape {tuple(x_sc.shape)} != x_t shape {tuple(x_t.shape)}"
                )
            else:
                present = x_t.new_ones((x_t.shape[0], 1, 1, 1))
            # Keep the projection in every autograd graph when self-conditioning
            # is configured, even on batches where the Bernoulli draw is false.
            # The indicator makes the absent branch an exact semantic zero while
            # avoiding intermittent unused parameters under DDP.
            sc_emb, _ = self._split_scale_embed(
                x_sc.to(dtype=x_t.dtype), batch.atom_mask
            )
            coord_emb = coord_emb + self.self_cond_proj(sc_emb) * present
        return coord_emb, ca_feat

    def _embed_plm(self, batch: ProteinBatch, dtype: torch.dtype) -> Tensor | None:
        """Project external PLM features (loud failure if expected but missing)."""
        if not self.use_plm:
            return None
        if batch.esm is None:
            raise RuntimeError(
                "MambaFoldAllAtom built with use_plm=True but batch.esm is None. "
                "Pre-compute PLM features (scripts/precompute_esm.py)."
            )
        if batch.esm.ndim != 3 or batch.esm.shape[1] != batch.res_type.shape[1]:
            raise RuntimeError(
                "batch.esm must have shape [rows, L, d_plm] with the same L as res_type."
            )
        if batch.esm.shape[-1] != self.d_plm:
            raise RuntimeError(f"Expected ESM dim {self.d_plm}, got {batch.esm.shape[-1]}.")
        batch_size = batch.res_type.shape[0]
        rows = batch.esm.shape[0]
        if rows not in (1, batch_size):
            raise RuntimeError(
                f"batch.esm has {rows} rows; expected {batch_size} or 1 (shared across copies)."
            )
        # A one-row embedding means every example in the batch is the same
        # protein — the collator's `copies_per_protein` path. Normalize and
        # project once and broadcast the result: the projection is d_plm ->
        # d_plm_proj (2560 -> 256 here), by far the widest matmul outside the
        # trunk, and doing it per copy computes the same thing every time.
        # `expand` is differentiable; gradients from the copies sum back.
        plm = self.plm_proj(self.plm_norm(batch.esm.to(dtype=dtype)))
        if rows == 1 and batch_size > 1:
            plm = plm.expand(batch_size, -1, -1)
        return plm  # [B, L, d_plm_proj]

    def _decode(
        self,
        batch: ProteinBatch,
        res: Tensor,
        atom_repr: Tensor,
        temb: Tensor,
    ) -> dict:
        """Decode the folding velocity and expose the residue latent."""
        mask_atom = batch.atom_mask.to(res.dtype).unsqueeze(-1)
        v_atom = self.atom_decoder(res, atom_repr, batch, temb) * mask_atom
        return {
            "v_atom": v_atom,
            "trunk_latent": res,
        }

    # ── forward ─────────────────────────────────────────────────────────

    def forward(
        self,
        batch: ProteinBatch,
    ) -> dict:
        """Run the single folding path used by training and sampling.

        Returns keys:
            v_atom       [B, L, A, 3]
            trunk_latent [B, L, d_res]
        """
        temb = self.time_embed(batch.t)  # [B, d_temb]
        n_res = batch.res_mask.sum(dim=1, keepdim=True).clamp(min=1)
        temb = temb + self.length_embed(torch.log(n_res.to(temb.dtype)))
        coord_emb, ca_feat = self._atom_coord_embed(batch)  # [B,L,A,d], [B,L,d]
        atom_token, atom_repr = self.atom_encoder(coord_emb, batch, temb)  # token [B,L,d_atom]
        plm = self._embed_plm(batch, atom_token.dtype)

        res = self._compose_residue_input(batch, atom_token, ca_feat, plm)
        res = self.film_trunk(res, temb)  # inject noise level
        reinject = (
            self.plm_reinject(plm) * batch.res_mask.unsqueeze(-1).to(res.dtype)
            if (self.plm_reinject is not None and plm is not None)
            else None
        )
        res = self.residue_trunk(
            res, batch.res_mask, temb, inject=reinject, inject_at=self.plm_inject_at
        )
        return self._decode(batch, res, atom_repr, temb)
