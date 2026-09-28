"""Atom-level BiMamba encoder/decoder (intra-residue SSM).

The direct all-atom model reasons at three levels, all SSM-based so the
"MambaFold" identity holds end to end:

    atom  →  AtomEncoder (BiMamba over the A atom slots of each residue) → pool
    token →  pair-free residue trunk (Bi-Mamba, no attention)           ← global
    atom  →  AtomDecoder (BiMamba over the A atom slots) → per-atom velocity

Atom attention is intentionally avoided: the canonical atom ordering within a
residue (N, CA, C, O, CB, CG, …) gives a meaningful 1-D scan, the per-residue
sequence is tiny (A = MAX_ATOMS_PER_RES = 14), and a masked SSM scan never
produces NaNs on fully-padded rows (unlike a softmax over an all-masked set).
Inter-residue reasoning is the token trunk's job; these blocks only own
intra-residue (side-chain) geometry.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mambafold.data.constants import ATOM_NAME_TO_ID, MAX_ATOMS_PER_RES, NUM_PAIR_TYPES
from mambafold.data.types import ProteinBatch
from mambafold.model.bimamba3 import AdaLNZero, MambaStack, SwiGLU

NUM_ATOM_TYPES = len(ATOM_NAME_TO_ID)  # 37 (36 atom names + PAD)


class FiLM(nn.Module):
    """Feature-wise linear modulation from a time/noise-level embedding.

    h ← (1 + γ(temb))·h + β(temb). Zero-initialised so it starts as identity and
    does not disturb early training. Used to inject the FM noise level into blocks
    that otherwise never see `t` (the atom encoder/decoder).
    """

    def __init__(self, d_feat: int, d_temb: int):
        super().__init__()
        self.proj = nn.Linear(d_temb, 2 * d_feat)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        """x: [B, ..., d_feat], temb: [B, d_temb]."""
        scale, shift = self.proj(temb).chunk(2, dim=-1)  # [B, d_feat] each
        while scale.dim() < x.dim():  # broadcast over L (and A)
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        return x * (1 + scale.to(x.dtype)) + shift.to(x.dtype)


class AtomTupleMixer(nn.Module):
    """MLP-Mixer over the atom slots of one residue.

    The A slots are not a sequence. Slots 0-3 are literally N, CA, C, O in every
    residue and slot 4 is CB in every residue that has one; the rest are the
    side chain in canonical order, so a slot index is a fixed label, not a
    position in a stream. A scan pays for order-dependence this data does not
    have, and pays twice more: the Mamba-3 kernel rounds the sequence up to its
    chunk size, so A=14 is processed as 16 or 64 depending on `mimo_rank`.

    This mixes slots with a fixed A x A map — every slot reaches every other in
    one layer, where a bidirectional A-step scan needs a full pass each way —
    followed by a SwiGLU over channels shared across slots. Both are plain
    GEMMs, so there is no chunk padding and no batch of 16k tiny sequences.

    What is given up is content-adaptive routing: a selective SSM can decide
    which slots to read per example, this cannot. That matters when the mapping
    from slot to meaning is ambiguous, and here it is not — it is a
    deterministic function of the residue type, which the atom-type and
    pair-type embeddings already carry.
    """

    def __init__(self, d_atom: int, n_slots: int, n_layers: int, expand: int = 2,
                 d_temb: int = 128):
        super().__init__()
        hidden = expand * d_atom
        self.adaln = nn.ModuleList(
            AdaLNZero(d_atom, d_temb, n_branches=2) for _ in range(n_layers)
        )
        self.slot_norms = nn.ModuleList(nn.LayerNorm(d_atom) for _ in range(n_layers))
        self.slot_mix = nn.ModuleList(nn.Linear(n_slots, n_slots) for _ in range(n_layers))
        self.chan_norms = nn.ModuleList(nn.LayerNorm(d_atom) for _ in range(n_layers))
        self.chan_mix = nn.ModuleList(SwiGLU(d_atom, hidden) for _ in range(n_layers))

    def forward(
        self, x: Tensor, mask: Tensor, temb: Tensor, temb_repeat: int = 1
    ) -> Tensor:
        """x: [N, A, d], mask: [N, A] bool. Returns [N, A, d] with padding zeroed."""
        m = mask.unsqueeze(-1).to(x.dtype)
        x = x * m
        for adaln, slot_norm, slot_mix, chan_norm, chan_mix in zip(
            self.adaln, self.slot_norms, self.slot_mix, self.chan_norms, self.chan_mix
        ):
            # Zero the padded slots before mixing so they contribute nothing,
            # and again after so the block's output stays exactly zero there.
            h, gate = adaln(slot_norm(x), temb, branch=0, repeat=temb_repeat)
            h = (h * m).transpose(1, 2)                     # [N, d, A]
            x = (x + gate * slot_mix(h).transpose(1, 2)) * m
            h, gate = adaln(chan_norm(x), temb, branch=1, repeat=temb_repeat)
            x = (x + gate * chan_mix(h)) * m
        return x


def _atom_mixer(
    d_atom: int,
    n_layers: int,
    *,
    kind: str,
    n_slots: int,
    d_temb: int,
    d_state: int,
    mimo_rank: int,
    expand: int,
    headdim: int,
    bimamba_share: bool,
):
    if kind == "mlp":
        return AtomTupleMixer(d_atom, n_slots, n_layers, expand=expand, d_temb=d_temb)
    if kind != "mamba":
        raise ValueError(f"unknown atom mixer: {kind!r}")
    return MambaStack(
        d_atom,
        n_layers,
        d_state=d_state,
        mimo_rank=mimo_rank,
        expand=expand,
        headdim=headdim,
        bidirectional=True,
        bimamba_share=bimamba_share,
        d_temb=d_temb,
    )


class BackboneStreamMixer(nn.Module):
    """BiMamba along the residue axis, on the backbone atom streams.

    Everything else at the atom level is strictly intra-residue: the slot mixer
    only sees the A slots of one residue, so no atom ever meets an atom of its
    neighbour. The peptide bond C(i)-N(i+1) — the thing that makes a chain a
    chain — is invisible until the trunk has already collapsed each residue to a
    single token. SimpleFold does not have this hole: its atom encoder and
    decoder run windowed attention (32 queries, 128 keys) over the whole flat
    atom sequence, so neighbouring residues' atoms mix before pooling.

    This restores that with a scan instead of attention, and only on the first
    `n_streams` slots — N, CA, C, O, CB, which are the same atoms in every
    residue. Backbone continuity and secondary structure live there; the
    side-chain tips do not need a cross-residue path. Cost is O(L * n_streams)
    at atom width, with no pair tensor.
    """

    def __init__(self, d_atom: int, n_layers: int, n_streams: int, *, d_temb: int,
                 d_state: int, mimo_rank: int, expand: int, headdim: int,
                 bimamba_share: bool):
        super().__init__()
        self.n_streams = n_streams
        self.stack = MambaStack(
            d_atom, n_layers, d_state=d_state, mimo_rank=mimo_rank, expand=expand,
            headdim=headdim, bidirectional=True, bimamba_share=bimamba_share,
            d_temb=d_temb,
        )

    def forward(self, a: Tensor, res_mask: Tensor, temb: Tensor) -> Tensor:
        """a: [B, L, A, d], res_mask: [B, L]. Returns a with backbone slots mixed."""
        B, L, A, d = a.shape
        k = min(self.n_streams, A)
        # [B, L, k, d] -> [B*k, L, d]; row order is (b0s0..b0s{k-1}, b1s0, ...),
        # which is exactly what `repeat_interleave(k)` produces for temb.
        streams = a[:, :, :k, :].permute(0, 2, 1, 3).reshape(B * k, L, d)
        mask = res_mask.unsqueeze(1).expand(B, k, L).reshape(B * k, L)
        out = self.stack(streams, mask, temb, k)
        out = out.reshape(B, k, L, d).permute(0, 2, 1, 3)
        return torch.cat([a[:, :, :k, :] + out, a[:, :, k:, :]], dim=2)


class AtomEncoder(nn.Module):
    """BiMamba over atom slots, then a gated masked-mean pool to a residue token.

    Args:
        d_atom: Atom-token width.
        d_ca_emb: Width of the per-atom Fourier coordinate embedding fed in.
        n_layers: Number of BiMamba layers over the atom axis.
    """

    def __init__(
        self,
        d_atom: int,
        d_ca_emb: int,
        n_layers: int = 2,
        *,
        d_temb: int = 128,
        d_state: int = 64,
        mimo_rank: int = 4,
        expand: int = 2,
        headdim: int = 64,
        bimamba_share: bool = False,
        mixer: str = "mamba",
        n_slots: int = MAX_ATOMS_PER_RES,
        n_cross_layers: int = 1,
        n_backbone_streams: int = 5,
    ):
        super().__init__()
        self.coord_proj = nn.Linear(d_ca_emb, d_atom)
        self.pair_type_embed = nn.Embedding(NUM_PAIR_TYPES, d_atom)
        self.atom_type_embed = nn.Embedding(NUM_ATOM_TYPES, d_atom)
        self.in_norm = nn.LayerNorm(d_atom)
        self.film = FiLM(d_atom, d_temb)
        self.mamba = _atom_mixer(
            d_atom,
            n_layers,
            kind=mixer,
            n_slots=n_slots,
            d_temb=d_temb,
            d_state=d_state,
            mimo_rank=mimo_rank,
            expand=expand,
            headdim=headdim,
            bimamba_share=bimamba_share,
        )
        self.cross = (
            BackboneStreamMixer(
                d_atom, n_cross_layers, n_backbone_streams, d_temb=d_temb,
                d_state=d_state, mimo_rank=mimo_rank, expand=expand,
                headdim=headdim, bimamba_share=bimamba_share,
            )
            if n_cross_layers > 0
            else None
        )
        self.pool_gate = nn.Linear(d_atom, 1)
        # A softmax gate alone can collapse onto one atom — CA is the obvious
        # attractor — which would starve the residue token of side-chain
        # information even though the decoder still gets `atom_repr` as a skip.
        # Concatenating a plain masked mean gives the token a path that cannot
        # collapse, and the projection lets the model decide how much of each
        # to keep.
        self.pool_proj = nn.Linear(2 * d_atom, d_atom)
        self.out_norm = nn.LayerNorm(d_atom)

    def forward(
        self, coord_emb: Tensor, batch: ProteinBatch, temb: Tensor
    ) -> tuple[Tensor, Tensor]:
        """coord_emb: [B, L, A, d_ca_emb], temb: [B, d_temb].
        Returns (token [B,L,d_atom], atom_repr [B,L,A,d_atom])."""
        B, L, A, _ = coord_emb.shape
        a = self.coord_proj(coord_emb)
        a = (
            a
            + self.pair_type_embed(batch.pair_type).to(a.dtype)
            + self.atom_type_embed(batch.atom_type).to(a.dtype)
        )
        a = self.film(self.in_norm(a), temb)

        m = batch.atom_mask  # [B, L, A] bool
        a = self.mamba(a.reshape(B * L, A, -1), m.reshape(B * L, A), temb, L)
        a = a.reshape(B, L, A, -1)
        if self.cross is not None:
            a = self.cross(a, batch.res_mask, temb) * m.unsqueeze(-1).to(a.dtype)

        # Gated masked-mean pool over atoms → residue token. Fully-padded
        # residues softmax to zero weight (nan_to_num), so the token is 0.
        w = self.pool_gate(a).squeeze(-1).masked_fill(~m, float("-inf"))  # [B, L, A]
        w = torch.nan_to_num(torch.softmax(w, dim=-1))
        gated = (a * w.unsqueeze(-1)).sum(dim=2)  # [B, L, d_atom]
        mf = m.unsqueeze(-1).to(a.dtype)
        mean = (a * mf).sum(dim=2) / mf.sum(dim=2).clamp(min=1.0)
        tok = self.pool_proj(torch.cat([gated, mean], dim=-1))
        return self.out_norm(tok), a


class AtomDecoder(nn.Module):
    """Broadcast the residue latent onto atoms, BiMamba over atom slots → velocity.

    Conditions on the trunk's residue latent (global reasoning) and skips in the
    encoder's per-atom representation (intra-residue identity/geometry).
    """

    def __init__(
        self,
        d_res: int,
        d_atom: int,
        n_layers: int = 2,
        *,
        d_temb: int = 128,
        d_state: int = 64,
        mimo_rank: int = 4,
        expand: int = 2,
        headdim: int = 64,
        bimamba_share: bool = False,
        mixer: str = "mamba",
        n_slots: int = MAX_ATOMS_PER_RES,
        n_cross_layers: int = 1,
        n_backbone_streams: int = 5,
    ):
        super().__init__()
        self.n_slots = n_slots
        # One projection per atom slot, not one projection broadcast to all of
        # them. The trunk does every bit of this model's global reasoning at
        # d_res, and the atoms are where that reasoning has to turn into
        # coordinates; a single `Linear(d_res, d_atom)` handed all A slots the
        # same vector and left the slot mixer to tell them apart with a fixed
        # A x A map that cannot vary per channel. Side-chain placement is a
        # function of the residue's environment, which is exactly what only the
        # trunk latent knows. Cost is A x d_res x d_atom.
        self.ctx_proj = nn.Sequential(
            nn.LayerNorm(d_res), nn.Linear(d_res, n_slots * d_atom)
        )
        self.in_norm = nn.LayerNorm(d_atom)
        self.film = FiLM(d_atom, d_temb)
        self.mamba = _atom_mixer(
            d_atom,
            n_layers,
            kind=mixer,
            n_slots=n_slots,
            d_temb=d_temb,
            d_state=d_state,
            mimo_rank=mimo_rank,
            expand=expand,
            headdim=headdim,
            bimamba_share=bimamba_share,
        )
        self.cross = (
            BackboneStreamMixer(
                d_atom, n_cross_layers, n_backbone_streams, d_temb=d_temb,
                d_state=d_state, mimo_rank=mimo_rank, expand=expand,
                headdim=headdim, bimamba_share=bimamba_share,
            )
            if n_cross_layers > 0
            else None
        )
        # SimpleFold's FinalLayer: a time-conditioned AdaLN with scale and shift
        # only — no gate — over an unaffine LayerNorm, and a zero-initialised
        # output projection. The zero init is the point: the flow field is
        # exactly zero at step 0, so training starts from "predict no motion"
        # rather than from a random velocity on every atom.
        self.out_norm = nn.LayerNorm(d_atom, elementwise_affine=False, eps=1e-6)
        self.out_mod = nn.Linear(d_temb, 2 * d_atom)
        nn.init.zeros_(self.out_mod.weight)
        nn.init.zeros_(self.out_mod.bias)
        self.out = nn.Linear(d_atom, 3)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self, res_latent: Tensor, atom_repr: Tensor, batch: ProteinBatch, temb: Tensor
    ) -> Tensor:
        """res_latent: [B,L,d_res], atom_repr: [B,L,A,d_atom], temb: [B,d_temb].
        Returns v_atom [B,L,A,3]."""
        B, L, A, _ = atom_repr.shape
        if A > self.n_slots:
            raise RuntimeError(f"AtomDecoder built for {self.n_slots} slots, got {A}.")
        ctx = self.ctx_proj(res_latent).view(B, L, self.n_slots, -1)[:, :, :A, :]
        a = self.film(self.in_norm(atom_repr + ctx), temb)
        m = batch.atom_mask
        a = self.mamba(a.reshape(B * L, A, -1), m.reshape(B * L, A), temb, L)
        a = a.reshape(B, L, A, -1)
        if self.cross is not None:
            a = self.cross(a, batch.res_mask, temb) * m.unsqueeze(-1).to(a.dtype)
        scale, shift = self.out_mod(F.silu(temb)).chunk(2, dim=-1)  # [B, d_atom]
        scale = scale.to(a.dtype)[:, None, None, :]
        shift = shift.to(a.dtype)[:, None, None, :]
        return self.out(self.out_norm(a) * (1 + scale) + shift)  # [B, L, A, 3]
