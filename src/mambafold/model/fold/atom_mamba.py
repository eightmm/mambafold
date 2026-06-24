"""Atom-level BiMamba encoder/decoder (intra-residue SSM).

The direct all-atom model reasons at three levels, all SSM-based so the
"MambaFold" identity holds end to end:

    atom  →  AtomEncoder (BiMamba over the A atom slots of each residue) → pool
    token →  residue trunk (Mamba + triangle-mult pair stack)            ← global
    atom  →  AtomDecoder (BiMamba over the A atom slots) → per-atom velocity

Atom attention is intentionally avoided: the canonical atom ordering within a
residue (N, CA, C, O, CB, CG, …) gives a meaningful 1-D scan, the per-residue
sequence is tiny (A = MAX_ATOMS_PER_RES = 15), and a masked SSM scan never
produces NaNs on fully-padded rows (unlike a softmax over an all-masked set).
Inter-residue reasoning is the token trunk's job; these blocks only own
intra-residue (side-chain) geometry.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from mambafold.data.constants import ATOM_NAME_TO_ID, NUM_PAIR_TYPES
from mambafold.data.types import ProteinBatch
from mambafold.model.bimamba3 import MambaStack

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
        scale, shift = self.proj(temb).chunk(2, dim=-1)              # [B, d_feat] each
        while scale.dim() < x.dim():                                 # broadcast over L (and A)
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        return x * (1 + scale.to(x.dtype)) + shift.to(x.dtype)


def _atom_mamba(d_atom: int, n_layers: int, *, d_state: int, mimo_rank: int,
                expand: int, headdim: int, bimamba_share: bool) -> MambaStack:
    return MambaStack(
        d_atom, n_layers,
        d_state=d_state, mimo_rank=mimo_rank, expand=expand, headdim=headdim,
        bidirectional=True, bimamba_share=bimamba_share,
    )


class AtomEncoder(nn.Module):
    """BiMamba over atom slots, then a gated masked-mean pool to a residue token.

    Args:
        d_atom: Atom-token width.
        d_ca_emb: Width of the per-atom Fourier coordinate embedding fed in.
        n_layers: Number of BiMamba layers over the atom axis.
    """

    def __init__(self, d_atom: int, d_ca_emb: int, n_layers: int = 2, *,
                 d_temb: int = 128,
                 d_state: int = 64, mimo_rank: int = 4, expand: int = 2,
                 headdim: int = 64, bimamba_share: bool = False):
        super().__init__()
        self.coord_proj = nn.Linear(d_ca_emb, d_atom)
        self.pair_type_embed = nn.Embedding(NUM_PAIR_TYPES, d_atom)
        self.atom_type_embed = nn.Embedding(NUM_ATOM_TYPES, d_atom)
        self.in_norm = nn.LayerNorm(d_atom)
        self.film = FiLM(d_atom, d_temb)
        self.mamba = _atom_mamba(
            d_atom, n_layers, d_state=d_state, mimo_rank=mimo_rank,
            expand=expand, headdim=headdim, bimamba_share=bimamba_share,
        )
        self.pool_gate = nn.Linear(d_atom, 1)
        self.out_norm = nn.LayerNorm(d_atom)

    def forward(self, coord_emb: Tensor, batch: ProteinBatch, temb: Tensor) -> tuple[Tensor, Tensor]:
        """coord_emb: [B, L, A, d_ca_emb], temb: [B, d_temb].
        Returns (token [B,L,d_atom], atom_repr [B,L,A,d_atom])."""
        B, L, A, _ = coord_emb.shape
        a = self.coord_proj(coord_emb)
        a = (a
             + self.pair_type_embed(batch.pair_type).to(a.dtype)
             + self.atom_type_embed(batch.atom_type).to(a.dtype))
        a = self.film(self.in_norm(a), temb)

        m = batch.atom_mask                                          # [B, L, A] bool
        a = self.mamba(a.reshape(B * L, A, -1), m.reshape(B * L, A)).reshape(B, L, A, -1)

        # Gated masked-mean pool over atoms → residue token. Fully-padded
        # residues softmax to zero weight (nan_to_num), so the token is 0.
        w = self.pool_gate(a).squeeze(-1).masked_fill(~m, float("-inf"))  # [B, L, A]
        w = torch.nan_to_num(torch.softmax(w, dim=-1))
        tok = (a * w.unsqueeze(-1)).sum(dim=2)                       # [B, L, d_atom]
        return self.out_norm(tok), a


class AtomDecoder(nn.Module):
    """Broadcast the residue latent onto atoms, BiMamba over atom slots → velocity.

    Conditions on the trunk's residue latent (global reasoning) and skips in the
    encoder's per-atom representation (intra-residue identity/geometry).
    """

    def __init__(self, d_res: int, d_atom: int, n_layers: int = 2, *,
                 d_temb: int = 128,
                 d_state: int = 64, mimo_rank: int = 4, expand: int = 2,
                 headdim: int = 64, bimamba_share: bool = False):
        super().__init__()
        self.ctx_proj = nn.Sequential(nn.LayerNorm(d_res), nn.Linear(d_res, d_atom))
        self.in_norm = nn.LayerNorm(d_atom)
        self.film = FiLM(d_atom, d_temb)
        self.mamba = _atom_mamba(
            d_atom, n_layers, d_state=d_state, mimo_rank=mimo_rank,
            expand=expand, headdim=headdim, bimamba_share=bimamba_share,
        )
        self.out = nn.Sequential(nn.LayerNorm(d_atom), nn.Linear(d_atom, 3))

    def forward(self, res_latent: Tensor, atom_repr: Tensor, batch: ProteinBatch,
                temb: Tensor) -> Tensor:
        """res_latent: [B,L,d_res], atom_repr: [B,L,A,d_atom], temb: [B,d_temb].
        Returns v_atom [B,L,A,3]."""
        B, L, A, _ = atom_repr.shape
        ctx = self.ctx_proj(res_latent).unsqueeze(2)                 # [B, L, 1, d_atom]
        a = self.film(self.in_norm(atom_repr + ctx), temb)
        m = batch.atom_mask
        a = self.mamba(a.reshape(B * L, A, -1), m.reshape(B * L, A)).reshape(B, L, A, -1)
        return self.out(a)                                           # [B, L, A, 3]
