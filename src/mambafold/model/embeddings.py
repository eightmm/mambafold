"""Coordinate and residue-position embeddings used by the folding model."""

import math

import torch
import torch.nn as nn
from torch import Tensor

from mambafold.data.constants import COORD_SCALE


class CoordinateFourierEmbedder(nn.Module):
    """Fourier embedding of 3D coordinates over an explicit Angstrom band.

    The band is declared in Angstrom and converted here with `COORD_SCALE`, so
    it stays pinned to physical distances instead of to whatever normalisation
    happens to be in force. The band this replaces was written directly in
    normalised frequency — `2**linspace(-3, 4)` — which at COORD_SCALE 16
    covered periods of 804 A down to 6.28 A. Both ends were wrong for this
    model: nothing in a protein has an 800 A period, and the finest band could
    not resolve a 1.5 A bond, which is the scale an all-atom coordinate model
    is asked to predict. Fourier features exist to defeat the MLP's bias
    against fine detail; a band that stops at 6.28 A defeats nothing.

    Args:
        d_out: Output embedding dimension.
        num_freqs: Number of Fourier bands, log-spaced across the period range.
        max_period_A / min_period_A: Coarsest and finest spatial period the
            embedding can express, in Angstrom.
        include_norm: Append |coords| as a channel. Used for the CA-relative
            embedder, where the norm is the atom's distance from its own CA and
            is the single most direct fine-scale quantity available.
    """

    def __init__(
        self,
        d_out: int = 128,
        num_freqs: int = 16,
        min_period_A: float = 1.0,
        max_period_A: float = 128.0,
        include_norm: bool = False,
    ):
        super().__init__()
        if not 0.0 < min_period_A < max_period_A:
            raise ValueError(
                f"need 0 < min_period_A < max_period_A, got {min_period_A}, {max_period_A}"
            )
        self.include_norm = include_norm
        raw_dim = 3 + 3 * 2 * num_freqs + (1 if include_norm else 0)
        self.proj = nn.Linear(raw_dim, d_out)

        periods = torch.logspace(
            math.log10(max_period_A), math.log10(min_period_A), num_freqs
        )
        self.register_buffer("freqs", 2.0 * math.pi * COORD_SCALE / periods)

    def forward(self, coords: Tensor) -> Tensor:
        """Embed coordinates shaped ``[..., 3]`` into ``[..., d_out]``."""
        scaled = coords.unsqueeze(-1) * self.freqs
        fourier = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1).flatten(-2)
        parts = [coords, fourier]
        if self.include_norm:
            # sqrt(sum + eps), not `norm`: padded slots and the CA slot itself
            # are exactly the zero vector, where `norm`'s backward is 0/0.
            parts.append(coords.pow(2).sum(dim=-1, keepdim=True).add(1e-12).sqrt())
        return self.proj(torch.cat(parts, dim=-1))


class SequenceFourierEmbedder(nn.Module):
    """Embed per-chain residue position plus chain/entity/symmetry identity."""

    MAX_CHAINS = 64
    MAX_ENTITIES = 64
    MAX_SYM = 32

    def __init__(
        self,
        d_out: int = 64,
        num_freqs: int = 10,
        d_chain: int = 16,
        d_entity: int = 16,
        d_sym: int = 8,
    ):
        super().__init__()
        self.chain_embed = nn.Embedding(self.MAX_CHAINS, d_chain)
        self.entity_embed = nn.Embedding(self.MAX_ENTITIES, d_entity)
        self.sym_embed = nn.Embedding(self.MAX_SYM, d_sym)
        in_dim = 3 + 2 * num_freqs + d_chain + d_entity + d_sym
        self.proj = nn.Linear(in_dim, d_out)
        # Period-based bands, not frequency-based. The previous
        # `2**linspace(0, 4)` put every band between 1 and 16 radians per
        # residue: the longest period it could express was 2*pi ~ 6.3 residues,
        # and seven of the eight bands sat above the Nyquist limit of pi for
        # integer positions, so they aliased. These cover 2 to 2**(num_freqs)
        # residues, which spans a helix turn up to a whole chain.
        periods = 2.0 ** torch.arange(1, num_freqs + 1, dtype=torch.float32)
        self.register_buffer("freqs", 2.0 * math.pi / periods)

    def forward(
        self,
        seq_nums: Tensor,
        mask: Tensor,
        chain_id: Tensor | None = None,
        entity_id: Tensor | None = None,
        sym_id: Tensor | None = None,
    ) -> Tensor:
        valid = mask.to(torch.bool)
        rel = seq_nums.to(self.freqs.dtype)

        if chain_id is None:
            chain_id = torch.zeros_like(seq_nums, dtype=torch.long)
        if entity_id is None:
            entity_id = torch.zeros_like(seq_nums, dtype=torch.long)
        if sym_id is None:
            sym_id = torch.zeros_like(seq_nums, dtype=torch.long)

        # Position relative to the first residue actually in the crop, not the
        # deposited author numbering, whose origin is arbitrary — it can start
        # at 1, at 100, or below zero. Gaps in the numbering survive, so an
        # unmodelled loop still shows up as a jump.
        first = torch.where(valid, rel, torch.full_like(rel, float("inf")))
        rel = rel - first.min(dim=1, keepdim=True).values.clamp(min=-1e9)

        rel_scaled = rel.unsqueeze(-1) * self.freqs
        fourier = torch.cat([torch.sin(rel_scaled), torch.cos(rel_scaled)], dim=-1)
        # Distance to each terminus, log-compressed. A bidirectional scan sees
        # both ends, but nothing in the features told it which end it was near.
        n_valid = valid.sum(dim=1, keepdim=True).to(rel.dtype).clamp(min=1)
        from_n = torch.log1p(rel.clamp(min=0))
        from_c = torch.log1p((n_valid - 1 - rel).clamp(min=0))
        chain = self.chain_embed(chain_id.clamp(max=self.MAX_CHAINS - 1))
        entity = self.entity_embed(entity_id.clamp(max=self.MAX_ENTITIES - 1))
        symmetry = self.sym_embed(sym_id.clamp(max=self.MAX_SYM - 1))
        features = torch.cat(
            [from_n.unsqueeze(-1), from_c.unsqueeze(-1), rel.unsqueeze(-1),
             fourier, chain, entity, symmetry],
            dim=-1,
        )
        output = self.proj(features)
        return output * valid.unsqueeze(-1).to(output.dtype)
