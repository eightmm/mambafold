"""Feature embeddings for atom and residue tokens."""

import torch
import torch.nn as nn
from torch import Tensor

from mambafold.data.constants import MAX_ATOMS_PER_RES, NUM_AA_TYPES, NUM_ATOM_TYPES, NUM_PAIR_TYPES


class CoordinateFourierEmbedder(nn.Module):
    """Fourier positional embedding for 3D coordinates."""

    def __init__(self, d_out: int = 128, num_freqs: int = 16):
        super().__init__()
        self.num_freqs = num_freqs
        # 3 coords × (sin + cos) × num_freqs + 3 raw
        raw_dim = 3 + 3 * 2 * num_freqs
        self.proj = nn.Linear(raw_dim, d_out)

        # Fixed frequency bands
        freqs = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer("freqs", freqs)  # [num_freqs]

    def forward(self, coords: Tensor) -> Tensor:
        """
        Args:
            coords: [*, 3]

        Returns: [*, d_out]
        """
        # Expand: [*, 3, 1] * [num_freqs] -> [*, 3, num_freqs]
        scaled = coords.unsqueeze(-1) * self.freqs  # [*, 3, F]
        fourier = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1).flatten(-2)
        return self.proj(torch.cat([coords, fourier], dim=-1))


class AtomFeatureEmbedder(nn.Module):
    """Embed atom/residue/pair type and coordinate features into atom tokens.

    Three complementary embeddings summed before projection:
      - res_type_embed:  amino acid identity (broadcast over atoms)
      - atom_type_embed: atomic element/type properties
      - pair_embed:      exact (residue, atom) chemical identity (167 unique pairs)
    """

    def __init__(self, d_atom: int = 256, d_fourier: int = 128):
        super().__init__()
        self.res_type_embed = nn.Embedding(NUM_AA_TYPES, d_atom)
        self.atom_type_embed = nn.Embedding(NUM_ATOM_TYPES, d_atom)
        self.pair_embed = nn.Embedding(NUM_PAIR_TYPES, d_atom)
        self.coord_embed = CoordinateFourierEmbedder(d_out=d_fourier)
        self.proj = nn.Linear(d_atom + d_fourier, d_atom)

    def forward(
        self,
        res_type: Tensor,    # [B, L]
        atom_type: Tensor,   # [B, L, A]
        pair_type: Tensor,   # [B, L, A]
        coords: Tensor,      # [B, L, A, 3]
        atom_mask: Tensor,   # [B, L, A]
    ) -> Tensor:
        """Returns: [B, L, A, d_atom] atom token embeddings"""
        B, L, A = atom_type.shape

        feat = (
            self.res_type_embed(res_type).unsqueeze(2).expand(-1, -1, A, -1)  # [B, L, A, d_atom]
            + self.atom_type_embed(atom_type)                                  # [B, L, A, d_atom]
            + self.pair_embed(pair_type)                                       # [B, L, A, d_atom]
        )
        coord_feat = self.coord_embed(coords)                                  # [B, L, A, d_fourier]

        out = self.proj(torch.cat([feat, coord_feat], dim=-1))
        return out * atom_mask.unsqueeze(-1).to(out.dtype)
