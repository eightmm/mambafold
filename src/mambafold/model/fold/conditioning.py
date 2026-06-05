"""Stage 1 → Stage 2 conditioning modules.

Two helpers used inside `MambaFoldStage2`:

* `CAAnchoredFourier` — per-atom Fourier features of `(x_t_atom - s1_ca)`.
  Protein analog of Seed3D-2's voxelized positional encoding: anchors every
  atom to its residue's CA frame.
* `Stage1LatentBroadcast` — projects Stage 1 trunk latent [B, L, d_res]
  into Stage 2's atom dim and broadcasts across atom slots.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from mambafold.model.embeddings import CoordinateFourierEmbedder


class CAAnchoredFourier(nn.Module):
    """Fourier encoding of per-atom offset from the residue's CA anchor.

    Args:
        d_out: Output embedding dim per atom (broadcast over the atom axis).
        num_freqs: Number of Fourier frequency bands (passed to
            CoordinateFourierEmbedder; controls bandwidth of relative-position
            features).
    """

    def __init__(self, d_out: int = 64, num_freqs: int = 8):
        super().__init__()
        self.fourier = CoordinateFourierEmbedder(d_out=d_out, num_freqs=num_freqs)

    def forward(self, x_t_atom: Tensor, ca_pos: Tensor) -> Tensor:
        """
        Args:
            x_t_atom: [B, L, A, 3]
            ca_pos:   [B, L, 3]
        Returns:
            [B, L, A, d_out]
        """
        delta = x_t_atom - ca_pos.unsqueeze(2)
        return self.fourier(delta)


class Stage1LatentBroadcast(nn.Module):
    """Project Stage 1 trunk latent into Stage 2's atom dim and broadcast.

    Args:
        d_res:  Stage 1 trunk latent dim.
        d_atom: Stage 2 atom token dim.
    """

    def __init__(self, d_res: int, d_atom: int):
        super().__init__()
        self.proj = nn.Linear(d_res, d_atom)

    def forward(self, s1_latent: Tensor) -> Tensor:
        """
        Args:
            s1_latent: [B, L, d_res]
        Returns:
            [B, L, 1, d_atom] (broadcastable along the atom axis)
        """
        return self.proj(s1_latent).unsqueeze(2)
