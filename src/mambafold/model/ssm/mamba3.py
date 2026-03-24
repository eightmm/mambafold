"""Mamba-3 layer: thin wrapper around official mamba_ssm.modules.mamba3.Mamba3.

Adds padding mask support (zero pad before SSM, zero out after).
Requires mamba-ssm installed from main branch:
    pip install git+https://github.com/state-spaces/mamba --no-build-isolation

Reference: github.com/state-spaces/mamba  |  arXiv:2603.15569
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mamba_ssm.modules.mamba3 import Mamba3 as _Mamba3


class Mamba3Layer(nn.Module):
    """Mamba-3 SSM block with padding mask support.

    Wraps the official mamba_ssm.modules.mamba3.Mamba3 and handles
    variable-length sequences by zeroing padding positions.

    Args:
        d_model:   input dimension
        d_state:   SSM state size N
        expand:    d_inner = d_model * expand
        headdim:   head dimension P  (d_inner must be divisible)
        mimo_rank: MIMO rank R  (1 = SISO)
        dtype:     weight dtype (pass bfloat16 for bf16 training)
        device:    device for weight initialization
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        mimo_rank: int = 4,
        dtype=None,
        device=None,
        **kwargs,
    ):
        super().__init__()
        is_mimo = mimo_rank > 1
        self.chunk_size = max(1, 64 // mimo_rank) if is_mimo else 64
        self.ssm = _Mamba3(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
            chunk_size=self.chunk_size,
            is_outproj_norm=False,
            dtype=dtype,
            device=device,
        )

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Args:
            x:    (B, L, D)
            mask: (B, L) bool  True=valid, False=padding
        Returns:  (B, L, D)
        """
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)

        # Mamba3 requires seq_len divisible by chunk_size — pad if needed
        B, S, D = x.shape
        pad = (self.chunk_size - S % self.chunk_size) % self.chunk_size
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))

        y = self.ssm(x)

        if pad > 0:
            y = y[:, :S]
        if mask is not None:
            y = y * mask.unsqueeze(-1).to(y.dtype)
        return y
