"""Mamba-3 SSM stack: primitives, causal/bidirectional blocks, reusable stack.

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


# ── Primitives ─────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Args:
            d_model (int): Feature dimension size. Initializes learnable scale
                weight of shape [d_model].
            eps (float): Small constant for numerical stability. Default: 1e-6.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape [*, d_model].

        Returns:
            Tensor: RMS-normalized tensor of shape [*, d_model].
        """
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, d_model: int, d_ff: int = None):
        """
        Args:
            d_model (int): Input and output feature dimension [*, d_model].
            d_ff (int | None): Hidden dimension. Defaults to floor(8/3 * d_model)
                rounded up to the nearest multiple of 8.
        """
        super().__init__()
        d_ff = d_ff or int(d_model * 8 / 3)
        d_ff = ((d_ff + 7) // 8) * 8
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape [*, d_model].

        Returns:
            Tensor: Output tensor of shape [*, d_model].
                Computed as w2(SiLU(w1(x)) * w3(x)).
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Mamba3Layer(nn.Module):
    """Mamba-3 SSM block with padding mask support.

    Wraps the official mamba_ssm.modules.mamba3.Mamba3 and handles
    variable-length sequences by zeroing padding positions.
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
        """
        Args:
            d_model (int): Token feature dimension. Input/output shape [B, S, d_model].
            d_state (int): SSM state expansion factor. Default: 128.
            expand (int): Inner dimension multiplier (inner_dim = expand * d_model).
                Default: 2.
            headdim (int): Dimension per attention head inside the SSM. Default: 64.
            mimo_rank (int): MIMO rank; values > 1 enable MIMO mode. Controls
                chunk_size = max(1, 64 // mimo_rank). Default: 4.
            dtype: Floating-point dtype forwarded to the underlying Mamba3 kernel.
            device: Device forwarded to the underlying Mamba3 kernel.
            **kwargs: Extra keyword arguments (ignored, for forward-compatibility).
        """
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
            x (Tensor): Input token sequence of shape [B, S, d_model].
            mask (Tensor | None): Boolean or float padding mask of shape [B, S].
                Padding positions (mask == 0) are zeroed before and after the SSM.

        Returns:
            Tensor: Output sequence of shape [B, S, d_model] with padding zeroed.
        """
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)

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


# ── Blocks ─────────────────────────────────────────────────────────────────

def _flip_by_mask(x: Tensor, mask: Tensor) -> Tensor:
    """Reverse valid positions along sequence dim, keeping padding at end.

    For each batch element, reverses only the valid (non-padding) tokens so
    that padding tokens stay at the tail. Used by BiMamba3Block to run the
    backward SSM pass.

    Args:
        x (Tensor): Input tensor of shape [B, S, D].
        mask (Tensor): Boolean or integer mask of shape [B, S].
            1 = valid token, 0 = padding.

    Returns:
        Tensor: Sequence-reversed tensor of shape [B, S, D].
            Valid tokens appear in reversed order; padding positions are zeroed.
    """
    lengths = mask.sum(dim=1)
    arange = torch.arange(mask.shape[1], device=x.device).unsqueeze(0).expand(mask.shape[0], -1)
    rev_idx = (lengths.unsqueeze(1) - 1 - arange).clamp(min=0)
    out = torch.gather(x, 1, rev_idx.unsqueeze(-1).expand_as(x))
    return out * mask.unsqueeze(-1).to(x.dtype)


class Mamba3Block(nn.Module):
    """Causal Mamba-3 block: pre-norm SSM + SwiGLU FFN."""

    def __init__(self, d_model: int, d_state: int = 64, mimo_rank: int = 4,
                 expand: int = 2, headdim: int = 64):
        """
        Args:
            d_model (int): Token feature dimension. Input/output shape [B, S, d_model].
            d_state (int): SSM state expansion factor. Default: 64.
            mimo_rank (int): MIMO rank forwarded to Mamba3Layer. Default: 4.
            expand (int): Inner dimension multiplier inside the SSM. Default: 2.
            headdim (int): Dimension per head inside the SSM. Default: 64.
        """
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.ssm = Mamba3Layer(d_model=d_model, d_state=d_state, expand=expand,
                               headdim=headdim, mimo_rank=mimo_rank)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input token sequence of shape [B, S, d_model].
            mask (Tensor): Boolean or float padding mask of shape [B, S].

        Returns:
            Tensor: Output tensor of shape [B, S, d_model].
                Residual stream: x + SSM(RMSNorm(x)) + FFN(RMSNorm(x)),
                then padding positions zeroed.
        """
        x = x + self.ssm(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x * mask.unsqueeze(-1).to(x.dtype)


class BiMamba3Block(nn.Module):
    """Bidirectional Mamba-3: forward + backward SSM summed."""

    def __init__(self, d_model: int, d_state: int = 64, mimo_rank: int = 4,
                 expand: int = 2, headdim: int = 64):
        """
        Args:
            d_model (int): Token feature dimension. Input/output shape [B, S, d_model].
            d_state (int): SSM state expansion factor shared by both directions.
                Default: 64.
            mimo_rank (int): MIMO rank forwarded to both Mamba3Layers. Default: 4.
            expand (int): Inner dimension multiplier inside each SSM. Default: 2.
            headdim (int): Dimension per head inside each SSM. Default: 64.
        """
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.mamba_f = Mamba3Layer(d_model=d_model, d_state=d_state, expand=expand,
                                   headdim=headdim, mimo_rank=mimo_rank)
        self.mamba_b = Mamba3Layer(d_model=d_model, d_state=d_state, expand=expand,
                                   headdim=headdim, mimo_rank=mimo_rank)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input token sequence of shape [B, S, d_model].
            mask (Tensor): Boolean or float padding mask of shape [B, S].

        Returns:
            Tensor: Output tensor of shape [B, S, d_model].
                Residual stream: x + mamba_f(h) + flip(mamba_b(flip(h))) + FFN(RMSNorm(x)),
                then padding positions zeroed. h = RMSNorm(x).
        """
        h = self.norm1(x)
        y_f = self.mamba_f(h, mask)
        y_b = _flip_by_mask(self.mamba_b(_flip_by_mask(h, mask), mask), mask)
        x = x + y_f + y_b
        x = x + self.ffn(self.norm2(x))
        return x * mask.unsqueeze(-1).to(x.dtype)


class MambaStack(nn.Module):
    """Reusable stack of Mamba-3 blocks (causal or bidirectional)."""

    def __init__(self, d_model: int, n_layers: int, d_state: int = 64,
                 mimo_rank: int = 4, expand: int = 2, headdim: int = 64,
                 bidirectional: bool = True):
        """
        Args:
            d_model (int): Token feature dimension. Input/output shape [B, S, d_model].
            n_layers (int): Number of stacked blocks.
            d_state (int): SSM state expansion factor for each block. Default: 64.
            mimo_rank (int): MIMO rank forwarded to each block. Default: 4.
            expand (int): Inner dimension multiplier inside each block's SSM.
                Default: 2.
            headdim (int): Dimension per head inside each block's SSM. Default: 64.
            bidirectional (bool): If True, uses BiMamba3Block (forward + backward);
                otherwise uses causal Mamba3Block. Default: True.
        """
        super().__init__()
        block_cls = BiMamba3Block if bidirectional else Mamba3Block
        self.layers = nn.ModuleList([
            block_cls(d_model=d_model, d_state=d_state, mimo_rank=mimo_rank,
                      expand=expand, headdim=headdim)
            for _ in range(n_layers)
        ])

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input token sequence of shape [B, S, d_model].
            mask (Tensor): Boolean or float padding mask of shape [B, S].

        Returns:
            Tensor: Output tensor of shape [B, S, d_model] after passing through
                all n_layers blocks sequentially.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x
