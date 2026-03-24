"""Mamba-3 blocks: causal, bidirectional, and reusable stack."""

import torch
import torch.nn as nn
from torch import Tensor

from mambafold.model.blocks import RMSNorm, SwiGLU
from mambafold.model.ssm.mamba3 import Mamba3Layer


def _flip_by_mask(x: Tensor, mask: Tensor) -> Tensor:
    """Reverse valid positions along sequence dim, keeping padding at end."""
    lengths = mask.sum(dim=1)
    arange = torch.arange(mask.shape[1], device=x.device).unsqueeze(0).expand(mask.shape[0], -1)
    rev_idx = (lengths.unsqueeze(1) - 1 - arange).clamp(min=0)
    out = torch.gather(x, 1, rev_idx.unsqueeze(-1).expand_as(x))
    return out * mask.unsqueeze(-1).to(x.dtype)


class Mamba3Block(nn.Module):
    """Causal Mamba-3 block: pre-norm SSM + SwiGLU FFN."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        mimo_rank: int = 4,
        expand: int = 2,
        headdim: int = 64,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.ssm = Mamba3Layer(d_model=d_model, d_state=d_state, expand=expand,
                               headdim=headdim, mimo_rank=mimo_rank)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = x + self.ssm(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x * mask.unsqueeze(-1).to(x.dtype)


class BiMamba3Block(nn.Module):
    """Bidirectional Mamba-3: forward + backward SSM with gated fusion."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        mimo_rank: int = 4,
        expand: int = 2,
        headdim: int = 64,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.mamba_f = Mamba3Layer(d_model=d_model, d_state=d_state, expand=expand,
                                   headdim=headdim, mimo_rank=mimo_rank)
        self.mamba_b = Mamba3Layer(d_model=d_model, d_state=d_state, expand=expand,
                                   headdim=headdim, mimo_rank=mimo_rank)
        self.gate_proj = nn.Linear(2 * d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        h = self.norm1(x)
        y_f = self.mamba_f(h, mask)
        y_b = _flip_by_mask(self.mamba_b(_flip_by_mask(h, mask), mask), mask)
        gate = torch.sigmoid(self.gate_proj(torch.cat([y_f, y_b], dim=-1)))
        x = x + self.out_proj(gate * y_f + (1 - gate) * y_b)
        x = x + self.ffn(self.norm2(x))
        return x * mask.unsqueeze(-1).to(x.dtype)


class MambaStack(nn.Module):
    """Reusable stack of Mamba-3 blocks (causal or bidirectional)."""

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        d_state: int = 64,
        mimo_rank: int = 4,
        expand: int = 2,
        headdim: int = 64,
        bidirectional: bool = True,
    ):
        super().__init__()
        block_cls = BiMamba3Block if bidirectional else Mamba3Block
        self.layers = nn.ModuleList([
            block_cls(d_model=d_model, d_state=d_state, mimo_rank=mimo_rank,
                      expand=expand, headdim=headdim)
            for _ in range(n_layers)
        ])

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x
