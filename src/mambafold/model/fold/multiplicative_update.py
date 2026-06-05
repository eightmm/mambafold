"""AlphaFold-2 triangular multiplicative update (outgoing + incoming).

Pair update rule (Jumper et al. 2021 supplementary):
    a_ik   = σ(W_a_g · Z_ik) ⊙ (W_a · Z_ik)            [B, L, L, c]
    b_jk   = σ(W_b_g · Z_jk) ⊙ (W_b · Z_jk)            [B, L, L, c]
    o_ij   = LN(Σ_k a_ik ⊙ b_jk)                       outgoing
    Z_ij ← σ(W_g · Z_ij) ⊙ W_z · o_ij                  gated output

Incoming variant uses Z_ki and Z_kj in place of Z_ik and Z_jk.

Memory: the intermediate [B, L, L, c] tensors are the dominant cost. With
c=128 and L=1024, each is ~270 MB bf16. Stack of 6 PairBlocks (2 each)
totals ~3 GB before grads. Enable `torch.utils.checkpoint` if tight.

See `docs/pair_module.md` §2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class TriangleMultiplicativeUpdate(nn.Module):
    """Outgoing or incoming triangular multiplicative update.

    Args:
        d_pair: Pair feature dim.
        mode:   "outgoing" (uses Z_ik, Z_jk) or "incoming" (Z_ki, Z_kj).
        c:      Intermediate "head" width. AF2 default 128.
    """

    def __init__(self, d_pair: int = 192, mode: str = "outgoing", c: int = 128):
        super().__init__()
        assert mode in ("outgoing", "incoming"), mode
        self.mode = mode
        self.c = c

        # Two gated linear projections producing a_ik and b_jk.
        self.lin_a = nn.Linear(d_pair, c)
        self.lin_a_g = nn.Linear(d_pair, c)
        self.lin_b = nn.Linear(d_pair, c)
        self.lin_b_g = nn.Linear(d_pair, c)
        # Output norm + gated projection back to d_pair.
        self.norm_o = nn.LayerNorm(c)
        self.lin_z = nn.Linear(c, d_pair)
        self.lin_g = nn.Linear(d_pair, d_pair)

    def forward(self, pair: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            pair: [B, L, L, d_pair]
            mask: [B, L, L] bool
        Returns:
            [B, L, L, d_pair] — the *update* to add (residual handled by caller).
        """
        m = mask.unsqueeze(-1).to(pair.dtype)            # [B, L, L, 1]

        a = torch.sigmoid(self.lin_a_g(pair)) * self.lin_a(pair) * m   # [B,L,L,c]
        b = torch.sigmoid(self.lin_b_g(pair)) * self.lin_b(pair) * m   # [B,L,L,c]

        if self.mode == "outgoing":
            # o_ij = Σ_k a_ik * b_jk
            o = torch.einsum("bikc,bjkc->bijc", a, b)
        else:  # incoming
            # o_ij = Σ_k a_ki * b_kj
            o = torch.einsum("bkic,bkjc->bijc", a, b)

        g = torch.sigmoid(self.lin_g(pair))              # [B, L, L, d_pair]
        out = g * self.lin_z(self.norm_o(o))
        return out * m
