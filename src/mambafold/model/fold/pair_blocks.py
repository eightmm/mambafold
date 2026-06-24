"""PairBlock — composes one round of pair-side updates.

Pairmixer design (arXiv:2510.18870 "Triangle Multiplication is All You Need"):
triangle multiplication is the load-bearing op, so the block is just the two
multiplicative updates + a transition (triangle attention dropped):

    PairBlock(Z, mask):
        Z ← Z + MultiplicativeUpdate(LN(Z), mask, mode="outgoing")
        Z ← Z + MultiplicativeUpdate(LN(Z), mask, mode="incoming")
        Z ← Z + Transition(LN(Z))
        return Z * mask
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mambafold.model.fold.multiplicative_update import (
    CuEqTriangleMultiplication,
    TriangleMultiplicativeUpdate,
)


class PairTransition(nn.Module):
    """Pair-side feed-forward block (AF2 standard, hidden_mult=2)."""

    def __init__(self, d_pair: int, hidden_mult: int = 2):
        super().__init__()
        self.lin1 = nn.Linear(d_pair, hidden_mult * d_pair)
        self.lin2 = nn.Linear(hidden_mult * d_pair, d_pair)

    def forward(self, pair: Tensor) -> Tensor:
        return self.lin2(F.relu(self.lin1(pair)))


class PairBlock(nn.Module):
    """One Pairmixer block: triangle mult (outgoing + incoming) + transition.

    Each sub-module is pre-normed and added as a residual; the block is masked
    at exit so padded rows stay zero. The mult kernel is either the native
    einsum (`TriangleMultiplicativeUpdate`) or the cuEq fused Triton kernel
    (`use_cueq_mult=True`) — the latter folds the pre-norm inside and fixes the
    intermediate width to d_pair (so `mult_c` is ignored).
    """

    def __init__(
        self,
        d_pair: int = 192,
        mult_c: int = 128,
        transition_hidden_mult: int = 2,
        use_cueq_mult: bool = False,
    ):
        super().__init__()
        self.d_pair = d_pair
        self.use_cueq_mult = use_cueq_mult

        if use_cueq_mult:
            self.mu_out = CuEqTriangleMultiplication(d_pair, "outgoing")
            self.mu_in = CuEqTriangleMultiplication(d_pair, "incoming")
        else:
            self.norm_mu_out = nn.LayerNorm(d_pair)
            self.mu_out = TriangleMultiplicativeUpdate(d_pair, "outgoing", c=mult_c)
            self.norm_mu_in = nn.LayerNorm(d_pair)
            self.mu_in = TriangleMultiplicativeUpdate(d_pair, "incoming", c=mult_c)

        self.norm_trans = nn.LayerNorm(d_pair)
        self.transition = PairTransition(d_pair, hidden_mult=transition_hidden_mult)

    def forward(self, pair: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            pair: [B, L, L, d_pair]
            mask: [B, L, L] bool
        Returns:
            [B, L, L, d_pair] with padding zeroed.
        """
        if self.use_cueq_mult:
            pair = pair + self.mu_out(pair, mask)   # cuEq folds the pre-norm in
            pair = pair + self.mu_in(pair, mask)
        else:
            pair = pair + self.mu_out(self.norm_mu_out(pair), mask)
            pair = pair + self.mu_in(self.norm_mu_in(pair), mask)
        pair = pair + self.transition(self.norm_trans(pair))
        return pair * mask.unsqueeze(-1).to(pair.dtype)


class PairToSingleAttention(nn.Module):
    """Attention pooling of each pair row into a per-residue bias [B, L, d_res].

    For residue i, attends over columns j with logits derived from `pair[i, j]`
    (multi-head, softmax over j), then takes a weighted sum of per-edge values —
    so the reduction keeps *which* j matters instead of mean-pooling it away.

    Values stay in `d_pair` width (the big [B,L,L,*] intermediate is no larger
    than the pair tensor itself); only the pooled [B, L, d_pair] is projected up
    to `d_res`.
    """

    def __init__(self, d_pair: int, d_res: int, n_heads: int = 4):
        super().__init__()
        assert d_pair % n_heads == 0, (d_pair, n_heads)
        self.n_heads = n_heads
        self.d_head = d_pair // n_heads
        self.to_score = nn.Linear(d_pair, n_heads)
        self.to_value = nn.Linear(d_pair, d_pair)
        self.out = nn.Linear(d_pair, d_res)

    def forward(self, pair: Tensor, res_mask: Tensor) -> Tensor:
        """
        Args:
            pair:     [B, L, L, d_pair]
            res_mask: [B, L] bool
        Returns:
            [B, L, d_res] with padding rows zeroed.
        """
        B, L, _, _ = pair.shape
        scores = self.to_score(pair)                                  # [B, L, L, h]
        col_valid = res_mask.unsqueeze(1).unsqueeze(-1)               # [B, 1, L, 1] over j
        scores = scores.masked_fill(~col_valid, float("-inf"))
        attn = torch.softmax(scores, dim=2)                           # over j
        attn = torch.nan_to_num(attn)                                 # fully-masked rows → 0
        v = self.to_value(pair).view(B, L, L, self.n_heads, self.d_head)
        pooled = torch.einsum("bljh,bljhd->blhd", attn, v)            # [B, L, h, d_head]
        pooled = pooled.reshape(B, L, self.n_heads * self.d_head)     # [B, L, d_pair]
        return self.out(pooled) * res_mask.unsqueeze(-1).to(pooled.dtype)
