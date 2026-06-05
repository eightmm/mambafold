"""PairBlock — composes one round of pair-side updates for Stage 1.

Per `docs/pair_module.md` §1:

    PairBlock(Z, mask):
        Z ← Z + MultiplicativeUpdate(LN(Z), mask, mode="outgoing")
        Z ← Z + MultiplicativeUpdate(LN(Z), mask, mode="incoming")
        Z ← Z + LinearTriangleAttention(LN(Z), mask, axis="start")
        Z ← Z + LinearTriangleAttention(LN(Z), mask, axis="end")
        Z ← Z + Transition(LN(Z))
        return Z * mask

Plus two helpers:
    PairTransition          — 2-layer FFN with ReLU (hidden_mult=2 default)
    pair_to_single          — masked row-mean reduction, projected to d_res
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mambafold.model.fold.linear_tri_attn import LinearTriangleAttention
from mambafold.model.fold.multiplicative_update import TriangleMultiplicativeUpdate


class PairTransition(nn.Module):
    """Pair-side feed-forward block (AF2 standard, hidden_mult=2)."""

    def __init__(self, d_pair: int, hidden_mult: int = 2):
        super().__init__()
        self.lin1 = nn.Linear(d_pair, hidden_mult * d_pair)
        self.lin2 = nn.Linear(hidden_mult * d_pair, d_pair)

    def forward(self, pair: Tensor) -> Tensor:
        return self.lin2(F.relu(self.lin1(pair)))


class PairBlock(nn.Module):
    """One pair stack block: 2 mult updates + 2 linear-tri attentions + transition.

    Each sub-module is wrapped in `LayerNorm` (pre-norm) and added as a
    residual. The whole block is masked at exit so padded rows stay zero.
    """

    def __init__(
        self,
        d_pair: int = 192,
        n_heads: int = 4,
        d_head: int = 48,
        mult_c: int = 128,
        transition_hidden_mult: int = 2,
        tri_attn_variant: str = "gated",
    ):
        super().__init__()
        self.d_pair = d_pair

        # Mult updates
        self.norm_mu_out = nn.LayerNorm(d_pair)
        self.mu_out = TriangleMultiplicativeUpdate(d_pair, "outgoing", c=mult_c)
        self.norm_mu_in = nn.LayerNorm(d_pair)
        self.mu_in = TriangleMultiplicativeUpdate(d_pair, "incoming", c=mult_c)

        # Linear triangular attentions
        self.norm_tri_start = nn.LayerNorm(d_pair)
        self.tri_start = LinearTriangleAttention(
            d_pair, n_heads, d_head, axis="start", variant=tri_attn_variant,
        )
        self.norm_tri_end = nn.LayerNorm(d_pair)
        self.tri_end = LinearTriangleAttention(
            d_pair, n_heads, d_head, axis="end", variant=tri_attn_variant,
        )

        # Transition
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
        pair = pair + self.mu_out(self.norm_mu_out(pair), mask)
        pair = pair + self.mu_in(self.norm_mu_in(pair), mask)
        pair = pair + self.tri_start(self.norm_tri_start(pair), mask)
        pair = pair + self.tri_end(self.norm_tri_end(pair), mask)
        pair = pair + self.transition(self.norm_trans(pair))
        return pair * mask.unsqueeze(-1).to(pair.dtype)


def pair_to_single(pair: Tensor, res_mask: Tensor, proj: nn.Linear) -> Tensor:
    """Reduce [B, L, L, d_pair] to per-residue bias [B, L, d_res] via masked row-mean.

    Legacy mean-pool reduction (kept for reference / ablation). The active path
    uses `PairToSingleAttention`, which learns *which* j matters per row instead
    of averaging all of them.

    Args:
        pair:     [B, L, L, d_pair]
        res_mask: [B, L] bool
        proj:     nn.Linear(d_pair, d_res)  — supplied by caller.
    """
    mask_j = res_mask.unsqueeze(1).unsqueeze(-1).to(pair.dtype)   # [B, 1, L, 1]
    denom = mask_j.sum(dim=2).clamp(min=1)                        # [B, 1, 1]
    pair_row = (pair * mask_j).sum(dim=2) / denom                 # [B, L, d_pair]
    return proj(pair_row) * res_mask.unsqueeze(-1).to(pair.dtype)


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
