"""Linear Triangular Attention (SeedFold arXiv:2512.24354v1).

Replaces classical triangle attention `softmax(QKᵀ + B)·V` with a ReLU
feature-map variant `ϕ(Q)·[ϕ(K)ᵀV]·ψ(B)`, enabling associative reordering
of the matrix product and reducing complexity from O(L²·d) to O(L·d²).

Gated variant: ψ(B) = sigmoid(B) is applied element-wise on the attention
output. SeedFold eq. 15 finalises with `Linear(σ(Linear(Z)) ⊙ LN(out))`.

See `docs/pair_module.md` §3 for equations and shapes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LinearTriangleAttention(nn.Module):
    """Linear-complexity triangular attention with ReLU feature maps.

    Two axes — start (rows) and end (columns) — are handled by the same
    module via the `axis` argument. Call once per axis inside a PairBlock.

    For axis="start", an "anchor" index i is fixed and attention runs over
    the L_seq dimension at that anchor's row. axis="end" transposes the
    pair tensor before/after processing.

    Args:
        d_pair: Pair tensor feature dim.
        n_heads: Number of parallel attention heads.
        d_head:  Per-head feature dim. n_heads * d_head should equal the
            inner attention dim (no constraint vs d_pair).
        axis: "start" or "end".
        variant: "gated" applies sigmoid(B) gating; "additive" adds B·V
            (no gating). Default "gated" matches SeedFold's primary config.
    """

    def __init__(
        self,
        d_pair: int = 192,
        n_heads: int = 4,
        d_head: int = 48,
        axis: str = "start",
        variant: str = "gated",
    ):
        super().__init__()
        assert axis in ("start", "end"), axis
        assert variant in ("gated", "additive"), variant
        self.d_pair = d_pair
        self.n_heads = n_heads
        self.d_head = d_head
        self.axis = axis
        self.variant = variant

        inner = n_heads * d_head
        self.lin_q = nn.Linear(d_pair, inner, bias=False)
        self.lin_k = nn.Linear(d_pair, inner, bias=False)
        self.lin_v = nn.Linear(d_pair, inner, bias=False)
        # Per-head scalar bias (sigmoid-gated). One per head per [anchor, seq] cell.
        self.lin_b = nn.Linear(d_pair, n_heads, bias=False)
        # SeedFold eq. 15: outer gate σ(Linear(Z)) ⊙ LN(out)
        self.lin_g = nn.Linear(d_pair, d_pair)
        self.norm_out = nn.LayerNorm(inner)
        self.lin_out = nn.Linear(inner, d_pair)

    def forward(self, pair: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            pair: [B, L, L, d_pair]
            mask: [B, L, L] boolean — True for valid pair positions.
        Returns:
            [B, L, L, d_pair] with padding rows zeroed.
        """
        if self.axis == "end":
            pair = pair.transpose(1, 2).contiguous()
            mask = mask.transpose(1, 2).contiguous()

        m_f = mask.unsqueeze(-1).to(pair.dtype)
        pair_in = pair * m_f

        # Projections — ϕ(Q), ϕ(K) = ReLU; V is linear.
        Q = F.relu(self.lin_q(pair_in))                       # [B, La, Ls, h*d]
        K = F.relu(self.lin_k(pair_in))
        V = self.lin_v(pair_in)
        B_bias = self.lin_b(pair_in)                          # [B, La, Ls, h]
        if self.variant == "gated":
            B_bias = torch.sigmoid(B_bias)

        # Reshape to per-head: [B, La, Ls, h, d_head]
        B_, La, Ls, _ = Q.shape
        Q = Q.view(B_, La, Ls, self.n_heads, self.d_head)
        K = K.view(B_, La, Ls, self.n_heads, self.d_head)
        V = V.view(B_, La, Ls, self.n_heads, self.d_head)

        # Mask K & V along the seq axis so padding positions don't contribute.
        # m_f shape [B, La, Ls, 1] → unsqueeze once → [B, La, Ls, 1, 1] broadcasts
        # against Q/K/V shape [B, La, Ls, h, d_head].
        seq_mask = m_f.unsqueeze(-1)                          # [B, La, Ls, 1, 1]
        K = K * seq_mask
        V = V * seq_mask

        # Right-product trick: KV = Σ_s K[anchor,s] ⊗ V[anchor,s]
        # Shape: [B, La, h, d_head, d_head]
        KV = torch.einsum("bnshd,bnsht->bnhdt", K, V)

        # Per-(anchor, seq) query against the aggregated KV: [B, La, Ls, h, d_head]
        out = torch.einsum("bnshd,bnhdt->bnsht", Q, KV)

        if self.variant == "gated":
            out = out * B_bias.unsqueeze(-1)
        else:  # additive: B_bias acts as an extra (sigmoid not applied) summand
            out = out + B_bias.unsqueeze(-1) * V

        # Combine heads, post-process per SeedFold eq. 15.
        out = out.reshape(B_, La, Ls, self.n_heads * self.d_head)
        out = self.norm_out(out)
        out = self.lin_out(out)                               # [B, La, Ls, d_pair]
        gate = torch.sigmoid(self.lin_g(pair))                # outer gate on original
        out = out * gate

        if self.axis == "end":
            out = out.transpose(1, 2).contiguous()
            mask = mask.transpose(1, 2).contiguous()

        return out * mask.unsqueeze(-1).to(out.dtype)
