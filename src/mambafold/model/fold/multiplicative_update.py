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

try:
    from cuequivariance_torch import triangle_multiplicative_update as _cueq_tmu
    _HAS_CUEQ = True
except ImportError:  # backend (cuequivariance-ops-torch-cuXX) not installed
    _cueq_tmu = None
    _HAS_CUEQ = False


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


class CuEqTriangleMultiplication(nn.Module):
    """Fused triangular multiplicative update via cuEquivariance (Triton kernel).

    Drop-in replacement for `LayerNorm + TriangleMultiplicativeUpdate`: the input
    LayerNorm is *internal* to the kernel, so the caller passes the raw pair and
    adds the returned update as a residual (no external pre-norm).

    Speedup vs the native einsum (fwd+bwd, bf16): ~1.3× at L=512, ~3× at L=1024
    (measured); the gain grows with L because the fused kernel avoids
    materialising the [B,L,L,c] intermediates. Cost: the intermediate width is
    fixed to `d_pair` (kernel constraint c==D), so `pair_mult_c` does not apply,
    and `d_pair` must be a multiple of 32.

    See cuequivariance_torch.triangle_multiplicative_update for the weight layout.
    """

    def __init__(self, d_pair: int = 192, mode: str = "outgoing", eps: float = 1e-5):
        super().__init__()
        assert _HAS_CUEQ, (
            "CuEqTriangleMultiplication requires cuequivariance-torch + "
            "cuequivariance-ops-torch-cuXX (the Triton backend)."
        )
        assert mode in ("outgoing", "incoming"), mode
        assert d_pair % 32 == 0, f"cuEq tmu needs d_pair % 32 == 0, got {d_pair}"
        self.direction = mode
        self.eps = eps
        # cuEq weight layout: input norm/gate/proj (D→2D), output norm/gate/proj (D→D).
        self.norm_in = nn.LayerNorm(d_pair)
        self.p_in = nn.Linear(d_pair, 2 * d_pair)
        self.g_in = nn.Linear(d_pair, 2 * d_pair)
        self.norm_out = nn.LayerNorm(d_pair)
        self.p_out = nn.Linear(d_pair, d_pair)
        self.g_out = nn.Linear(d_pair, d_pair)

    def forward(self, pair: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            pair: [B, L, L, d_pair]  (raw — internal LayerNorm is applied)
            mask: [B, L, L] bool
        Returns:
            [B, L, L, d_pair] — the *update* to add (residual handled by caller).
        """
        return _cueq_tmu(
            pair, direction=self.direction, mask=mask,
            norm_in_weight=self.norm_in.weight, norm_in_bias=self.norm_in.bias,
            p_in_weight=self.p_in.weight, p_in_bias=self.p_in.bias,
            g_in_weight=self.g_in.weight, g_in_bias=self.g_in.bias,
            norm_out_weight=self.norm_out.weight, norm_out_bias=self.norm_out.bias,
            p_out_weight=self.p_out.weight, p_out_bias=self.p_out.bias,
            g_out_weight=self.g_out.weight, g_out_bias=self.g_out.bias,
            eps=self.eps,
        )
