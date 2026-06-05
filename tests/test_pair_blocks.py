"""Shape / mask / gradient tests for pair blocks (CPU-only).

Covers:
- LinearTriangleAttention (start + end axes; gated + additive variants)
- TriangleMultiplicativeUpdate (outgoing + incoming)
- PairBlock composition
- pair_to_single reduction
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch.nn as nn  # noqa: E402

from mambafold.model.fold.linear_tri_attn import LinearTriangleAttention  # noqa: E402
from mambafold.model.fold.multiplicative_update import (  # noqa: E402
    TriangleMultiplicativeUpdate,
)
from mambafold.model.fold.pair_blocks import (  # noqa: E402
    PairBlock,
    PairTransition,
    pair_to_single,
)

# ── Linear Triangle Attention ─────────────────────────────────────────────

def test_linear_tri_attn_shape_start():
    mod = LinearTriangleAttention(d_pair=64, n_heads=4, d_head=16, axis="start")
    pair = torch.randn(2, 8, 8, 64)
    mask = torch.ones(2, 8, 8, dtype=torch.bool)
    out = mod(pair, mask)
    assert out.shape == pair.shape


def test_linear_tri_attn_shape_end():
    mod = LinearTriangleAttention(d_pair=64, n_heads=4, d_head=16, axis="end")
    pair = torch.randn(2, 8, 8, 64)
    mask = torch.ones(2, 8, 8, dtype=torch.bool)
    out = mod(pair, mask)
    assert out.shape == pair.shape


def test_linear_tri_attn_mask_zeros_padding():
    """Padding cells (mask=False) must be zero in output."""
    mod = LinearTriangleAttention(d_pair=32, n_heads=2, d_head=16)
    pair = torch.randn(1, 6, 6, 32)
    mask = torch.zeros(1, 6, 6, dtype=torch.bool)
    mask[0, :4, :4] = True
    out = mod(pair, mask)
    assert torch.allclose(out[0, 4:, :], torch.zeros_like(out[0, 4:, :]))
    assert torch.allclose(out[0, :, 4:], torch.zeros_like(out[0, :, 4:]))


def test_linear_tri_attn_additive_variant_runs():
    mod = LinearTriangleAttention(
        d_pair=32, n_heads=2, d_head=16, variant="additive",
    )
    pair = torch.randn(1, 5, 5, 32)
    mask = torch.ones(1, 5, 5, dtype=torch.bool)
    out = mod(pair, mask)
    assert out.shape == pair.shape


def test_linear_tri_attn_gradient_flow():
    mod = LinearTriangleAttention(d_pair=32, n_heads=2, d_head=16)
    pair = torch.randn(1, 5, 5, 32, requires_grad=True)
    mask = torch.ones(1, 5, 5, dtype=torch.bool)
    out = mod(pair, mask)
    out.sum().backward()
    assert pair.grad is not None
    for name, p in mod.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"


# ── Triangle Multiplicative Update ────────────────────────────────────────

def test_mult_update_outgoing_shape():
    mod = TriangleMultiplicativeUpdate(d_pair=64, mode="outgoing", c=32)
    pair = torch.randn(2, 7, 7, 64)
    mask = torch.ones(2, 7, 7, dtype=torch.bool)
    out = mod(pair, mask)
    assert out.shape == pair.shape


def test_mult_update_incoming_shape():
    mod = TriangleMultiplicativeUpdate(d_pair=64, mode="incoming", c=32)
    pair = torch.randn(2, 7, 7, 64)
    mask = torch.ones(2, 7, 7, dtype=torch.bool)
    out = mod(pair, mask)
    assert out.shape == pair.shape


def test_mult_update_mask_zeros_padding():
    mod = TriangleMultiplicativeUpdate(d_pair=32, mode="outgoing", c=16)
    pair = torch.randn(1, 6, 6, 32)
    mask = torch.zeros(1, 6, 6, dtype=torch.bool)
    mask[0, :3, :3] = True
    out = mod(pair, mask)
    assert torch.allclose(out[0, 3:, :], torch.zeros_like(out[0, 3:, :]))
    assert torch.allclose(out[0, :, 3:], torch.zeros_like(out[0, :, 3:]))


def test_mult_update_gradient_flow():
    mod = TriangleMultiplicativeUpdate(d_pair=32, mode="incoming", c=16)
    pair = torch.randn(1, 5, 5, 32, requires_grad=True)
    mask = torch.ones(1, 5, 5, dtype=torch.bool)
    mod(pair, mask).sum().backward()
    assert pair.grad is not None


# ── PairBlock composition ─────────────────────────────────────────────────

def test_pair_block_shape_and_mask():
    blk = PairBlock(d_pair=48, n_heads=2, d_head=12, mult_c=24)
    pair = torch.randn(2, 8, 8, 48)
    mask = torch.zeros(2, 8, 8, dtype=torch.bool)
    mask[:, :5, :5] = True
    out = blk(pair, mask)
    assert out.shape == pair.shape
    assert torch.allclose(out[:, 5:, :], torch.zeros_like(out[:, 5:, :]))


def test_pair_block_gradient_flow():
    blk = PairBlock(d_pair=48, n_heads=2, d_head=12, mult_c=24)
    pair = torch.randn(1, 6, 6, 48, requires_grad=True)
    mask = torch.ones(1, 6, 6, dtype=torch.bool)
    blk(pair, mask).sum().backward()
    assert pair.grad is not None
    no_grad = [n for n, p in blk.named_parameters() if p.grad is None]
    assert no_grad == [], f"params without grad: {no_grad}"


def test_pair_block_param_count_in_range():
    """Sanity check on PairBlock size at production-ish dims.

    At d_pair=192, n_heads=4, d_head=48, mult_c=128 expect ~840K params per
    block (×6 blocks → ~5M pair-side). Big enough to make the pair-side path meaningful
    capacity, but well under the 1.5M/block range that would push total
    Stage-1 past target.
    """
    blk = PairBlock(d_pair=192, n_heads=4, d_head=48, mult_c=128)
    n = sum(p.numel() for p in blk.parameters())
    assert 500_000 <= n <= 1_500_000, f"unexpected param count: {n}"


# ── pair_to_single reduction ──────────────────────────────────────────────

def test_pair_to_single_shape_and_mask():
    proj = nn.Linear(48, 96)
    pair = torch.randn(2, 6, 6, 48)
    res_mask = torch.ones(2, 6, dtype=torch.bool)
    res_mask[1, 4:] = False
    out = pair_to_single(pair, res_mask, proj)
    assert out.shape == (2, 6, 96)
    assert torch.allclose(out[1, 4:, :], torch.zeros_like(out[1, 4:, :]))


# ── PairTransition (small smoke) ──────────────────────────────────────────

def test_pair_transition_shape():
    t = PairTransition(d_pair=32, hidden_mult=2)
    pair = torch.randn(1, 4, 4, 32)
    assert t(pair).shape == pair.shape
