"""Shape / mask / gradient tests for pair blocks (CPU-only).

Covers:
- TriangleMultiplicativeUpdate (outgoing + incoming)
- PairBlock composition (Pairmixer: mult×2 + transition)
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mambafold.model.fold.multiplicative_update import (  # noqa: E402
    TriangleMultiplicativeUpdate,
)
from mambafold.model.fold.pair_blocks import (  # noqa: E402
    PairBlock,
    PairTransition,
)


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


# ── PairBlock composition (Pairmixer) ─────────────────────────────────────

def test_pair_block_shape_and_mask():
    blk = PairBlock(d_pair=48, mult_c=24)
    pair = torch.randn(2, 8, 8, 48)
    mask = torch.zeros(2, 8, 8, dtype=torch.bool)
    mask[:, :5, :5] = True
    out = blk(pair, mask)
    assert out.shape == pair.shape
    assert torch.allclose(out[:, 5:, :], torch.zeros_like(out[:, 5:, :]))


def test_pair_block_gradient_flow():
    blk = PairBlock(d_pair=48, mult_c=24)
    pair = torch.randn(1, 6, 6, 48, requires_grad=True)
    mask = torch.ones(1, 6, 6, dtype=torch.bool)
    blk(pair, mask).sum().backward()
    assert pair.grad is not None
    no_grad = [n for n, p in blk.named_parameters() if p.grad is None]
    assert no_grad == [], f"params without grad: {no_grad}"


def test_pair_block_param_count_in_range():
    """Sanity check on Pairmixer PairBlock size at production-ish dims.

    At d_pair=192, mult_c=128 (native einsum) expect ~470K params per block
    (mult×2 + transition; triangle attention dropped).
    """
    blk = PairBlock(d_pair=192, mult_c=128)
    n = sum(p.numel() for p in blk.parameters())
    assert 350_000 <= n <= 700_000, f"unexpected param count: {n}"


# ── PairTransition (small smoke) ──────────────────────────────────────────

def test_pair_transition_shape():
    t = PairTransition(d_pair=32, hidden_mult=2)
    pair = torch.randn(1, 4, 4, 32)
    assert t(pair).shape == pair.shape
