"""Smoke tests for the paper-style Mamba-3 implementation."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mambafold.data.types import ProteinBatch
from mambafold.model.mambafold import MambaFoldEqM
from mambafold.model.ssm.bimamba3 import BiMamba3Block, Mamba3Block
from mambafold.model.ssm.mamba3 import Mamba3Layer


def test_mamba3_layer_shape_and_mask():
    layer = Mamba3Layer(d_model=16, d_state=8, mimo_rank=2)
    x = torch.randn(2, 5, 16)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)

    y = layer(x, mask)

    assert y.shape == x.shape
    assert torch.allclose(y[0, 3:], torch.zeros_like(y[0, 3:]))


def test_causal_and_bidirectional_blocks_run():
    x = torch.randn(2, 6, 24)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)

    causal = Mamba3Block(d_model=24, d_state=8, mimo_rank=2)
    bidir = BiMamba3Block(d_model=24, d_state=8, mimo_rank=2)

    y_causal = causal(x, mask)
    y_bidir = bidir(x, mask)

    assert y_causal.shape == x.shape
    assert y_bidir.shape == x.shape
    assert torch.allclose(y_causal[0, 4:], torch.zeros_like(y_causal[0, 4:]))
    assert torch.allclose(y_bidir[0, 4:], torch.zeros_like(y_bidir[0, 4:]))


def test_mambafold_runs_with_causal_paper_blocks():
    B, L, A = 2, 4, 15
    batch = ProteinBatch(
        res_type=torch.zeros(B, L, dtype=torch.long),
        atom_type=torch.zeros(B, L, A, dtype=torch.long),
        res_mask=torch.ones(B, L, dtype=torch.bool),
        atom_mask=torch.ones(B, L, A, dtype=torch.bool),
        valid_mask=torch.ones(B, L, A, dtype=torch.bool),
        ca_mask=torch.ones(B, L, dtype=torch.bool),
        x_clean=torch.randn(B, L, A, 3),
        x_gamma=torch.randn(B, L, A, 3),
        eps=torch.randn(B, L, A, 3),
        gamma=torch.rand(B, 1, 1, 1),
        esm=torch.randn(B, L, 32),
    )

    model = MambaFoldEqM(
        d_atom=16,
        d_res=32,
        d_plm=32,
        n_atom_enc=1,
        n_trunk=1,
        n_atom_dec=1,
        use_plm=True,
        atom_d_state=8,
        atom_mimo_rank=2,
        atom_bidirectional=False,
        d_state=8,
        mimo_rank=2,
        trunk_bidirectional=False,
    )

    out = model(batch)

    assert out.shape == (B, L, A, 3)
