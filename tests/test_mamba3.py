"""Smoke tests for the paper-style Mamba-3 implementation."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mambafold.data.types import ProteinBatch
from mambafold.model.bimamba3 import BiMamba3Block, Mamba3Block, Mamba3Layer
from mambafold.model.fold import MambaFoldAllAtom

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Mamba-3 kernels require a CUDA device"
)


def test_mamba3_layer_shape_and_mask():
    # d_model * expand must be divisible by headdim (default 64 in Mamba3Layer).
    layer = Mamba3Layer(d_model=64, d_state=16, mimo_rank=2).cuda()
    x = torch.randn(2, 5, 64, device="cuda")
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool, device="cuda")

    y = layer(x, mask)

    assert y.shape == x.shape
    assert torch.allclose(y[0, 3:], torch.zeros_like(y[0, 3:]))


def test_causal_and_bidirectional_blocks_run():
    x = torch.randn(2, 6, 64, device="cuda")
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool, device="cuda")

    causal = Mamba3Block(d_model=64, d_state=16, mimo_rank=2).cuda()
    bidir = BiMamba3Block(d_model=64, d_state=16, mimo_rank=2).cuda()

    y_causal = causal(x, mask)
    y_bidir = bidir(x, mask)

    assert y_causal.shape == x.shape
    assert y_bidir.shape == x.shape
    assert torch.allclose(y_causal[0, 4:], torch.zeros_like(y_causal[0, 4:]))
    assert torch.allclose(y_bidir[0, 4:], torch.zeros_like(y_bidir[0, 4:]))


def test_all_atom_model_runs_with_mamba_blocks():
    B, L, A = 2, 4, 15
    dev = "cuda"
    batch = ProteinBatch(
        res_type=torch.zeros(B, L, dtype=torch.long, device=dev),
        res_seq_nums=torch.arange(L, device=dev).unsqueeze(0).expand(B, L).contiguous(),
        atom_type=torch.zeros(B, L, A, dtype=torch.long, device=dev),
        pair_type=torch.zeros(B, L, A, dtype=torch.long, device=dev),
        res_mask=torch.ones(B, L, dtype=torch.bool, device=dev),
        atom_mask=torch.ones(B, L, A, dtype=torch.bool, device=dev),
        valid_mask=torch.ones(B, L, A, dtype=torch.bool, device=dev),
        ca_mask=torch.ones(B, L, dtype=torch.bool, device=dev),
        chain_id=torch.zeros(B, L, dtype=torch.long, device=dev),
        entity_id=torch.zeros(B, L, dtype=torch.long, device=dev),
        sym_id=torch.zeros(B, L, dtype=torch.long, device=dev),
        is_nterm=torch.zeros(B, L, dtype=torch.bool, device=dev),
        is_cterm=torch.zeros(B, L, dtype=torch.bool, device=dev),
        x_clean=torch.randn(B, L, A, 3, device=dev),
        x_t=torch.randn(B, L, A, 3, device=dev),
        eps=torch.randn(B, L, A, 3, device=dev),
        t=torch.rand(B, 1, 1, 1, device=dev),
        esm=torch.randn(B, L, 32, device=dev),
    )

    model = MambaFoldAllAtom(
        d_res=64,
        d_plm=32,
        d_plm_proj=32,
        d_ca_emb=32,
        n_trunk=1,
        d_pair=32,
        n_pair_blocks=1,
        n_pair_heads=2,
        pair_mult_c=32,
        use_plm=True,
        d_state=16,
        mimo_rank=2,
        headdim=32,
    ).to(dev)

    out = model(batch, return_aux=True)

    assert out["v_atom"].shape == (B, L, A, 3)
    assert out["v_ca"].shape == (B, L, 3)
    assert out["distogram_logits"].shape[:3] == (B, L, L)
