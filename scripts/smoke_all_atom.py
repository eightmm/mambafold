#!/usr/bin/env python
"""GPU smoke test for the direct all-atom MambaFold model."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if "TMPDIR" not in os.environ:
    tmp = ROOT / ".cache" / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp)
    tempfile.tempdir = str(tmp)

import torch  # noqa: E402

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mambafold.data.types import ProteinBatch
from mambafold.model.fold import MambaFoldAllAtom


def make_dummy_batch(B: int, L: int, A: int, d_plm: int, device: str) -> ProteinBatch:
    return ProteinBatch(
        res_type=torch.zeros(B, L, dtype=torch.long, device=device),
        res_seq_nums=torch.arange(L, device=device).unsqueeze(0).expand(B, L).contiguous(),
        atom_type=torch.zeros(B, L, A, dtype=torch.long, device=device),
        pair_type=torch.zeros(B, L, A, dtype=torch.long, device=device),
        res_mask=torch.ones(B, L, dtype=torch.bool, device=device),
        atom_mask=torch.ones(B, L, A, dtype=torch.bool, device=device),
        valid_mask=torch.ones(B, L, A, dtype=torch.bool, device=device),
        ca_mask=torch.ones(B, L, dtype=torch.bool, device=device),
        chain_id=torch.zeros(B, L, dtype=torch.long, device=device),
        entity_id=torch.zeros(B, L, dtype=torch.long, device=device),
        sym_id=torch.zeros(B, L, dtype=torch.long, device=device),
        is_nterm=torch.zeros(B, L, dtype=torch.bool, device=device),
        is_cterm=torch.zeros(B, L, dtype=torch.bool, device=device),
        x_clean=torch.randn(B, L, A, 3, device=device),
        x_t=torch.randn(B, L, A, 3, device=device),
        eps=torch.randn(B, L, A, 3, device=device),
        t=torch.rand(B, 1, 1, 1, device=device),
        esm=torch.randn(B, L, d_plm, device=device),
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[smoke] CUDA not available; Mamba kernels require CUDA.")
        return

    B, L, A, d_plm = 2, 16, 15, 16
    model = MambaFoldAllAtom(
        d_res=128, n_trunk=2,
        d_res_type=16, d_res_pos=16,
        d_plm=d_plm, d_plm_proj=16, d_ca_emb=32,
        d_pair=32, n_pair_blocks=2, n_pair_heads=2, pair_mult_c=32,
        mimo_rank=2,
    ).to(device)
    batch = make_dummy_batch(B, L, A, d_plm, device)

    out = model(batch, return_aux=True)
    assert out["v_atom"].shape == (B, L, A, 3), out["v_atom"].shape
    assert out["v_ca"].shape == (B, L, 3), out["v_ca"].shape
    assert out["distogram_logits"].shape[:3] == (B, L, L), out["distogram_logits"].shape
    loss = sum(v.float().pow(2).sum() for v in out.values() if torch.is_tensor(v))
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not no_grad, f"params without grad: {no_grad[:5]}..."

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[smoke] OK - v_atom={tuple(out['v_atom'].shape)} params={n_params:.2f}M")


if __name__ == "__main__":
    main()
