#!/usr/bin/env python
"""GPU smoke test for MambaFoldStage1 — small L, single forward + backward.

Mamba SSM kernels are CUDA-only, so this is GPU-required. Use a small L
(default 16) to keep it under 1 second.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src uv run python scripts/smoke_stage1.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# TileLang JIT (used inside Mamba kernels) needs an exec-capable tmpdir.
# Match conftest.py's fallback so /tmp (noexec on this host) is not used.
_ROOT = Path(__file__).resolve().parents[1]
if "TMPDIR" not in os.environ:
    _tmp = _ROOT / ".cache" / "tmp"
    _tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(_tmp)
    tempfile.tempdir = str(_tmp)

import torch  # noqa: E402  — must come after TMPDIR env var

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mambafold.data.types import ProteinBatch
from mambafold.model.fold import MambaFoldStage1


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
        print("[smoke] CUDA not available — Mamba SSM kernels are CUDA-only. Aborting.")
        return

    # Small dims for fast smoke. Real Phase 1 uses d_res=1024, n_trunk=12.
    B, L, A = 2, 16, 15
    d_plm = 16
    model = MambaFoldStage1(
        d_res=128, n_trunk=2,
        d_res_type=16, d_res_pos=16,
        d_plm=d_plm, d_plm_proj=16, d_ca_emb=32,
        d_pair=32, n_pair_blocks=2, n_pair_heads=2, d_pair_head=16, pair_mult_c=32,
        mimo_rank=2,
    ).to(device)

    batch = make_dummy_batch(B, L, A, d_plm, device)

    # Lean dict forward (inference-style).
    out = model(batch)
    assert out["v_ca"].shape == (B, L, 3), out["v_ca"].shape
    assert out["trunk_latent"].shape == (B, L, 128), out["trunk_latent"].shape
    assert out["pcb_dir"].shape == (B, L, 3), out["pcb_dir"].shape
    assert out["conf"].shape == (B, L), out["conf"].shape

    # Full forward: return_aux + 2-cycle recycling exercises every head and the
    # recycle distance embedding; one backward must reach all trainable params.
    aux = model(batch, return_aux=True, n_cycles=2)
    assert aux["distogram_logits"].shape[:3] == (B, L, L), aux["distogram_logits"].shape
    assert aux["contact_logits"].shape == (B, L, L), aux["contact_logits"].shape
    loss = (aux["v_ca"].pow(2).sum() + aux["trunk_latent"].pow(2).sum()
            + aux["pcb_dir"].pow(2).sum() + aux["conf"].sum()
            + aux["distogram_logits"].pow(2).sum() + aux["contact_logits"].pow(2).sum())
    loss.backward()

    no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not no_grad, f"params without grad: {no_grad[:5]}..."

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[smoke] OK — v_ca {tuple(out['v_ca'].shape)}, pcb_dir {tuple(out['pcb_dir'].shape)}, "
          f"conf {tuple(out['conf'].shape)}, contact {tuple(aux['contact_logits'].shape)}")
    print(f"[smoke] params (small smoke config): {n_params:.2f}M")
    print(f"[smoke] all {sum(1 for _ in model.parameters())} params received gradients (2-cycle)")


if __name__ == "__main__":
    main()
