#!/usr/bin/env python
"""GPU smoke for MambaFoldStage2 and TwoStageMambaFold.

Verifies:
    1. Stage 2 forward + backward in isolation (S1 outputs are mocked).
    2. TwoStageMambaFold forward in Phase-2 (frozen S1) mode — Stage 2 grads
       flow, Stage 1 grads are None.
    3. TwoStageMambaFold forward in Phase-3 (joint) mode — gradients reach
       both stages.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# TileLang JIT needs an exec-capable tmpdir.
_ROOT = Path(__file__).resolve().parents[1]
if "TMPDIR" not in os.environ:
    _tmp = _ROOT / ".cache" / "tmp"
    _tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(_tmp)
    tempfile.tempdir = str(_tmp)

import torch  # noqa: E402

SRC = _ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mambafold.data.types import ProteinBatch  # noqa: E402
from mambafold.model.fold import (  # noqa: E402
    MambaFoldStage1,
    MambaFoldStage2,
    TwoStageMambaFold,
)


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

    B, L, A = 2, 16, 15
    d_plm = 16
    # Use a small Stage 1 just to produce s1_ca and s1_latent for the test.
    s1 = MambaFoldStage1(
        d_res=128, n_trunk=2, d_res_type=16, d_res_pos=16,
        d_plm=d_plm, d_plm_proj=16, d_ca_emb=32,
        d_pair=32, n_pair_blocks=2, n_pair_heads=2, d_pair_head=16, pair_mult_c=32,
        mimo_rank=2,
    ).to(device)
    s2 = MambaFoldStage2(
        d_atom=64, d_res_polish=64,
        n_atom_enc=2, n_polish=2, n_atom_dec=2,
        d_s1_res=128, d_ca_anchor=32, d_res_type_atom=16, d_atom_slot=16,
        d_fourier=32, mimo_rank=2,
    ).to(device)
    batch = make_dummy_batch(B, L, A, d_plm, device)

    # ── Test 1: Stage 2 forward + backward in isolation ─────────────────
    s1_ca = torch.randn(B, L, 3, device=device, requires_grad=False)
    s1_latent = torch.randn(B, L, 128, device=device, requires_grad=False)
    v_atom = s2(batch, s1_ca=s1_ca, s1_latent=s1_latent)
    assert v_atom.shape == (B, L, A, 3), v_atom.shape
    v_atom.pow(2).sum().backward()
    s2_no_grad = [n for n, p in s2.named_parameters() if p.grad is None]
    assert not s2_no_grad, f"S2 params without grad: {s2_no_grad[:5]}"
    print(f"[1] Stage 2 isolated: v_atom {tuple(v_atom.shape)}  "
          f"params={sum(p.numel() for p in s2.parameters())/1e6:.2f}M  "
          f"all grads received ✓")

    # ── Test 2: TwoStage Phase-2 (frozen S1) — only S2 grads ────────────
    for p in s2.parameters():
        if p.grad is not None:
            p.grad = None
    for p in s1.parameters():
        if p.grad is not None:
            p.grad = None
    two_stage_p2 = TwoStageMambaFold(s1, s2, freeze_stage1=True).to(device)
    out = two_stage_p2(batch)
    loss = out["v_atom"].pow(2).sum()
    loss.backward()
    s1_no_grad = sum(1 for p in s1.parameters() if p.grad is None)
    s2_with_grad = sum(1 for p in s2.parameters() if p.grad is not None)
    assert s1_no_grad == sum(1 for _ in s1.parameters()), \
        "frozen S1 should have NO grads in Phase 2"
    assert s2_with_grad > 0, "S2 should have grads in Phase 2"
    print(f"[2] Phase-2 (S1 frozen): S1 params no-grad = {s1_no_grad} (= total), "
          f"S2 params with-grad = {s2_with_grad} ✓")

    # ── Test 3: TwoStage Phase-3 (joint) — both stages get grads ─────────
    for p in s1.parameters():
        if p.grad is not None:
            p.grad = None
    for p in s2.parameters():
        if p.grad is not None:
            p.grad = None
    two_stage_p3 = TwoStageMambaFold(s1, s2, freeze_stage1=False).to(device)
    out = two_stage_p3(batch)
    loss = out["v_atom"].pow(2).sum() + out["v_ca"].pow(2).sum()
    loss.backward()
    s1_with_grad = sum(1 for p in s1.parameters() if p.grad is not None)
    s2_with_grad = sum(1 for p in s2.parameters() if p.grad is not None)
    assert s1_with_grad > 0 and s2_with_grad > 0, \
        f"Phase 3 should grad both stages — S1 {s1_with_grad}, S2 {s2_with_grad}"
    print(f"[3] Phase-3 (joint): S1 with-grad = {s1_with_grad}, "
          f"S2 with-grad = {s2_with_grad} ✓")
    print()
    print(f"[smoke] ALL OK. Stage 1+2 total params (small smoke): "
          f"{sum(p.numel() for p in s1.parameters() if p.requires_grad)/1e6 + sum(p.numel() for p in s2.parameters())/1e6:.2f}M")


if __name__ == "__main__":
    main()
