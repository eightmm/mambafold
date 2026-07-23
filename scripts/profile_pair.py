#!/usr/bin/env python
"""GPU memory + timing profile for PairBlock.

Run with B200 idle. Records peak memory and fwd+bwd time for the
production dims at L=1024 and a stack of N blocks.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src uv run python scripts/profile_pair.py \\
        [--L 1024] [--d_pair 192] [--n_heads 4] [--d_head 48] [--n_blocks 6] \\
        [--batch 1] [--dtype bf16]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mambafold.model.fold.pair_blocks import PairBlock


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--d_pair", type=int, default=192)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_head", type=int, default=48)
    ap.add_argument("--mult_c", type=int, default=128)
    ap.add_argument("--n_blocks", type=int, default=6)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
    ap.add_argument("--no_backward", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; aborting profile.")
        return

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = "cuda"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(
        f"[profile] L={args.L} d_pair={args.d_pair} n_heads={args.n_heads} "
        f"d_head={args.d_head} mult_c={args.mult_c} "
        f"n_blocks={args.n_blocks} batch={args.batch} dtype={args.dtype}"
    )

    # Stack of PairBlocks.
    blocks = torch.nn.ModuleList(
        [
            PairBlock(
                d_pair=args.d_pair, n_heads=args.n_heads, d_head=args.d_head, mult_c=args.mult_c
            )
            for _ in range(args.n_blocks)
        ]
    ).to(device=device, dtype=dtype)
    n_params = sum(p.numel() for p in blocks.parameters()) / 1e6
    print(f"[profile] pair stack params: {n_params:.2f}M")

    # Build pair + mask
    pair = torch.randn(
        args.batch,
        args.L,
        args.L,
        args.d_pair,
        device=device,
        dtype=dtype,
        requires_grad=not args.no_backward,
    )
    mask = torch.ones(args.batch, args.L, args.L, device=device, dtype=torch.bool)

    # Warmup
    with torch.no_grad():
        x = pair
        for blk in blocks:
            x = blk(x, mask)
    torch.cuda.synchronize()

    # Forward timing
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    x = pair
    for blk in blocks:
        x = blk(x, mask)
    torch.cuda.synchronize()
    t_fwd = (time.perf_counter() - t0) * 1000

    peak_fwd_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"[profile] fwd: {t_fwd:.1f} ms   peak mem (fwd): {peak_fwd_gb:.2f} GB")

    if not args.no_backward:
        torch.cuda.reset_peak_memory_stats()
        t1 = time.perf_counter()
        loss = x.sum()
        loss.backward()
        torch.cuda.synchronize()
        t_bwd = (time.perf_counter() - t1) * 1000
        peak_total_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"[profile] bwd: {t_bwd:.1f} ms   peak mem (fwd+bwd): {peak_total_gb:.2f} GB")
        print(f"[profile] total step: {t_fwd + t_bwd:.1f} ms")

    print()
    if args.no_backward:
        print(
            f"[summary] L={args.L} | params={n_params:.1f}M | "
            f"peak GB={peak_fwd_gb:.2f} (fwd-only) | fwd ms={t_fwd:.0f}"
        )
    else:
        print(
            f"[summary] L={args.L} | params={n_params:.1f}M | "
            f"peak GB={peak_total_gb:.2f} | fwd+bwd ms={t_fwd + t_bwd:.0f}"
        )


if __name__ == "__main__":
    main()
