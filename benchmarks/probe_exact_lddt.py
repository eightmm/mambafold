#!/usr/bin/env python
"""Measure chunked exact all-atom lDDT time and peak accelerator memory."""

from __future__ import annotations

import argparse
import json
import time

import torch

from mambafold.data.constants import COORD_SCALE
from mambafold.losses.lddt import soft_lddt_all_atom_loss


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atom-counts", type=int, nargs="+", default=[2048, 4096, 8192])
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--density", type=float, default=0.05, help="Synthetic atoms per Å^3")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_memory_mib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def _run_one(
    num_atoms: int,
    *,
    batch_size: int,
    chunk_size: int,
    density: float,
    repeats: int,
    device: torch.device,
) -> dict[str, float | int | str | None]:
    generator = torch.Generator(device=device).manual_seed(17 + num_atoms)
    side_A = (num_atoms / density) ** (1.0 / 3.0)
    true = torch.rand((batch_size, 1, num_atoms, 3), generator=generator, device=device)
    true = true * (side_A / COORD_SCALE)
    pred = true + 0.5 * torch.randn(
        true.shape,
        generator=generator,
        device=device,
    ) / COORD_SCALE
    mask = torch.ones((batch_size, 1, num_atoms), dtype=torch.bool, device=device)

    forward_times = []
    backward_times = []
    peak_memory = []
    final_loss = None
    for _ in range(repeats):
        current_pred = pred.detach().clone().requires_grad_(True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        started = time.perf_counter()
        loss = soft_lddt_all_atom_loss(
            current_pred,
            true,
            mask,
            pair_chunk_size=chunk_size,
        )
        _sync(device)
        forward_times.append(time.perf_counter() - started)

        started = time.perf_counter()
        loss.backward()
        _sync(device)
        backward_times.append(time.perf_counter() - started)
        peak = _peak_memory_mib(device)
        if peak is not None:
            peak_memory.append(peak)
        final_loss = float(loss.detach())

    return {
        "device": str(device),
        "batch_size": batch_size,
        "atoms": num_atoms,
        "chunk_size": chunk_size,
        "density_atoms_per_A3": density,
        "loss": final_loss,
        "forward_ms": 1000.0 * min(forward_times),
        "backward_ms": 1000.0 * min(backward_times),
        "total_ms": 1000.0 * min(f + b for f, b in zip(forward_times, backward_times)),
        "peak_memory_MiB": max(peak_memory) if peak_memory else None,
    }


def main() -> None:
    args = _parser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.density <= 0:
        raise ValueError("--density must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    for atom_count in args.atom_counts:
        result = _run_one(
            atom_count,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            density=args.density,
            repeats=args.repeats,
            device=device,
        )
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
