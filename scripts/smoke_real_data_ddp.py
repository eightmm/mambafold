#!/usr/bin/env python
"""Exercise real training data on every DDP rank without building the model."""

import argparse
import resource
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mambafold.data.loader import build_dataloaders  # noqa: E402
from mambafold.train.config import parse_args  # noqa: E402
from mambafold.train.distributed import (  # noqa: E402
    any_rank_true,
    distributed_max_int,
    resolve_dataloader_workers,
    setup_dist,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/direct_allatom_puremamba_attn6_geo_adaln_sf360_esmc6b.yaml",
    )
    parser.add_argument("--batches", type=int, default=4)
    cli = parser.parse_args()

    is_dist, rank, world_size, device = setup_dist()
    args, _ = parse_args(["--config", cli.config])
    requested_workers = args.num_workers
    args.num_workers, available_cpus, cpu_source = resolve_dataloader_workers(
        requested_workers, world_size
    )

    if rank == 0:
        print(
            f"[preflight] ranks={world_size} requested_workers={requested_workers}/rank "
            f"effective_workers={args.num_workers}/rank available_cpus={available_cpus} "
            f"source={cpu_source}",
            flush=True,
        )

    loader, _, _, dataset = build_dataloaders(args, is_dist)
    iterator = iter(loader)
    started = time.monotonic()
    for batch_idx in range(cli.batches):
        batch = next(iterator)
        missing = batch is None
        if is_dist:
            missing = any_rank_true(missing, device)
        if missing:
            raise RuntimeError(f"invalid batch at preflight index {batch_idx}")
        if batch.esm is None or batch.esm.ndim != 3 or batch.esm.shape[-1] != args.d_plm:
            shape = None if batch.esm is None else tuple(batch.esm.shape)
            raise RuntimeError(
                f"rank {rank} invalid ESMC batch shape={shape}; expected d_plm={args.d_plm}"
            )
        local_length = batch.max_len
        global_length = (
            distributed_max_int(local_length, device) if is_dist else local_length
        )
        batch = batch.pad_to_length(global_length)
        if batch.max_len != global_length:
            raise RuntimeError(
                f"rank {rank} failed shape sync: {batch.max_len} != {global_length}"
            )
        print(
            f"[preflight] rank={rank} batch={batch_idx + 1}/{cli.batches} "
            f"local_L={local_length} global_L={global_length} "
            f"esm_dtype={batch.esm.dtype}",
            flush=True,
        )

    rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    rss = torch.tensor([rss_gib], device=device)
    if is_dist:
        dist.all_reduce(rss, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(
            f"[preflight] PASS dataset={len(dataset)} batches={cli.batches} "
            f"elapsed={time.monotonic() - started:.1f}s max_main_rss={rss.item():.2f}GiB",
            flush=True,
        )
    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
