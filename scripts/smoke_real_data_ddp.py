#!/usr/bin/env python
"""Exercise real training data on every DDP rank without building the model."""

import argparse
import os
import resource
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mambafold.data.loader import MixedRCSBDataset, build_dataloaders  # noqa: E402
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
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional per-rank worker override for loader diagnosis.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help="Optional DataLoader prefetch override for loader diagnosis.",
    )
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument(
        "--esm-io-mode",
        choices=("eager", "mmap"),
        default="mmap",
        help="Diagnostic comparison for sequential eager reads versus mmap slices.",
    )
    parser.add_argument("--quiet-batches", action="store_true")
    cli = parser.parse_args()

    is_dist, rank, world_size, device = setup_dist()
    args, _ = parse_args(["--config", cli.config])
    requested_workers = args.num_workers
    args.num_workers, available_cpus, cpu_source = resolve_dataloader_workers(
        requested_workers, world_size
    )
    if cli.workers is not None:
        if cli.workers < 0:
            raise ValueError("--workers must be >= 0")
        args.num_workers = min(cli.workers, args.num_workers)
    if cli.prefetch_factor is not None:
        if cli.prefetch_factor < 1:
            raise ValueError("--prefetch-factor must be >= 1")
        args.prefetch_factor = cli.prefetch_factor
    if cli.epoch < 0 or cli.start_batch < 0:
        raise ValueError("--epoch and --start-batch must be non-negative")

    # Keep the legacy mmap path available only in this diagnostic entry point,
    # so one test-node job can compare I/O modes without changing production
    # configuration or duplicating the dataset implementation.
    if cli.esm_io_mode == "eager":
        import mambafold.data.dataset as dataset_module

        original_np_load = dataset_module.np.load

        def benchmark_np_load(file, *load_args, **load_kwargs):
            if (
                isinstance(file, (str, os.PathLike))
                and Path(file).suffix == ".npy"
                and load_kwargs.get("mmap_mode") == "r"
            ):
                load_kwargs.pop("mmap_mode")
                load_kwargs["allow_pickle"] = False
            return original_np_load(file, *load_args, **load_kwargs)

        dataset_module.np.load = benchmark_np_load

    if rank == 0:
        print(
            f"[preflight] ranks={world_size} requested_workers={requested_workers}/rank "
            f"effective_workers={args.num_workers}/rank available_cpus={available_cpus} "
            f"source={cpu_source} prefetch={args.prefetch_factor} "
            f"esm_io={cli.esm_io_mode} epoch={cli.epoch} start_batch={cli.start_batch}",
            flush=True,
        )

    loader, sampler, _, dataset = build_dataloaders(args, is_dist)
    if sampler is not None:
        sampler.set_epoch(cli.epoch)
        if hasattr(sampler, "set_start_batch"):
            sampler.set_start_batch(cli.start_batch)
    planned_iterator = iter(loader.batch_sampler)
    iterator = iter(loader)
    started = time.monotonic()
    load_times = []
    for batch_idx in range(cli.batches):
        planned_indices = next(planned_iterator)
        load_started = time.monotonic()
        batch = next(iterator)
        load_seconds = time.monotonic() - load_started
        load_times.append(load_seconds)
        missing = batch is None
        if is_dist:
            missing = any_rank_true(missing, device)
        if missing:
            raise RuntimeError(f"invalid batch at preflight index {batch_idx}")
        local_batch_size = int(batch.res_type.shape[0])
        size_mismatch = local_batch_size != args.batch_size
        if is_dist:
            size_mismatch = any_rank_true(size_mismatch, device)
        if size_mismatch:
            refs = []
            if local_batch_size != args.batch_size:
                for idx in planned_indices:
                    ds = dataset
                    local_idx = idx
                    source = type(dataset).__name__
                    if isinstance(dataset, MixedRCSBDataset):
                        source_idx, local_idx = dataset._loc(idx)
                        ds = dataset.datasets[source_idx]
                        source = dataset.names[source_idx]
                    if ds.extract_monomer_chains and ds.chain_index is not None:
                        file_idx, origin, _ = ds.chain_index[local_idx]
                        refs.append(f"{source}:{ds.files[file_idx]}:chain{origin}")
                    else:
                        refs.append(f"{source}:{ds.files[local_idx]}")
            raise RuntimeError(
                f"rank {rank} batch {batch_idx + 1} dropped "
                f"{args.batch_size - local_batch_size}/{args.batch_size} samples; "
                f"planned={refs}"
            )
        invalid_esm = batch.esm is None or batch.esm.ndim != 3 or batch.esm.shape[-1] != args.d_plm
        if is_dist:
            invalid_esm = any_rank_true(invalid_esm, device)
        if invalid_esm:
            shape = None if batch.esm is None else tuple(batch.esm.shape)
            raise RuntimeError(
                f"rank {rank} invalid ESMC batch shape={shape}; expected d_plm={args.d_plm}"
            )
        local_length = batch.max_len
        global_length = distributed_max_int(local_length, device) if is_dist else local_length
        batch = batch.pad_to_length(global_length)
        invalid_padding = batch.max_len != global_length
        if is_dist:
            invalid_padding = any_rank_true(invalid_padding, device)
        if invalid_padding:
            raise RuntimeError(f"rank {rank} failed shape sync: {batch.max_len} != {global_length}")
        if not cli.quiet_batches:
            print(
                f"[preflight] rank={rank} batch={batch_idx + 1}/{cli.batches} "
                f"local_L={local_length} global_L={global_length} "
                f"esm_dtype={batch.esm.dtype} load_s={load_seconds:.2f}",
                flush=True,
            )

    rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    rss = torch.tensor([rss_gib], device=device)
    if is_dist:
        dist.all_reduce(rss, op=dist.ReduceOp.MAX)
    # Exclude worker startup from the steady-state summary when possible.  The
    # per-batch maximum is the DDP-relevant tail: every rank eventually waits
    # for the slowest loader.
    steady_load_times = load_times[1:] if len(load_times) > 1 else load_times
    local_loads = torch.tensor(steady_load_times, device=device, dtype=torch.float64)
    if is_dist:
        gathered_loads = [torch.zeros_like(local_loads) for _ in range(world_size)]
        dist.all_gather(gathered_loads, local_loads)
        rank_loads = torch.stack(gathered_loads)
    else:
        rank_loads = local_loads.unsqueeze(0)
    if rank == 0:
        tail_loads = rank_loads.max(dim=0).values
        mean_s = float(tail_loads.mean().item())
        median_s = float(tail_loads.median().item())
        p90_s = float(torch.quantile(tail_loads, 0.9).item())
        max_s = float(tail_loads.max().item())
        samples_per_s = args.batch_size * world_size / mean_s
        print(
            f"[preflight] PASS dataset={len(dataset)} batches={cli.batches} "
            f"elapsed={time.monotonic() - started:.1f}s max_main_rss={rss.item():.2f}GiB",
            flush=True,
        )
        print(
            f"[loader-result] esm_io={cli.esm_io_mode} "
            f"prefetch={args.prefetch_factor} workers={args.num_workers}/rank "
            f"epoch={cli.epoch} start_batch={cli.start_batch} "
            f"tail_mean_s={mean_s:.4f} tail_p50_s={median_s:.4f} "
            f"tail_p90_s={p90_s:.4f} tail_max_s={max_s:.4f} "
            f"samples_per_s={samples_per_s:.2f}",
            flush=True,
        )
    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
