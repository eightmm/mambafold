#!/usr/bin/env python
"""MambaFold EqM — full training script (single/multi-GPU DDP).

Single GPU:
    PYTHONPATH=src python -u scripts/train.py --config configs/train_base.yaml

Multi-GPU (torchrun):
    PYTHONPATH=src torchrun --nproc_per_node=4 scripts/train.py \
        --config configs/train_base.yaml

Resume:
    PYTHONPATH=src torchrun --nproc_per_node=4 scripts/train.py \
        --config configs/train_base.yaml \
        --resume outputs/train/run1/ckpt_latest.pt
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mambafold.data.loader import build_dataloaders, inf_loader
from mambafold.train.config import parse_args
from mambafold.train.distributed import GPUMonitor, all_reduce_mean, setup_dist
from mambafold.train.ema import EMA
from mambafold.train.engine import eval_step, train_step
from mambafold.train.logging import init_wandb, log_metrics, log_val_metrics
from mambafold.train.trainer import (
    build_model,
    cosine_warmup_lr,
    load_checkpoint,
    save_checkpoint,
    seed_all,
)


def main():
    # ── distributed init ─────────────────────────────────────────────────────
    is_dist, rank, world_size, device = setup_dist()
    is_main = (rank == 0)
    args, _ = parse_args()

    if is_main:
        print(f"Config: {args.config}")

    # ── output dir ───────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
        print(f"Output dir: {out_dir}")
        print(f"Device: {device} | world_size: {world_size} "
              f"| total_batch: {args.batch_size * world_size}")

    # ── dataset ──────────────────────────────────────────────────────────────
    loader, sampler, val_loader, dataset = build_dataloaders(args, is_dist)
    if is_main:
        print(f"Dataset: {len(dataset)} proteins "
              f"| per-GPU batch={args.batch_size} "
              f"| effective batch={args.batch_size * world_size}")

    # ── model ────────────────────────────────────────────────────────────────
    seed_all(args.seed)
    model = build_model(vars(args), device)

    if is_dist:
        model = DDP(model, device_ids=[int(device.split(":")[-1])],
                    broadcast_buffers=False, find_unused_parameters=False)

    ema = EMA(model.module if is_dist else model, decay=args.ema_decay)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    if is_main:
        print(f"Model: {n_params:.2f}M params")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = cosine_warmup_lr(optimizer, args.warmup_steps, args.total_steps)

    # ── resume ───────────────────────────────────────────────────────────────
    start_step = 0
    resume_run_id = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        # Only resume the wandb run when continuing the same training curve.
        # When --reset_optimizer (stage transition), start a fresh wandb run
        # so the new step counter doesn't clash with the old one.
        if not args.reset_optimizer:
            resume_run_id = ckpt.get("wandb_run_id")
        raw_model = model.module if is_dist else model
        raw_model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        if args.reset_optimizer:
            # Stage transition (e.g. 256→512): keep weights+EMA, fresh
            # optimizer/scheduler with current args (lr, warmup, total_steps).
            start_step = args.start_step
            if is_main:
                print(f"Resumed weights from {args.resume} at ckpt step "
                      f"{ckpt['step']} → fresh optimizer/scheduler, "
                      f"start_step={start_step}", flush=True)
        else:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_step = ckpt["step"]
            if is_main:
                print(f"Resumed full state from {args.resume} at step "
                      f"{start_step}", flush=True)
        del ckpt

    # ── wandb ────────────────────────────────────────────────────────────────
    if is_main:
        init_wandb(args, out_dir, world_size, n_params, len(dataset),
                   resume_run_id=resume_run_id)

    # ── train loop ───────────────────────────────────────────────────────────
    gpu_monitor = GPUMonitor(interval=60) if is_main else None
    if gpu_monitor:
        gpu_monitor.start()

    model.train()
    step = start_step
    metric_sums: dict[str, float] = {}
    metric_count = 0

    try:
        for batch in inf_loader(loader, sampler):
            if step >= args.total_steps:
                break
            if batch is None:
                continue

            batch = batch.to(torch.device(device))
            oom = False
            try:
                metrics = train_step(model, batch, optimizer,
                                     grad_clip=args.grad_clip,
                                     alpha_mode=args.alpha_mode, use_amp=True)
            except torch.cuda.OutOfMemoryError:
                oom = True
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)

            # Check for NaN loss (indicates corrupted batch)
            if not oom and not np.isfinite(metrics["loss"]):
                oom = True  # reuse skip path
                optimizer.zero_grad(set_to_none=True)

            # Sync skip flag across ranks so all skip together
            if is_dist:
                skip_t = torch.tensor([1 if oom else 0], device=device)
                dist.all_reduce(skip_t, op=dist.ReduceOp.MAX)
                oom = skip_t.item() > 0
            if oom:
                if is_main:
                    print(f"[step {step}] OOM/NaN — skipped batch "
                          f"(L={batch.res_mask.shape[1]})", flush=True)
                continue

            scheduler.step()
            ema.update(model.module if is_dist else model)
            step += 1

            # Metric accumulation
            if is_dist:
                t = torch.tensor(metrics["loss"], device=device)
                metrics["loss"] = all_reduce_mean(t)
            for k, v in metrics.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v
            metric_count += 1

            # Logging
            if is_main and step % args.log_interval == 0:
                avgs = {k: v / metric_count for k, v in metric_sums.items()}
                log_metrics(step, args.total_steps, avgs,
                            scheduler.get_last_lr()[0],
                            world_size, args.batch_size,
                            args.copies_per_protein)
                metric_sums, metric_count = {}, 0

            # Validation (rank 0 only; other ranks wait on barrier to avoid DDP desync)
            if args.eval_interval > 0 and step % args.eval_interval == 0:
                if is_main and val_loader:
                    model.eval()
                    val_metrics: dict[str, list[float]] = {}
                    with torch.no_grad():
                        for vbatch in val_loader:
                            if vbatch is None:
                                continue
                            vbatch = vbatch.to(torch.device(device))
                            vm = eval_step(ema.shadow, vbatch, use_amp=True)
                            for k, v in vm.items():
                                val_metrics.setdefault(k, []).append(v)
                    log_val_metrics(step,
                                    {k: float(np.mean(v)) for k, v in val_metrics.items()})
                    model.train()
                if is_dist:
                    dist.barrier()

            # Checkpoint (rank 0 saves; other ranks wait on barrier to avoid DDP desync)
            if step % args.ckpt_interval == 0:
                if is_main:
                    save_checkpoint(out_dir, step, model, ema,
                                    optimizer, scheduler, args)
                if is_dist:
                    dist.barrier()

    finally:
        if gpu_monitor:
            gpu_monitor.stop()

    # ── final ────────────────────────────────────────────────────────────────
    if is_main:
        save_checkpoint(out_dir, step, model, ema, optimizer, scheduler, args)
        if wandb.run is not None:
            wandb.finish()
        print(f"\nDone. Total steps: {step}")

    if is_dist:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
