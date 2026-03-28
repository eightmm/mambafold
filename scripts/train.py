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

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import wandb
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mambafold.data.collate import ProteinCollator
from mambafold.data.dataset import AFDBDataset
from mambafold.data.loader import inf_loader
from mambafold.train.distributed import GPUMonitor, all_reduce_mean, setup_dist
from mambafold.train.ema import EMA
from mambafold.train.engine import eval_step, train_step
from mambafold.train.trainer import (
    build_model,
    cosine_warmup_lr,
    load_checkpoint,
    save_checkpoint,
)


def main():
    # ── distributed init ─────────────────────────────────────────────────────
    is_dist, rank, world_size, device = setup_dist()
    is_main = (rank == 0)

    # ── 1. config ─────────────────────────────────────────────────────────────
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    cfg = {}
    if pre_args.config:
        with open(pre_args.config) as f:
            cfg = yaml.safe_load(f) or {}
        if is_main:
            print(f"Config: {pre_args.config}")

    parser = argparse.ArgumentParser(description="MambaFold full training")
    parser.add_argument("--config", default=None)
    # Data
    parser.add_argument("--data_dir", default="afdb_data/train")
    parser.add_argument("--val_data_dir", default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Per-GPU batch size")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--copies_per_protein", type=int, default=1)
    # Output
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--resume", default=None)
    # Training
    parser.add_argument("--total_steps", type=int, default=200_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=2_000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--ckpt_interval", type=int, default=5_000)
    parser.add_argument("--eval_interval", type=int, default=0)
    parser.add_argument("--gamma_schedule", default="logit_normal")
    # Model
    parser.add_argument("--d_atom", type=int, default=256)
    parser.add_argument("--d_res", type=int, default=256)
    parser.add_argument("--d_state", type=int, default=32)
    parser.add_argument("--mimo_rank", type=int, default=2)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--n_atom_enc", type=int, default=2)
    parser.add_argument("--n_trunk", type=int, default=6)
    parser.add_argument("--n_atom_dec", type=int, default=2)
    parser.add_argument("--d_res_pos", type=int, default=64)
    parser.add_argument("--d_atom_slot", type=int, default=32)
    parser.add_argument("--d_local_frame", type=int, default=64)
    # PLM
    parser.add_argument("--use_plm", action="store_true", default=False)
    parser.add_argument("--d_plm", type=int, default=1024)
    parser.add_argument("--plm_mode", default="blend")
    # W&B
    parser.add_argument("--wandb_project", default="mambafold")
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=[])
    parser.add_argument("--wandb_offline", action="store_true", default=False)
    parser.add_argument("--no_wandb", action="store_true", default=False)

    parser.set_defaults(**cfg)
    args = parser.parse_args()

    # ── 2. output dir (rank 0만 생성) ──────────────────────────────────────────
    if args.out_dir is None:
        job_id = os.environ.get("SLURM_JOB_ID", None)
        tag = job_id if job_id else time.strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"outputs/train/{tag}"
    out_dir = Path(args.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
        print(f"Output dir: {out_dir}")
        print(f"Device: {device} | world_size: {world_size} "
              f"| total_batch: {args.batch_size * world_size}")

    if is_dist:
        dist.barrier()

    # ── 3. dataset ────────────────────────────────────────────────────────────
    dataset = AFDBDataset(data_dir=args.data_dir, max_length=args.max_length)
    collator = ProteinCollator(
        augment=True,
        copies_per_protein=args.copies_per_protein,
        gamma_schedule=args.gamma_schedule,
    )
    sampler = DistributedSampler(dataset, shuffle=True) if is_dist else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )
    if is_main:
        print(f"Dataset: {len(dataset)} proteins "
              f"| per-GPU batch={args.batch_size} "
              f"| effective batch={args.batch_size * world_size}")

    val_loader = None
    if args.val_data_dir and args.eval_interval > 0:
        val_ds = AFDBDataset(data_dir=args.val_data_dir, max_length=args.max_length)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=ProteinCollator(augment=False),
            num_workers=2, drop_last=False,
        )

    # ── 4. model ──────────────────────────────────────────────────────────────
    model = build_model(vars(args), device)

    # LazyLinear 초기화 (PLM 사용 시)
    if args.use_plm:
        if is_main:
            print("Initializing PLM lazy parameters...")
        _tmp = ProteinCollator(augment=False)
        with torch.no_grad():
            _dummy = _tmp([dataset[0]]).to(torch.device(device))
            model(_dummy)
        del _dummy, _tmp
        if is_main:
            print("PLM initialized.")

    # DDP 래핑
    if is_dist:
        model = DDP(model, device_ids=[int(device.split(":")[-1])],
                    find_unused_parameters=False)

    ema = EMA(model.module if is_dist else model, decay=0.999)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    if is_main:
        print(f"Model: {n_params:.2f}M params")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = cosine_warmup_lr(optimizer, args.warmup_steps, args.total_steps)

    # ── 5. resume ─────────────────────────────────────────────────────────────
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(
            Path(args.resume), model, ema, optimizer, scheduler, device
        )

    # ── 6. wandb (rank 0만) ───────────────────────────────────────────────────
    if is_main and not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or out_dir.name,
            tags=args.wandb_tags or [],
            config={**{k: v for k, v in vars(args).items()
                       if not k.startswith("wandb") and k != "no_wandb"},
                    "world_size": world_size,
                    "effective_batch": args.batch_size * world_size},
            mode="offline" if args.wandb_offline else "online",
            resume="allow",
        )
        wandb.config.update({"n_params_M": round(n_params, 2),
                              "n_train": len(dataset)})

    # ── 7. train loop ─────────────────────────────────────────────────────────
    gpu_monitor = GPUMonitor(interval=60) if is_main else None
    if gpu_monitor:
        gpu_monitor.start()

    model.train()
    step = start_step
    loss_sum = 0.0
    loss_count = 0

    if is_main:
        print(f"\nTraining {args.total_steps} steps "
              f"(resume={start_step}, world={world_size})...", flush=True)

    try:
        for batch in inf_loader(loader, sampler):
            if step >= args.total_steps:
                break

            batch = batch.to(torch.device(device))
            metrics = train_step(model, batch, optimizer,
                                 grad_clip=args.grad_clip,
                                 alpha_mode="const",
                                 use_amp=True)
            scheduler.step()
            ema.update(model.module if is_dist else model)
            step += 1

            # loss 집계 (DDP: 각 rank 평균)
            loss_val = metrics["loss"]
            if is_dist:
                t = torch.tensor(loss_val, device=device)
                loss_val = all_reduce_mean(t)
            loss_sum += loss_val
            loss_count += 1

            # ── log (rank 0) ──────────────────────────────────────────────────
            if is_main and step % args.log_interval == 0:
                avg_loss = loss_sum / loss_count
                lr = scheduler.get_last_lr()[0]
                vram = ""
                if torch.cuda.is_available():
                    alloc = torch.cuda.memory_allocated() / 1024**3
                    reserv = torch.cuda.memory_reserved() / 1024**3
                    vram = f" | vram={alloc:.2f}/{reserv:.2f}GB"
                print(f"  step {step:>7d}/{args.total_steps} | "
                      f"avg_loss={avg_loss:.4f} | lr={lr:.2e}{vram}",
                      flush=True)
                if wandb.run is not None:
                    log_d = {"train/loss": loss_val,
                             "train/avg_loss": avg_loss,
                             "train/lr": lr}
                    if torch.cuda.is_available():
                        log_d["gpu/vram_alloc_gb"] = alloc
                        log_d["gpu/vram_reserved_gb"] = reserv
                    wandb.log(log_d, step=step)
                loss_sum = 0.0
                loss_count = 0

            # ── val (rank 0) ──────────────────────────────────────────────────
            if (is_main and val_loader and args.eval_interval > 0
                    and step % args.eval_interval == 0):
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for vbatch in val_loader:
                        vbatch = vbatch.to(torch.device(device))
                        vm = eval_step(ema.shadow, vbatch, use_amp=True)
                        val_losses.append(vm["eqm"])
                val_loss = float(np.mean(val_losses))
                print(f"  [val] step={step} eqm={val_loss:.4f}", flush=True)
                if wandb.run is not None:
                    wandb.log({"val/eqm": val_loss}, step=step)
                model.train()

            # ── checkpoint (rank 0) ───────────────────────────────────────────
            if is_main and step % args.ckpt_interval == 0:
                save_checkpoint(out_dir, step, model, ema,
                                optimizer, scheduler, args)

            if is_dist and step % args.ckpt_interval == 0:
                dist.barrier()

    finally:
        if gpu_monitor:
            gpu_monitor.stop()

    # ── 8. final checkpoint ───────────────────────────────────────────────────
    if is_main:
        save_checkpoint(out_dir, step, model, ema, optimizer, scheduler, args)
        if wandb.run is not None:
            wandb.finish()
        print(f"\nDone. Total steps: {step}")

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
