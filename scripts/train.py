#!/usr/bin/env python
"""MambaFold — full training script (single/multi-GPU DDP).

Preferred launcher (sets NETRC / NCCL / CUDA_VISIBLE_DEVICES for you):
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh

Direct torchrun (equivalent):
    PYTHONPATH=src torchrun --nproc_per_node=4 scripts/train.py \
        --config configs/direct_allatom_360m.yaml

Resume:
    RESUME=outputs/train/run1/ckpt_latest.pt \
        CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh
"""

import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mambafold.data.loader import build_dataloaders, inf_loader
from mambafold.train.config import parse_args
from mambafold.train.crop_schedule import pick_crop_length
from mambafold.train.distributed import (
    GPUMonitor,
    all_reduce_mean,
    enable_cuda_perf_flags,
    setup_dist,
)
from mambafold.train.ema import EMA
from mambafold.train.engine import (
    allatom_eval_step,
    allatom_forward_and_loss,
)
from mambafold.train.logging import init_wandb, log_metrics, log_val_metrics
from mambafold.train.trainer import (
    build_model,
    cosine_warmup_lr,
    save_checkpoint,
    seed_all,
)


def main():
    # ── distributed init ─────────────────────────────────────────────────────
    is_dist, rank, world_size, device = setup_dist()
    is_main = (rank == 0)
    args, _ = parse_args()
    enable_cuda_perf_flags()

    if is_main:
        print(f"Config: {args.config}")

    # ── output dir ───────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
        print(f"Output dir: {out_dir}")
        print(f"Device: {device} | world_size: {world_size} "
              f"| total_batch: {args.batch_size * world_size * args.grad_accum_steps}")

    # ── dataset ──────────────────────────────────────────────────────────────
    loader, sampler, val_loader, dataset = build_dataloaders(args, is_dist)
    if is_main:
        print(f"Dataset: {len(dataset)} proteins "
              f"| per-GPU batch={args.batch_size} "
              f"| grad_accum={args.grad_accum_steps} "
              f"| effective batch={args.batch_size * world_size * args.grad_accum_steps}")

    # ── model ────────────────────────────────────────────────────────────────
    seed_all(args.seed)
    model = build_model(vars(args), device)

    if is_dist:
        model = DDP(
            model,
            device_ids=[int(device.split(":")[-1])],
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            # NOTE: static_graph=True is incompatible with the grad-accum
            # `model.no_sync()` path below (triggers reducer.cpp
            # `expect_autograd_hooks_` assert on the first micro-step backward).
        )

    ema = EMA(model.module if is_dist else model, decay=args.ema_decay)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    if is_main:
        print(f"Model: {n_params:.2f}M params")

    # Only optimize trainable params.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=1e-2,
        fused=torch.cuda.is_available(),
    )
    scheduler = cosine_warmup_lr(optimizer, args.warmup_steps, args.total_steps)

    # ── resume ───────────────────────────────────────────────────────────────
    start_step = 0
    resume_run_id = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        # Only resume the wandb run when continuing the same training curve.
        # When resetting optimizer/scheduler, start a fresh wandb run so the new
        # step counter does not clash with the old one.
        if not args.reset_optimizer:
            resume_run_id = ckpt.get("wandb_run_id")
        raw_model = model.module if is_dist else model
        missing, unexpected = raw_model.load_state_dict(ckpt["model"], strict=False)
        ema_missing, ema_unexpected = ema.load_state_dict(ckpt["ema"], strict=False)
        if is_main and (missing or unexpected):
            print(f"  [resume] model missing={len(missing)} unexpected={len(unexpected)}", flush=True)
            if missing:
                print(f"    missing: {missing[:5]}{'...' if len(missing) > 5 else ''}", flush=True)
            if unexpected:
                print(f"    unexpected: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}", flush=True)
        if is_main and (ema_missing or ema_unexpected):
            print(f"  [resume] ema missing={len(ema_missing)} unexpected={len(ema_unexpected)}", flush=True)
        if args.reset_optimizer:
            # Keep weights+EMA, fresh optimizer/scheduler with current args
            # (lr, warmup, total_steps).
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
    grad_accum = max(1, int(getattr(args, "grad_accum_steps", 1)))
    crop_schedule = getattr(args, "crop_schedule", None)
    loader_iter = iter(inf_loader(loader, sampler))

    try:
        while step < args.total_steps:
            target_L = pick_crop_length(step, crop_schedule, args.max_length)

            optimizer.zero_grad(set_to_none=True)
            accum: dict[str, float] = {}
            oom = False
            for micro_idx in range(grad_accum):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(inf_loader(loader, sampler))
                    batch = next(loader_iter)
                if batch is None:
                    oom = True
                    break
                batch = batch.to(torch.device(device)).truncate_length(target_L)

                is_last = (micro_idx == grad_accum - 1)
                # Skip DDP all-reduce on intermediate micro-steps; final micro
                # synchronizes the accumulated grads across ranks once.
                sync_ctx = (model.no_sync() if (is_dist and not is_last)
                            else nullcontext())
                try:
                    with sync_ctx:
                        loss, m = allatom_forward_and_loss(
                            model, batch,
                            alpha_mode=args.alpha_mode, use_amp=True,
                            w_fm=getattr(args, "w_fm", 1.0),
                            w_lddt_atom=getattr(args, "w_lddt_atom", 1.0),
                            w_lddt_ca=getattr(args, "w_lddt_ca", 0.5),
                            w_bond=getattr(args, "w_bond", 0.05),
                            w_clash=getattr(args, "w_clash", 0.02),
                            w_ca_clash=getattr(args, "w_ca_clash", 0.01),
                            w_distogram=getattr(args, "w_distogram", 0.5),
                            w_drmsd=getattr(args, "w_drmsd", 0.75),
                            w_contact=getattr(args, "w_contact", 0.5),
                            w_pcb=getattr(args, "w_pcb", 0.2),
                            w_conf=getattr(args, "w_conf", 0.05),
                            w_ca_angle=getattr(args, "w_ca_angle", 0.1),
                            w_ca_self_clash=getattr(args, "w_ca_self_clash", 0.1),
                            w_chirality=getattr(args, "w_chirality", 1.0),
                            w_chirality_atom=getattr(args, "w_chirality_atom", 0.5),
                            max_lddt_atoms=getattr(args, "max_lddt_atoms", 2048),
                            max_clash_atoms=getattr(args, "max_clash_atoms", 2048),
                        )
                        if not torch.isfinite(loss):
                            oom = True
                            break
                        (loss / grad_accum).backward()
                except torch.cuda.OutOfMemoryError:
                    oom = True
                    torch.cuda.empty_cache()
                    break
                for k, v in m.items():
                    accum[k] = accum.get(k, 0.0) + v / grad_accum

            if oom:
                optimizer.zero_grad(set_to_none=True)

            # Sync skip across ranks so all rank skip together
            if is_dist:
                skip_t = torch.tensor([1 if oom else 0], device=device)
                dist.all_reduce(skip_t, op=dist.ReduceOp.MAX)
                oom = skip_t.item() > 0
            if oom:
                if is_main:
                    print(f"[step {step}] OOM/NaN — skipped (target_L={target_L})",
                          flush=True)
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip).item()
            skipped = (not (grad_norm < 1e4)) or (grad_norm != grad_norm)
            if not skipped:
                optimizer.step()
            scheduler.step()
            if not skipped:
                ema.update(model.module if is_dist else model)
            elif is_main:
                print(f"[step {step}] gnorm spike skipped "
                      f"(gnorm={grad_norm:.2e})", flush=True)

            accum["grad_norm"] = grad_norm
            accum["target_L"] = float(target_L)
            step += 1

            # Metric accumulation
            if is_dist:
                for k, v in list(accum.items()):
                    t = torch.tensor(v, device=device)
                    accum[k] = all_reduce_mean(t)
            for k, v in accum.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v
            metric_count += 1

            # Logging
            if is_main and step % args.log_interval == 0:
                avgs = {k: v / metric_count for k, v in metric_sums.items()}
                log_metrics(step, args.total_steps, avgs,
                            scheduler.get_last_lr()[0],
                            world_size, args.batch_size,
                            args.copies_per_protein,
                            args.grad_accum_steps)
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
                            vm = allatom_eval_step(
                                ema.shadow, vbatch, use_amp=True,
                                max_lddt_atoms=getattr(args, "max_lddt_atoms", 2048),
                                max_clash_atoms=getattr(args, "max_clash_atoms", 2048),
                            )
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
