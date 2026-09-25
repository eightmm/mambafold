#!/usr/bin/env python
"""MambaFold-clean training entry point (single- or multi-GPU DDP).

Single GPU:
    PYTHONPATH=src python scripts/train.py --config configs/run_a_mamba.yaml

Multi-GPU:
    PYTHONPATH=src torchrun --nproc_per_node=8 scripts/train.py \
        --config configs/run_a_mamba.yaml
"""

import json
import os
import signal
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mambafold.data.loader import build_dataloaders, inf_loader
from mambafold.train.config import parse_args
from mambafold.train.distributed import (
    GPUMonitor,
    any_rank_true,
    distributed_max_int,
    enable_cuda_perf_flags,
    resolve_dataloader_workers,
    setup_dist,
)
from mambafold.train.ema import EMA
from mambafold.train.engine import (
    allatom_forward_and_loss,
)
from mambafold.train.logging import init_wandb, log_metrics
from mambafold.train.trainer import (
    build_model,
    capture_rng_state,
    linear_warmup_lr,
    prewarm_sequence_kernels,
    restore_rng_state,
    save_checkpoint,
    seed_all,
    validate_data_resume_state,
)


def main():
    # ── distributed init ─────────────────────────────────────────────────────
    is_dist, rank, world_size, device = setup_dist()
    is_main = rank == 0
    args, _ = parse_args()
    enable_cuda_perf_flags()
    preempt_state = {"requested": False}

    def _request_preemption(signum, _frame):
        # Keep the signal handler free of I/O. GPUMonitor writes from another
        # thread, so printing here can deadlock on Python's buffered stdout
        # lock if USR1 arrives during a monitor update.
        preempt_state["requested"] = True
        preempt_state["signal"] = signum

    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _request_preemption)

    requested_workers = args.num_workers
    args.num_workers, allocated_cpus, cpu_source = resolve_dataloader_workers(
        requested_workers, world_size
    )
    args.requested_num_workers = requested_workers

    if is_main:
        print(f"Config: {args.config}")
        print(
            "DataLoader workers: "
            f"requested={requested_workers}/rank effective={args.num_workers}/rank "
            f"available_cpus={allocated_cpus} source={cpu_source}",
            flush=True,
        )

    # ── output dir ───────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / ".rank_pids").mkdir(exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
        print(f"Output dir: {out_dir}")
        print(
            f"Device: {device} | world_size: {world_size} "
            f"| total_batch: {args.batch_size * world_size * args.grad_accum_steps}"
        )
    if is_dist:
        dist.barrier()
    (out_dir / ".rank_pids" / str(rank)).write_text(f"{os.getpid()}\n")

    # ── dataset ──────────────────────────────────────────────────────────────
    loader, sampler, dataset = build_dataloaders(args, is_dist)
    batches_per_epoch = len(loader)
    if is_main:
        print(
            f"Dataset: {len(dataset)} proteins "
            f"| per-GPU batch={args.batch_size} "
            f"| grad_accum={args.grad_accum_steps} "
            f"| effective batch={args.batch_size * world_size * args.grad_accum_steps}"
        )

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
        raw_model = model.module if is_dist else model
        print(
            "Architecture: "
            f"self_conditioning={getattr(raw_model, 'self_conditioning', False)} "
            f"self_condition_prob={args.self_condition_prob}",
            flush=True,
        )

    # Only optimize trainable params.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=torch.cuda.is_available(),
    )
    scheduler = linear_warmup_lr(
        optimizer, args.warmup_steps, args.lr, args.min_lr,
        total_steps=args.total_steps, cooldown_steps=args.lr_cooldown_steps,
    )

    if args.prewarm_kernels and device.startswith("cuda"):
        # Every rank runs the same shapes, so this stays in lockstep and no rank
        # can be stranded in a collective while another compiles.
        t_warm = time.time()
        warmed = prewarm_sequence_kernels(
            model,
            batch_size=args.batch_size * max(1, int(args.copies_per_protein)),
            max_length=args.max_length,
            length_bin=args.length_bin,
            device=device,
        )
        if is_main:
            print(
                f"[prewarm] compiled {len(warmed)} sequence shapes "
                f"({warmed[0] if warmed else '-'}..{warmed[-1] if warmed else '-'}) "
                f"in {time.time() - t_warm:.1f}s",
                flush=True,
            )

    # ── resume ───────────────────────────────────────────────────────────────
    start_step = 0
    micro_batches_consumed = 0
    resume_run_id = None
    resume_rng_state = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (
            args.expected_resume_step is not None
            and int(ckpt.get("step", -1)) != args.expected_resume_step
        ):
            raise RuntimeError(
                "Resume checkpoint step mismatch: "
                f"expected={args.expected_resume_step} "
                f"actual={ckpt.get('step')!r} path={args.resume}"
            )
        # Only resume the wandb run when continuing the same training curve.
        # When resetting optimizer/scheduler, start a fresh wandb run so the new
        # step counter does not clash with the old one.
        if not args.reset_optimizer:
            resume_run_id = ckpt.get("wandb_run_id")
        raw_model = model.module if is_dist else model
        model_state_key = "ema" if args.initialize_model_from_ema else "model"
        # Load non-strictly first so an explicitly declared architecture
        # transition can still be checked exactly. This keeps --strict_resume
        # strict: absent a declaration both expected missing-key sets are empty,
        # reproducing torch's strict=True contract while giving a useful list in
        # the error. The self-conditioning continuation uses this to admit only
        # its zero-initialized projection, never an arbitrary partial checkpoint.
        missing, unexpected = raw_model.load_state_dict(ckpt[model_state_key], strict=False)
        ema_missing, ema_unexpected = ema.load_state_dict(ckpt["ema"], strict=False)
        if args.strict_resume:
            expected_missing = set(args.expected_missing_resume_keys)
            model_missing = set(missing)
            shadow_missing = set(ema_missing)
            if (
                model_missing != expected_missing
                or shadow_missing != expected_missing
                or unexpected
                or ema_unexpected
            ):
                raise RuntimeError(
                    "Strict resume state mismatch: "
                    f"expected_missing={sorted(expected_missing)} "
                    f"model_missing={sorted(model_missing)} "
                    f"ema_missing={sorted(shadow_missing)} "
                    f"model_unexpected={sorted(unexpected)} "
                    f"ema_unexpected={sorted(ema_unexpected)}"
                )
        if is_main and (missing or unexpected):
            print(
                f"  [resume] model missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )
            if missing:
                print(f"    missing: {missing[:5]}{'...' if len(missing) > 5 else ''}", flush=True)
            if unexpected:
                suffix = "..." if len(unexpected) > 5 else ""
                print(f"    unexpected: {unexpected[:5]}{suffix}", flush=True)
        if is_main and (ema_missing or ema_unexpected):
            print(
                f"  [resume] ema missing={len(ema_missing)} unexpected={len(ema_unexpected)}",
                flush=True,
            )
        if args.reset_optimizer:
            # Keep weights+EMA, fresh optimizer/scheduler with current args
            # (lr, warmup, total_steps).
            start_step = args.start_step
            if is_main:
                print(
                    f"Resumed weights from {args.resume} at ckpt step "
                    f"{ckpt['step']} ({model_state_key}) → fresh "
                    "optimizer/scheduler, "
                    f"start_step={start_step}",
                    flush=True,
                )
        else:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_step = ckpt["step"]
            if is_main:
                print(f"Resumed full state from {args.resume} at step {start_step}", flush=True)
        if args.reset_optimizer:
            # A weights-only restart owns a fresh sampler/RNG stream.
            micro_batches_consumed = 0
        else:
            data_state = ckpt.get("data_state") or {}
            validate_data_resume_state(
                data_state,
                ckpt.get("args") or {},
                world_size=world_size,
                batch_size=args.batch_size,
                grad_accum_steps=args.grad_accum_steps,
                batches_per_epoch=batches_per_epoch,
                dataset_size=len(dataset),
                sampler_type=type(sampler).__name__,
                seed=args.seed,
            )
            micro_batches_consumed = int(
                data_state.get(
                    "micro_batches_consumed",
                    start_step * max(1, int(args.grad_accum_steps)),
                )
            )
            rng_states = ckpt.get("rng_states")
            if rng_states and rank < len(rng_states):
                resume_rng_state = rng_states[rank]
            elif is_main and rng_states:
                print(
                    f"[resume] checkpoint has RNG state for {len(rng_states)} ranks, "
                    f"current world_size={world_size}; unmatched ranks use the seed.",
                    flush=True,
                )
        del ckpt

    # ── wandb ────────────────────────────────────────────────────────────────
    if is_main:
        init_wandb(args, out_dir, world_size, n_params, len(dataset), resume_run_id=resume_run_id)
    # W&B initialization may consume host RNG; restore immediately before the
    # first resumed training batch so the model path continues deterministically.
    if args.resume and not args.reset_optimizer:
        restore_rng_state(resume_rng_state)

    # ── train loop ───────────────────────────────────────────────────────────
    gpu_monitor = GPUMonitor(interval=60) if is_main else None
    if gpu_monitor:
        gpu_monitor.start()

    model.train()
    step = start_step
    metric_sums: dict[str, float] = {}
    metric_count = 0
    grad_accum = max(1, int(getattr(args, "grad_accum_steps", 1)))
    start_epoch, start_batch = divmod(micro_batches_consumed, batches_per_epoch)
    if is_main and micro_batches_consumed:
        print(
            f"Data resume: micro_batches={micro_batches_consumed} "
            f"epoch={start_epoch} batch={start_batch}/{batches_per_epoch}",
            flush=True,
        )
    loader_iter = iter(
        inf_loader(
            loader,
            sampler,
            start_epoch=start_epoch,
            start_batch=start_batch,
        )
    )
    last_checkpoint_step = -1
    preempted = False

    def checkpoint_at_step(current_step: int) -> None:
        nonlocal last_checkpoint_step
        local_rng_state = capture_rng_state()
        if is_dist:
            gathered_rng_states = [None] * world_size if is_main else None
            dist.gather_object(
                local_rng_state,
                gathered_rng_states,
                dst=0,
            )
        else:
            gathered_rng_states = [local_rng_state]
        if is_main:
            save_checkpoint(
                out_dir,
                current_step,
                model,
                ema,
                optimizer,
                scheduler,
                args,
                rng_states=gathered_rng_states,
                data_state={
                    "micro_batches_consumed": micro_batches_consumed,
                    "world_size": world_size,
                    "batch_size": args.batch_size,
                    "grad_accum_steps": grad_accum,
                    "batches_per_epoch": batches_per_epoch,
                    "dataset_size": len(dataset),
                    "sampler_type": type(sampler).__name__,
                    "seed": args.seed,
                },
            )
        if is_dist:
            dist.barrier()
        last_checkpoint_step = current_step

    try:
        while step < args.total_steps:
            step_started = time.perf_counter()
            loader_wait_s = 0.0
            optimizer.zero_grad(set_to_none=True)
            accum: dict[str, float] = {}
            oom = False
            for micro_idx in range(grad_accum):
                loader_wait_started = time.perf_counter()
                try:
                    batch = next(loader_iter)
                    micro_batches_consumed += 1
                except StopIteration:
                    loader_iter = iter(inf_loader(loader, sampler))
                    batch = next(loader_iter)
                    micro_batches_consumed += 1
                loader_wait_s += time.perf_counter() - loader_wait_started
                missing_batch = batch is None
                if is_dist:
                    # A single invalid local batch must make every rank skip
                    # before any rank enters a shape or gradient collective.
                    missing_batch = any_rank_true(missing_batch, device)
                if missing_batch:
                    oom = True
                    break
                if is_dist:
                    # TileLang kernels specialize on sequence length. Keep all
                    # ranks on the same shape so JIT compilation cannot strand
                    # faster ranks inside a DDP all-reduce.
                    global_L = distributed_max_int(batch.max_len, device)
                    batch = batch.pad_to_length(global_L)
                batch = batch.to(torch.device(device))

                is_last = micro_idx == grad_accum - 1
                # Skip DDP all-reduce on intermediate micro-steps; final micro
                # synchronizes the accumulated grads across ranks once.
                sync_ctx = model.no_sync() if (is_dist and not is_last) else nullcontext()
                try:
                    with sync_ctx:
                        loss, m = allatom_forward_and_loss(
                            model,
                            batch,
                            alpha_mode=args.alpha_mode,
                            use_rigid_align=args.use_rigid_align,
                            use_amp=True,
                            w_fm=args.w_fm,
                            w_lddt_atom=args.w_lddt_atom,
                            w_bond=args.w_bond,
                            w_angle=args.w_angle,
                            w_clash=args.w_clash,
                            lddt_cutoff_A=args.lddt_cutoff_A,
                            lddt_pair_chunk_size=args.lddt_pair_chunk_size,
                            clash_overlap_tolerance_A=args.clash_overlap_tolerance_A,
                            clash_margin_A=args.clash_margin_A,
                            clash_huber_delta_A=args.clash_huber_delta_A,
                            clash_soft_count_tau_A=args.clash_soft_count_tau_A,
                            clash_pair_chunk_size=args.clash_pair_chunk_size,
                            self_condition_prob=args.self_condition_prob,
                        )
                        nonfinite = not bool(torch.isfinite(loss).item())
                        if is_dist:
                            nonfinite = any_rank_true(nonfinite, device)
                        if nonfinite:
                            oom = True
                            break
                        (loss / grad_accum).backward()
                except torch.cuda.OutOfMemoryError as exc:
                    if is_dist:
                        raise RuntimeError(
                            f"rank {rank} CUDA OOM at step={step} "
                            f"micro_step={micro_idx} length={batch.max_len}; "
                            "failing the DDP run instead of desynchronizing ranks"
                        ) from exc
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
                    print(f"[step {step}] OOM/NaN — skipped", flush=True)
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
            skipped = (not (grad_norm < 1e4)) or (grad_norm != grad_norm)
            if not skipped:
                optimizer.step()
            scheduler.step()
            if not skipped:
                ema.update(model.module if is_dist else model)
            elif is_main:
                print(f"[step {step}] gnorm spike skipped (gnorm={grad_norm:.2e})", flush=True)

            timing_ms = torch.tensor(
                [
                    loader_wait_s * 1000.0,
                    (time.perf_counter() - step_started) * 1000.0,
                ],
                device=device,
                dtype=torch.float32,
            )
            if is_dist:
                # Global progress is limited by the slowest rank.  One packed
                # collective exposes that tail without adding separate
                # synchronization points.
                dist.all_reduce(timing_ms, op=dist.ReduceOp.MAX)
            accum["perf_data_wait_ms_max"] = float(timing_ms[0].item())
            accum["perf_train_step_ms_max"] = float(timing_ms[1].item())
            accum["grad_norm"] = grad_norm
            step += 1

            # Metric accumulation. Values stay rank-local here and are reduced
            # once per logging interval, not once per metric per step. For a
            # mean the two are identical — every rank runs the same number of
            # accumulated steps, because an OOM/NaN skip is already agreed
            # across ranks above — so this is the same number for one collective
            # instead of `len(accum)` of them, each of which also cost a scalar
            # host-to-device upload and a device sync to read back. The `perf_`
            # timings keep their own per-step MAX collective: "mean over steps
            # of the slowest rank" is the throughput diagnostic, and it is not
            # recoverable from a reduction taken once per interval.
            for k, v in accum.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v
            metric_count += 1

            # Logging
            if step % args.log_interval == 0:
                avgs = {k: v / metric_count for k, v in metric_sums.items()}
                if is_dist:
                    # One packed collective. Key order is sorted so every rank
                    # packs the same slots; the key sets cannot diverge because
                    # every rank took the same accumulated steps.
                    keys = sorted(k for k in avgs if not k.startswith("perf_"))
                    packed = torch.tensor(
                        [avgs[k] for k in keys], device=device, dtype=torch.float32
                    )
                    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                    packed /= world_size
                    avgs.update(zip(keys, packed.tolist()))
                if is_main:
                    log_metrics(
                        step,
                        args.total_steps,
                        avgs,
                        scheduler.get_last_lr()[0],
                        world_size,
                        args.batch_size,
                        args.copies_per_protein,
                        args.grad_accum_steps,
                    )
                # Reset on every rank, not just rank 0: the sums are rank-local
                # now, and a rank that never reset would average over the whole
                # run instead of the interval.
                metric_sums, metric_count = {}, 0


            # Checkpoint (rank 0 saves; other ranks wait on barrier to avoid DDP desync)
            if step % args.ckpt_interval == 0:
                checkpoint_at_step(step)

            preempt_requested = preempt_state["requested"]
            if is_dist:
                preempt_requested = any_rank_true(preempt_requested, device)
            if preempt_requested:
                if last_checkpoint_step != step:
                    checkpoint_at_step(step)
                if is_main:
                    (out_dir / ".requeue_requested").write_text(f"step={step}\n")
                    print(
                        f"Preemption checkpoint complete at step {step}.",
                        flush=True,
                    )
                if is_dist:
                    dist.barrier()
                preempted = True
                break

    finally:
        if gpu_monitor:
            gpu_monitor.stop()

    # ── final ────────────────────────────────────────────────────────────────
    if not preempted and last_checkpoint_step != step:
        checkpoint_at_step(step)
    if is_main:
        if wandb.run is not None:
            wandb.finish()
        status = "Paused for requeue" if preempted else "Done"
        print(f"\n{status}. Total steps: {step}")

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
