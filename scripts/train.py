#!/usr/bin/env python
"""MambaFold-clean training entry point (single- or multi-GPU DDP).

Single GPU:
    PYTHONPATH=src python scripts/train.py --config configs/run_a_mamba.yaml

Multi-GPU:
    PYTHONPATH=src torchrun --nproc_per_node=8 scripts/train.py \
        --config configs/run_a_mamba.yaml
"""

import hashlib
import json
import os
import signal
import sys
import time
from contextlib import nullcontext
from itertools import chain
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mambafold.data.loader import build_dataloaders, inf_loader
from mambafold.train.config import parse_args
from mambafold.train.distributed import (
    GPUMonitor,
    any_rank_true,
    distributed_batch_info,
    enable_cuda_perf_flags,
    resolve_dataloader_workers,
    setup_dist,
)
from mambafold.train.ema import EMA
from mambafold.train.engine import (
    allatom_forward_and_loss,
)
from mambafold.train.logging import init_wandb, log_metrics
from mambafold.train.optim import build_optimizer, load_optimizer_state, optimizer_summary
from mambafold.train.trainer import (
    build_model,
    capture_rng_state,
    freeze_training_setup,
    initialize_refiner_from_checkpoint,
    linear_warmup_lr,
    prewarm_sequence_kernels,
    restore_rng_state,
    sample_recycle_loops,
    sample_refinement_loops,
    save_checkpoint,
    seed_all,
    validate_data_resume_state,
)


def rank_stream_seed(seed: int, rank: int) -> int:
    """Seed for one rank's host and CUDA training draws; rank 0 keeps the run seed.

    Other ranks hash (seed, rank) instead of using seed + rank, which would
    replay the stream of a neighbouring run seed (20261023 + 1 is 20261024).
    Four bytes keep it a valid NumPy seed.
    """
    if type(seed) is not int or type(rank) is not int or rank < 0:
        raise ValueError("rank stream seeding requires an integer seed and nonnegative rank")
    if rank == 0:
        return seed
    digest = hashlib.sha256(f"mambafold-rank-stream:{seed}:{rank}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def seed_rank_streams(seed: int, rank: int) -> None:
    """Give each DDP rank its own training-draw streams without touching rank 0.

    Every rank is seeded identically for model construction. Without this the
    DataLoader worker seeds (drawn from the host RNG when the loader iterator
    starts) and the CUDA draws are identical on every rank, so all ranks share
    crop offsets, t, noise, rotations and depth grids. The batch sampler keeps
    its own seeded generator and stays shared, so rank slices never overlap.
    Rank 0 is not reseeded: a single-process run consumes the stream it always did.
    """
    seed = rank_stream_seed(seed, rank)
    if rank:
        seed_all(seed)


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
    if args.initialize_refiner_from:
        receipt = initialize_refiner_from_checkpoint(
            model, args.initialize_refiner_from, vars(args),
        )
        if is_main:
            (out_dir / "refiner_initialization.json").write_text(
                json.dumps(receipt, indent=2)
            )
            print(
                "[initialize] complete EMA backbone preserved; fresh additive refiner", flush=True,
            )

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

    raw_model = model.module if is_dist else model
    ema = EMA(raw_model, decay=args.ema_decay)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    if is_main:
        print(f"Model: {n_params:.2f}M params")
        print(
            "Architecture: "
            f"self_conditioning={getattr(raw_model, 'self_conditioning', False)} "
            f"self_condition_prob={args.self_condition_prob}",
            flush=True,
        )

    optimizer = build_optimizer(model, vars(args))
    if is_main:
        print(f"Optimizer: {args.optimizer} {optimizer_summary(optimizer)}", flush=True)
    scheduler = linear_warmup_lr(
        optimizer, args.warmup_steps, args.lr, args.min_lr,
        total_steps=args.total_steps, cooldown_steps=args.lr_cooldown_steps,
    )

    if args.prewarm_kernels and device.startswith("cuda"):
        # Every rank runs the same shapes, so this stays in lockstep and no rank
        # can be stranded in a collective while another compiles.
        t_warm = time.time()
        sampled_recycling = args.recycle_policy == "sampled16_tail2"
        if sampled_recycling:
            raw_model.refinement_training_loops = 16
        elif args.recycle_policy == "fixed4":
            raw_model.refinement_training_loops = 4
        elif args.latent_loop_sampling:
            raw_model.refinement_training_loops = 3
        try:
            warmed = prewarm_sequence_kernels(
                model,
                batch_size=args.batch_size * max(1, int(args.copies_per_protein)),
                max_length=args.max_length,
                length_bin=args.length_bin,
                device=device,
            )
        finally:
            raw_model.refinement_training_loops = None
        expected_shapes = (args.max_length + args.length_bin - 1) // args.length_bin
        if (sampled_recycling or args.latent_loop_sampling or args.recycle_policy == "fixed4") \
                and len(warmed) != expected_shapes:
            budget = 16 if sampled_recycling else (4 if args.recycle_policy == "fixed4" else 3)
            raise RuntimeError(f'K={budget} prewarm did not fit every training shape; reduce batch copies')
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
            load_optimizer_state(optimizer, ckpt["optimizer"])
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

    # Before the loader iterator draws worker seeds, and before a full resume
    # restores each rank's saved RNG state below, which takes precedence.
    seed_rank_streams(args.seed, rank)

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
    if step < args.total_steps:
        # Initialize lazy sampler/worker state before freezing setup roots.
        # Replay this batch through the ordinary accounting and DDP protocol.
        first_batch = next(loader_iter)
        loader_iter = chain((first_batch,), loader_iter)
        del first_batch
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
        with freeze_training_setup() as frozen_count:
            if is_main and frozen_count:
                print(f"[gc] froze {frozen_count} persistent setup objects", flush=True)
            while step < args.total_steps:
                step_started = time.perf_counter()
                loader_wait_s = 0.0
                optimizer.zero_grad(set_to_none=True)
                accum: dict[str, float] = {}
                if args.recycle_policy != "legacy":
                    raw_model.refinement_training_loops = (
                        sample_recycle_loops(step, args.seed)
                        if args.recycle_policy == "sampled16_tail2"
                        else (4 if args.recycle_policy == "fixed4" else 2)
                    )
                    accum['refinement_loops'] = float(raw_model.refinement_training_loops)
                elif args.latent_loop_sampling:
                    raw_model.refinement_training_loops = sample_refinement_loops(step, args.seed)
                    accum['refinement_loops'] = float(raw_model.refinement_training_loops)
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
                        missing_batch, global_L = distributed_batch_info(
                            None if missing_batch else batch.max_len, device,
                        )
                    if missing_batch:
                        oom = True
                        break
                    if is_dist:
                        # TileLang kernels specialize on sequence length. Keep all
                        # ranks on the same shape so JIT compilation cannot strand
                        # faster ranks inside a DDP all-reduce.
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

                # Missing batches/nonfinite losses already agree before backward.
                # A distributed CUDA OOM raises above and fails the whole run.
                if oom:
                    if is_main:
                        print(f"[step {step}] OOM/NaN — skipped", flush=True)
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip,
                ).item()
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
                data_wait_ms, train_step_ms = timing_ms.tolist()
                accum["perf_data_wait_ms_max"] = data_wait_ms
                accum["perf_train_step_ms_max"] = train_step_ms
                accum["grad_norm"] = grad_norm
                accum["update_applied"] = float(not skipped)
                for group in optimizer.param_groups:
                    kind = group.get("optimizer_kind", args.optimizer)
                    accum[f"lr_{kind}"] = group["lr"]
                    if "parameter_role" in group:
                        accum[f"lr_{kind}_{group['parameter_role']}"] = group["lr"]
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
                        with (out_dir / "metrics.jsonl").open("a") as stream:
                            stream.write(json.dumps({"step": step, **avgs}) + "\n")
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
