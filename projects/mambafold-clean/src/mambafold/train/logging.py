"""W&B logging utilities for training."""

import time

import torch


def init_wandb(args, out_dir, world_size, n_params, n_train, resume_run_id: str | None = None):
    """Initialize wandb run (call on rank 0 only).

    Args:
        resume_run_id: If resuming from checkpoint, pass the saved wandb_run_id
            to continue logging to the same run.
    """
    import wandb

    if args.no_wandb:
        return
    copies = getattr(args, "copies_per_protein", 1)
    grad_accum = getattr(args, "grad_accum_steps", 1)
    eff_batch = args.batch_size * world_size * copies * grad_accum
    wandb.init(
        project=args.wandb_project,
        id=resume_run_id,
        name=args.wandb_name or out_dir.name,
        tags=args.wandb_tags or [],
        config={
            **{
                k: v for k, v in vars(args).items() if not k.startswith("wandb") and k != "no_wandb"
            },
            "world_size": world_size,
            "effective_batch": eff_batch,
        },
        mode="offline" if args.wandb_offline else "online",
        resume="must" if resume_run_id else "allow",
    )
    wandb.config.update({"n_params_M": round(n_params, 2), "n_train": n_train})

    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("gpu/*", step_metric="train/step")
    wandb.define_metric("perf/*", step_metric="train/step")


_last_log_time: float | None = None
_last_log_step: int | None = None


def log_metrics(step, total_steps, avgs, lr, world_size, batch_size, copies, grad_accum_steps=1):
    """Log training metrics to stdout and wandb."""
    import wandb

    global _last_log_time, _last_log_step

    now = time.time()

    # Throughput
    step_time_ms = 0.0
    samples_per_sec = 0.0
    if _last_log_time is not None and _last_log_step is not None:
        elapsed = now - _last_log_time
        steps_done = step - _last_log_step
        if elapsed > 0 and steps_done > 0:
            step_time_ms = elapsed / steps_done * 1000
            samples_per_step = batch_size * world_size * copies * grad_accum_steps
            samples_per_sec = samples_per_step * steps_done / elapsed
    _last_log_time = now
    _last_log_step = step

    # VRAM
    alloc = reserv = 0.0
    vram = ""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserv = torch.cuda.memory_reserved() / 1024**3
        vram = f" | vram={alloc:.2f}/{reserv:.2f}GB"

    progress = step / total_steps * 100
    throughput = f" | {samples_per_sec:.0f} samp/s" if samples_per_sec > 0 else ""
    train_step_ms = avgs.get("perf_train_step_ms_max", 0.0)
    data_wait_ms = avgs.get("perf_data_wait_ms_max", 0.0)
    data_wait_pct = 100.0 * data_wait_ms / train_step_ms if train_step_ms > 0 else 0.0
    data_timing = (
        f" | data_wait={data_wait_ms:.0f}ms ({data_wait_pct:.1f}%)" if train_step_ms > 0 else ""
    )
    geometry = ""
    if any(avgs.get(name, 0.0) for name in ("bond", "angle", "clash")):
        geometry = (
            f" | bond={avgs.get('bond_mae_A', 0.0):.3f}A"
            f" angle={avgs.get('angle_mae_deg', 0.0):.2f}deg"
            f" clash={avgs.get('clashes_per_1000_atoms', 0.0):.1f}/1k"
            f" soft={avgs.get('soft_clashes_per_1000_atoms', 0.0):.1f}/1k"
            f" overlap={avgs.get('mean_clash_overlap_A', 0.0):.2f}A"
        )

    fm = avgs["fm_atom"]
    lddt = avgs["lddt_atom"]
    print(
        f"  step {step:>7d}/{total_steps} ({progress:.1f}%) | "
        f"loss={avgs['loss']:.4f} | fm={fm:.4f} | "
        f"lddt={lddt:.4f} | t={avgs['t_mean']:.3f} | "
        f"gnorm={avgs['grad_norm']:.2f} | lr={lr:.2e}{vram}"
        f"{throughput}{data_timing}{geometry}",
        flush=True,
    )
    if wandb.run is not None:
        log_d = {
            "train/step": step,
            "train/loss": avgs["loss"],
            "train/fm_atom": fm,
            "train/lddt_atom": lddt,
            "train/t_mean": avgs["t_mean"],
            "train/grad_norm": avgs["grad_norm"],
            "train/alpha": avgs["alpha"],
            "train/lr": lr,
            "train/progress": progress,
        }
        if step_time_ms > 0:
            log_d["perf/step_time_ms"] = step_time_ms
            log_d["perf/samples_per_sec"] = samples_per_sec
        if train_step_ms > 0:
            log_d["perf/train_step_ms_max"] = train_step_ms
            log_d["perf/data_wait_ms_max"] = data_wait_ms
            log_d["perf/data_wait_fraction"] = data_wait_pct / 100.0
        if torch.cuda.is_available():
            log_d["gpu/vram_alloc_gb"] = alloc
            log_d["gpu/vram_reserved_gb"] = reserv
        # Forward the remaining scalar diagnostics from the core objective.
        _curated = {
            "loss",
            "t_mean",
            "grad_norm",
            "alpha",
            "fm_atom",
            "lddt_atom",
        }
        for k, v in avgs.items():
            if k not in _curated and not k.startswith("perf_") and isinstance(v, (int, float)):
                log_d[f"train/{k}"] = v
        wandb.log(log_d)  # step_metric="train/step" drives the x-axis
