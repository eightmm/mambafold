"""W&B logging utilities for training."""

import time

import torch


def init_wandb(args, out_dir, world_size, n_params, n_train,
               resume_run_id: str | None = None):
    """Initialize wandb run (call on rank 0 only).

    Args:
        resume_run_id: If resuming from checkpoint, pass the saved wandb_run_id
            to continue logging to the same run.
    """
    import wandb

    if args.no_wandb:
        return
    copies = getattr(args, "copies_per_protein", 1)
    eff_batch = args.batch_size * world_size * copies
    wandb.init(
        project=args.wandb_project,
        id=resume_run_id,
        name=args.wandb_name or out_dir.name,
        tags=args.wandb_tags or [],
        config={
            **{k: v for k, v in vars(args).items()
               if not k.startswith("wandb") and k != "no_wandb"},
            "world_size": world_size,
            "effective_batch": eff_batch,
        },
        mode="offline" if args.wandb_offline else "online",
        resume="must" if resume_run_id else "allow",
    )
    wandb.config.update({"n_params_M": round(n_params, 2), "n_train": n_train})


_last_log_time: float | None = None
_last_log_step: int | None = None


def log_metrics(step, total_steps, avgs, lr, world_size, batch_size, copies):
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
            samples_per_step = batch_size * world_size * copies
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

    print(
        f"  step {step:>7d}/{total_steps} ({progress:.1f}%) | "
        f"loss={avgs['loss']:.4f} | eqm={avgs['eqm']:.4f} | "
        f"lddt={avgs['lddt']:.4f} | γ={avgs['gamma_mean']:.3f} | "
        f"gnorm={avgs['grad_norm']:.2f} | lr={lr:.2e}{vram}{throughput}",
        flush=True,
    )
    if wandb.run is not None:
        log_d = {
            "train/loss":        avgs["loss"],
            "train/loss_eqm":    avgs["eqm"],
            "train/loss_lddt":   avgs["lddt"],
            "train/gamma_mean":  avgs["gamma_mean"],
            "train/grad_norm":   avgs["grad_norm"],
            "train/alpha":       avgs["alpha"],
            "train/lr":          lr,
            "train/progress":    progress,
        }
        if step_time_ms > 0:
            log_d["perf/step_time_ms"] = step_time_ms
            log_d["perf/samples_per_sec"] = samples_per_sec
        if torch.cuda.is_available():
            log_d["gpu/vram_alloc_gb"] = alloc
            log_d["gpu/vram_reserved_gb"] = reserv
        wandb.log(log_d, step=step)


def log_val_metrics(step, val_avgs):
    """Log validation metrics to stdout and wandb."""
    import wandb

    print(
        f"  [val] step={step} | "
        f"eqm={val_avgs.get('eqm', 0):.4f} | "
        f"lddt={val_avgs.get('lddt', 0):.4f} | "
        f"grad_rms={val_avgs.get('grad_rms', 0):.4f}",
        flush=True,
    )
    if wandb.run is not None:
        wandb.log({f"val/{k}": v for k, v in val_avgs.items()}, step=step)
