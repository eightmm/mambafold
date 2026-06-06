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

    # Independent step axes so val logs (every 5K) don't collide with the
    # explicit-step commits done by train logs (every 50). Without this,
    # wandb silently drops val/* when the internal step has advanced past
    # the explicit step=N passed by log_val_metrics.
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("val/*",   step_metric="val/step")
    wandb.define_metric("gpu/*",   step_metric="train/step")
    wandb.define_metric("perf/*",  step_metric="train/step")


_last_log_time: float | None = None
_last_log_step: int | None = None


def _first_metric(metrics: dict[str, float], *names: str) -> float:
    for name in names:
        if name in metrics:
            return metrics[name]
    return 0.0


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

    # Metric names differ across Stage 1, Stage 2, and joint runs.
    main_v = _first_metric(avgs, "main", "fm", "fm_atom", "s1_fm", "s2_fm_atom")
    lddt_v = _first_metric(avgs, "lddt", "lddt_full", "s1_lddt", "s2_lddt_full")
    bond_v = _first_metric(avgs, "bond", "bond_caca", "s1_bond_caca", "s2_bond")
    clash_v = _first_metric(avgs, "clash", "s2_clash")
    distogram_v = _first_metric(avgs, "distogram", "s1_distogram")
    extra = ""
    if bond_v or clash_v:
        extra = f" | bond={bond_v:.4f} | clash={clash_v:.4f}"
    if distogram_v:
        extra += f" | dist={distogram_v:.4f}"
    print(
        f"  step {step:>7d}/{total_steps} ({progress:.1f}%) | "
        f"loss={avgs['loss']:.4f} | main={main_v:.4f} | "
        f"lddt={lddt_v:.4f}{extra} | t={avgs['t_mean']:.3f} | "
        f"gnorm={avgs['grad_norm']:.2f} | lr={lr:.2e}{vram}{throughput}",
        flush=True,
    )
    if wandb.run is not None:
        log_d = {
            "train/step":              step,
            "train/loss":              avgs["loss"],
            "train/loss_main":         main_v,
            "train/loss_lddt":         lddt_v,
            "train/loss_bond":         bond_v,
            "train/loss_clash":        clash_v,
            "train/loss_distogram":    distogram_v,
            "train/t_mean":            avgs["t_mean"],
            "train/grad_norm":         avgs["grad_norm"],
            "train/alpha":             avgs["alpha"],
            "train/lr":                lr,
            "train/progress":          progress,
        }
        if step_time_ms > 0:
            log_d["perf/step_time_ms"] = step_time_ms
            log_d["perf/samples_per_sec"] = samples_per_sec
        if torch.cuda.is_available():
            log_d["gpu/vram_alloc_gb"] = alloc
            log_d["gpu/vram_reserved_gb"] = reserv
        # Forward every remaining scalar metric (contact, drmsd, pcb, conf,
        # ca_angle, ca_self_clash, s1_conf_mean, joint s1_/s2_ keys, ...) so
        # auxiliary losses aren't silently dropped from W&B.
        _curated = {"loss", "t_mean", "grad_norm", "alpha", "target_L",
                    "main", "fm", "fm_atom", "s1_fm", "s2_fm_atom",
                    "lddt", "lddt_full", "s1_lddt", "s2_lddt_full",
                    "bond", "bond_caca", "s1_bond_caca", "s2_bond",
                    "clash", "s2_clash", "distogram", "s1_distogram"}
        for k, v in avgs.items():
            if k not in _curated and isinstance(v, (int, float)):
                log_d[f"train/{k}"] = v
        wandb.log(log_d)   # step_metric="train/step" drives the x-axis


def log_val_metrics(step, val_avgs):
    """Log validation metrics to stdout and wandb."""
    import wandb

    main_v = val_avgs.get("main", 0.0)
    print(
        f"  [val] step={step} | "
        f"main={main_v:.4f} | "
        f"lddt={val_avgs.get('lddt', 0):.4f} | "
        f"bond={val_avgs.get('bond', 0):.4f} | "
        f"clash={val_avgs.get('clash', 0):.4f} | "
        f"if_l={val_avgs.get('iface_lddt', 0):.4f} | "
        f"if_c={val_avgs.get('iface_clash', 0):.4f} | "
        f"v_rms={val_avgs.get('v_rms', 0):.4f}",
        flush=True,
    )
    if wandb.run is not None:
        log_d = {f"val/{k}": v for k, v in val_avgs.items()}
        log_d["val/step"] = step
        wandb.log(log_d)   # step_metric="val/step" drives the val x-axis
