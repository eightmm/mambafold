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

    # Independent step axes so val logs (every 5K) don't collide with the
    # explicit-step commits done by train logs (every 50). Without this,
    # wandb silently drops val/* when the internal step has advanced past
    # the explicit step=N passed by log_val_metrics.
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("val/*", step_metric="val/step")
    wandb.define_metric("gpu/*", step_metric="train/step")
    wandb.define_metric("perf/*", step_metric="train/step")


_last_log_time: float | None = None
_last_log_step: int | None = None


def _first_metric(metrics: dict[str, float], *names: str) -> float:
    for name in names:
        if name in metrics:
            return metrics[name]
    return 0.0


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

    main_v = _first_metric(avgs, "fm_atom")
    lddt_v = _first_metric(avgs, "lddt_atom", "lddt_ca")
    bond_v = _first_metric(avgs, "bond")
    clash_v = _first_metric(avgs, "clash", "ca_clash")
    distogram_v = _first_metric(avgs, "distogram")
    extra = ""
    if bond_v or clash_v:
        extra = f" | bond={bond_v:.4f} | clash={clash_v:.4f}"
    if distogram_v:
        extra += f" | dist={distogram_v:.4f}"
    chir_v = avgs.get("chirality", 0.0) + avgs.get("chirality_atom", 0.0)
    conf_v = avgs.get("conf", 0.0)
    if chir_v:
        extra += f" | chir={chir_v:.4f}"
    if conf_v:
        extra += f" | conf={conf_v:.4f}"
    ost_rate = avgs.get("ost_hard_per_1k", 0.0)
    if ost_rate or avgs.get("ost_clash", 0.0):
        extra += f" | ost={ost_rate:.2f}/1k | geo={avgs.get('ost_clash', 0.0):.4f}"
    print(
        f"  step {step:>7d}/{total_steps} ({progress:.1f}%) | "
        f"loss={avgs['loss']:.4f} | main={main_v:.4f} | "
        f"lddt={lddt_v:.4f}{extra} | t={avgs['t_mean']:.3f} | "
        f"gnorm={avgs['grad_norm']:.2f} | lr={lr:.2e}{vram}"
        f"{throughput}{data_timing}",
        flush=True,
    )
    if wandb.run is not None:
        log_d = {
            "train/step": step,
            "train/loss": avgs["loss"],
            "train/loss_main": main_v,
            "train/loss_lddt": lddt_v,
            "train/loss_bond": bond_v,
            "train/loss_clash": clash_v,
            "train/loss_distogram": distogram_v,
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
        # Forward every remaining scalar metric so auxiliary losses are logged.
        # NOTE: lddt_ca / ca_clash are intentionally NOT curated here — the
        # explicit keys above collapse them into lddt/clash via _first_metric,
        # so the catch-all below is the only place they reach W&B.
        _curated = {
            "loss",
            "t_mean",
            "grad_norm",
            "alpha",
            "target_L",
            "fm_atom",
            "lddt_atom",
            "bond",
            "clash",
            "distogram",
        }
        for k, v in avgs.items():
            if k not in _curated and not k.startswith("perf_") and isinstance(v, (int, float)):
                log_d[f"train/{k}"] = v
        wandb.log(log_d)  # step_metric="train/step" drives the x-axis


def log_val_metrics(step, val_avgs):
    """Log validation metrics to stdout and wandb."""
    import wandb

    main_v = val_avgs.get("fm_atom", val_avgs.get("main", 0.0))
    lddt_v = val_avgs.get("lddt_atom", val_avgs.get("lddt_ca", val_avgs.get("lddt", 0.0)))
    print(
        f"  [val] step={step} | "
        f"main={main_v:.4f} | "
        f"lddt={lddt_v:.4f} | "
        f"ca_fm={val_avgs.get('ca_fm', 0):.4f} | "
        f"lddt_ca={val_avgs.get('lddt_ca', 0):.4f} | "
        f"bond={val_avgs.get('bond', 0):.4f} | "
        f"clash={val_avgs.get('clash', 0):.4f} | "
        f"ost={val_avgs.get('ost_hard_per_1k', 0):.2f}/1k | "
        f"v_rms={val_avgs.get('v_rms', 0):.4f}",
        flush=True,
    )
    if wandb.run is not None:
        log_d = {f"val/{k}": v for k, v in val_avgs.items()}
        log_d["val/step"] = step
        wandb.log(log_d)  # step_metric="val/step" drives the val x-axis
