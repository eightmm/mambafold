"""Model construction, LR scheduler, checkpoint I/O, and seeding."""

import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP


def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict:
    """Capture one rank's host and CUDA RNG state for checkpoint resume."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng_state(state: dict | None) -> None:
    """Restore RNG state saved by :func:`capture_rng_state`."""
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"])


def build_model(cfg: dict, device: str = "cpu"):
    """Build the direct all-atom MambaFold model."""
    from mambafold.model.fold import MambaFoldAllAtom

    return MambaFoldAllAtom(
        d_res=cfg.get("d_res", 1024),
        n_trunk=cfg.get("n_trunk", 12),
        d_res_type=cfg.get("d_res_type", 32),
        d_res_pos=cfg.get("d_res_pos", 64),
        d_plm=cfg.get("d_plm", 1536),
        d_plm_proj=cfg.get("d_plm_proj", 256),
        d_ca_emb=cfg.get("d_ca_emb", 128),
        use_plm=cfg.get("use_plm", True),
        d_pair=cfg.get("d_pair", 192),
        n_pair_blocks=cfg.get("n_pair_blocks", 4),
        n_pair_heads=cfg.get("n_pair_heads", 4),
        pair_mult_c=cfg.get("pair_mult_c", 128),
        mimo_rank=cfg.get("mimo_rank", 4),
        d_state=cfg.get("d_state", 64),
        expand=cfg.get("expand", 2),
        headdim=cfg.get("headdim", 64),
        pair_use_cueq=cfg.get("pair_use_cueq", False),
        trunk_attn_layers=cfg.get("trunk_attn_layers", None),
        trunk_attn_every=cfg.get("trunk_attn_every", None),
        n_attn_heads=cfg.get("n_attn_heads", 16),
        trunk_time_film=cfg.get("trunk_time_film", False),
        trunk_adaln_zero=cfg.get("trunk_adaln_zero", False),
        self_conditioning=cfg.get("self_conditioning", False),
        bimamba_share=cfg.get("bimamba_share", False),
        d_atom=cfg.get("d_atom", 128),
        n_atom_layers=cfg.get("n_atom_layers", 4),
        use_pair_stack=cfg.get("use_pair_stack", True),
        pairfree_aux_heads=cfg.get("pairfree_aux_heads", False),
    ).to(torch.device(device))


def cosine_warmup_lr(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    decay_fraction: float = 0.5,
):
    decay_start = int(total_steps * (1.0 - decay_fraction))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if step < decay_start:
            return 1.0
        progress = (step - decay_start) / max(1, total_steps - decay_start)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def validate_data_resume_state(
    data_state: dict,
    checkpoint_args: dict,
    *,
    world_size: int,
    batch_size: int,
    grad_accum_steps: int,
    batches_per_epoch: int,
    dataset_size: int,
    sampler_type: str,
    seed: int,
) -> None:
    """Reject a full-state resume whose sampler contract has changed.

    Older checkpoints do not contain every field, so only recorded fields are
    checked. A weights-only restart owns a fresh data stream and should bypass
    this validation.
    """
    saved = dict(data_state or {})
    for key in ("batch_size", "seed"):
        if key not in saved and key in checkpoint_args:
            saved[key] = checkpoint_args[key]
    expected = {
        "world_size": int(world_size),
        "batch_size": int(batch_size),
        "grad_accum_steps": int(grad_accum_steps),
        "batches_per_epoch": int(batches_per_epoch),
        "dataset_size": int(dataset_size),
        "sampler_type": str(sampler_type),
        "seed": int(seed),
    }
    mismatches = [
        f"{key}: checkpoint={saved[key]!r} current={current!r}"
        for key, current in expected.items()
        if key in saved and saved[key] != current
    ]
    if mismatches:
        raise RuntimeError(
            "Data resume contract mismatch; use a matching configuration or "
            "--reset_optimizer for a fresh data stream: " + "; ".join(mismatches)
        )


def save_checkpoint(
    out_dir: Path,
    step: int,
    model,
    ema,
    optimizer,
    scheduler,
    args,
    *,
    rng_states: list[dict] | None = None,
    data_state: dict | None = None,
):
    import wandb

    raw_model = model.module if isinstance(model, DDP) else model
    path = out_dir / f"ckpt_{step:07d}.pt"
    tmp_path = out_dir / f".{path.name}.tmp"
    torch.save(
        {
            "checkpoint_version": 2,
            "step": step,
            "model": raw_model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args) if not isinstance(args, dict) else args,
            "wandb_run_id": wandb.run.id if wandb.run is not None else None,
            "rng_states": rng_states,
            "data_state": data_state or {},
        },
        tmp_path,
    )
    os.replace(tmp_path, path)
    latest = out_dir / "ckpt_latest.pt"
    latest_tmp = out_dir / ".ckpt_latest.pt.tmp"
    if latest_tmp.exists() or latest_tmp.is_symlink():
        latest_tmp.unlink()
    latest_tmp.symlink_to(path.name)
    os.replace(latest_tmp, latest)
    keep_last = max(
        1,
        int(
            args.get("keep_last_checkpoints", 3)
            if isinstance(args, dict)
            else getattr(args, "keep_last_checkpoints", 3)
        ),
    )
    milestone_values = (
        args.get("keep_checkpoint_steps", [])
        if isinstance(args, dict)
        else getattr(args, "keep_checkpoint_steps", [])
    )
    milestones = {int(value) for value in (milestone_values or [])}
    numbered = sorted(out_dir.glob("ckpt_[0-9][0-9][0-9][0-9][0-9][0-9][0-9].pt"))
    recent = set(numbered[-keep_last:])
    for old_path in numbered:
        try:
            old_step = int(old_path.stem.removeprefix("ckpt_"))
        except ValueError:
            continue
        if old_path not in recent and old_step not in milestones:
            old_path.unlink()
    print(f"Saved: {path}", flush=True)


def load_from_checkpoint(ckpt_path: str | Path, device: str = "cpu", use_ema: bool = True):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    cfg = dict(a if isinstance(a, dict) else vars(a))
    cfg.setdefault("trunk_attn_layers", None)
    cfg.setdefault("trunk_time_film", False)
    cfg.setdefault("trunk_adaln_zero", False)
    cfg.setdefault("self_conditioning", False)
    cfg.setdefault("bimamba_share", False)
    cfg.setdefault("pair_use_cueq", False)
    cfg.setdefault("use_pair_stack", True)
    cfg.setdefault("pairfree_aux_heads", False)
    model = build_model(cfg, device)
    key = "ema" if (use_ema and "ema" in ckpt) else "model"
    missing, unexpected = model.load_state_dict(ckpt[key], strict=False)
    if missing:
        print(f"  [load] freshly initialized: {missing}")
    if unexpected:
        print(f"  [load] dropped unexpected keys: {unexpected}")
    model.eval()
    step = ckpt.get("step", "?")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded checkpoint step={step} ({key}): {ckpt_path} [{n_params:.1f}M params]")
    return model


def load_checkpoint(path: Path, model, ema, optimizer, scheduler, device) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_model = model.module if isinstance(model, DDP) else model
    missing, unexpected = raw_model.load_state_dict(ckpt["model"], strict=False)
    ema_missing, ema_unexpected = ema.load_state_dict(ckpt["ema"], strict=False)
    if missing or unexpected:
        print(f"[resume] model missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if ema_missing or ema_unexpected:
        print(
            f"[resume] ema missing={len(ema_missing)} unexpected={len(ema_unexpected)}",
            flush=True,
        )
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt["step"])
