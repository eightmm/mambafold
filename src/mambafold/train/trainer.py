"""Model construction, LR scheduler, checkpoint I/O, and seeding."""

import math
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP


def seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        bimamba_share=cfg.get("bimamba_share", False),
        d_atom=cfg.get("d_atom", 128),
        n_atom_layers=cfg.get("n_atom_layers", 4),
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


def save_checkpoint(out_dir: Path, step: int, model, ema, optimizer, scheduler, args):
    import wandb

    raw_model = model.module if isinstance(model, DDP) else model
    path = out_dir / f"ckpt_{step:07d}.pt"
    torch.save({
        "step": step,
        "model": raw_model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": vars(args) if not isinstance(args, dict) else args,
        "wandb_run_id": wandb.run.id if wandb.run is not None else None,
    }, path)
    latest = out_dir / "ckpt_latest.pt"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(path.name)
    print(f"Saved: {path}", flush=True)


def load_from_checkpoint(ckpt_path: str | Path, device: str = "cpu", use_ema: bool = True):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    cfg = dict(a if isinstance(a, dict) else vars(a))
    cfg.setdefault("trunk_attn_layers", None)
    cfg.setdefault("bimamba_share", False)
    cfg.setdefault("pair_use_cueq", False)
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
        print(f"[resume] ema missing={len(ema_missing)} unexpected={len(ema_unexpected)}", flush=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt["step"])
