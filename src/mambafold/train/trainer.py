"""MambaFold EqM — model construction, LR scheduler, and checkpoint I/O."""

import math
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from mambafold.model.mambafold import MambaFoldEqM


def build_model(cfg: dict, device: str = "cpu") -> MambaFoldEqM:
    """cfg dict(또는 vars(args))로 MambaFoldEqM을 생성해 device로 이동."""
    return MambaFoldEqM(
        d_atom=cfg.get("d_atom", 256),
        d_res=cfg.get("d_res", 256),
        d_plm=cfg.get("d_plm", 1024),
        n_atom_enc=cfg.get("n_atom_enc", 2),
        n_trunk=cfg.get("n_trunk", 6),
        n_atom_dec=cfg.get("n_atom_dec", 2),
        use_plm=cfg.get("use_plm", False),
        plm_mode=cfg.get("plm_mode", "blend"),
        d_res_pos=cfg.get("d_res_pos", 64),
        d_atom_slot=cfg.get("d_atom_slot", 32),
        atom_d_state=cfg.get("d_state", 64),
        atom_mimo_rank=cfg.get("mimo_rank", 4),
        atom_headdim=cfg.get("headdim", 64),
        d_state=cfg.get("d_state", 64),
        mimo_rank=cfg.get("mimo_rank", 4),
        headdim=cfg.get("headdim", 64),
    ).to(torch.device(device))


def cosine_warmup_lr(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup → cosine decay LR 스케줄러."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(out_dir: Path, step: int, model, ema,
                    optimizer, scheduler, args):
    """DDP-aware 체크포인트 저장 (rank 0에서만 호출).

    ckpt_latest.pt 심볼릭 링크를 최신 파일로 갱신함.
    """
    raw_model = model.module if isinstance(model, DDP) else model
    path = out_dir / f"ckpt_{step:07d}.pt"
    torch.save({
        "step": step,
        "model": raw_model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": vars(args) if not isinstance(args, dict) else args,
    }, path)
    latest = out_dir / "ckpt_latest.pt"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(path.name)
    print(f"Saved: {path}", flush=True)


def load_checkpoint(path: Path, model, ema, optimizer, scheduler, device) -> int:
    """DDP-aware 체크포인트 로드. 재개할 step 번호를 반환."""
    raw_model = model.module if isinstance(model, DDP) else model
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    step = ckpt["step"]
    print(f"Resumed from step {step}: {path}", flush=True)
    return step
