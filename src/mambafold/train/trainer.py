"""MambaFold model construction, LR scheduler, checkpoint I/O, and seed."""

import math
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP


def seed_all(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def _build_stage1_module(cfg: dict, device: str = "cpu"):
    """Bare Stage 1 model (no wrapping)."""
    from mambafold.model.fold import MambaFoldStage1
    return MambaFoldStage1(
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
        d_pair_head=cfg.get("d_pair_head", 48),
        pair_mult_c=cfg.get("pair_mult_c", 128),
        mimo_rank=cfg.get("mimo_rank", 4),
        d_state=cfg.get("d_state", 64),
        expand=cfg.get("expand", 2),
        headdim=cfg.get("headdim", 64),
        n_cycles=cfg.get("n_cycles_train", 1),
    ).to(torch.device(device))


def _build_stage2_module(cfg: dict, device: str = "cpu"):
    """Bare Stage 2 model (no wrapping)."""
    from mambafold.model.fold import MambaFoldStage2
    return MambaFoldStage2(
        d_atom=cfg.get("d_atom", 384),
        d_res_polish=cfg.get("d_res_polish", 512),
        n_atom_enc=cfg.get("n_atom_enc", 4),
        n_polish=cfg.get("n_polish", 4),
        n_atom_dec=cfg.get("n_atom_dec", 4),
        d_s1_res=cfg.get("d_res", 1024),
        d_ca_anchor=cfg.get("d_ca_anchor", 64),
        d_res_type_atom=cfg.get("d_res_type", 32),
        d_atom_slot=cfg.get("d_atom_slot", 32),
        d_fourier=cfg.get("d_fourier", 128),
        mimo_rank=cfg.get("mimo_rank", 4),
        d_state=cfg.get("d_state", 64),
        expand=cfg.get("expand", 2),
        headdim=cfg.get("headdim", 64),
    ).to(torch.device(device))


def _build_model_stage1(cfg: dict, device: str = "cpu"):
    """Stage 1 — CA-only flow matching (Linear Triangle pair stack)."""
    return _build_stage1_module(cfg, device)


def _load_stage1_into(stage1, ckpt_path: str | Path, device: str):
    """Load Stage 1 weights from a Phase-1 checkpoint into an existing
    `MambaFoldStage1` module. Prefers EMA when present.

    Tolerates missing/extra keys with `strict=False` and prints a one-line
    summary so structural drift is visible at launch time.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("ema", ckpt.get("model"))
    if state is None:
        raise RuntimeError(f"ckpt at {ckpt_path} has no 'model' or 'ema' state dict")
    missing, unexpected = stage1.load_state_dict(state, strict=False)
    print(f"[stage2 init] loaded Stage 1 from {ckpt_path} "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    if missing:
        print(f"  missing in ckpt → freshly initialised: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  unexpected in ckpt → dropped: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")


def _build_model_stage2(cfg: dict, device: str = "cpu"):
    """Stage 2 — TwoStageMambaFold with frozen Stage 1.

    When starting Phase 2 from scratch, `cfg.stage1_ckpt` must point at a
    Phase-1 ckpt whose weights are loaded into Stage 1; Stage 2 starts fresh.

    When called from `load_from_checkpoint` to reconstruct a saved Phase-2
    model for inference, set `cfg.skip_ckpt_load=True` to skip the Stage 1
    load — the caller will overwrite the full TwoStage state_dict.
    """
    from mambafold.model.fold import TwoStageMambaFold
    stage1 = _build_stage1_module(cfg, device)
    stage2 = _build_stage2_module(cfg, device)
    if not cfg.get("skip_ckpt_load", False):
        ckpt_path = cfg.get("stage1_ckpt")
        if not ckpt_path:
            raise ValueError(
                "stage=2 build requires `stage1_ckpt` (Phase-1 ckpt). "
                "Pass `skip_ckpt_load=True` for inference-time reconstruction."
            )
        _load_stage1_into(stage1, ckpt_path, device)
    return TwoStageMambaFold(stage1, stage2, freeze_stage1=True).to(torch.device(device))


def _load_two_stage_into(model, ckpt_path: str | Path, device: str):
    """Load both Stage 1 and Stage 2 weights from a TwoStageMambaFold ckpt
    (Phase-2 or Phase-3 output). EMA preferred when present.

    The state dict carries `stage1.*` and `stage2.*` prefixes, so a plain
    `load_state_dict(strict=False)` on the wrapper drops everything else.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("ema", ckpt.get("model"))
    if state is None:
        raise RuntimeError(f"ckpt at {ckpt_path} has no 'model' or 'ema' state dict")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[joint finetune init] loaded TwoStage weights from {ckpt_path} "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    if missing:
        print(f"  missing → freshly initialised: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  unexpected → dropped: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")


def _build_model_joint(cfg: dict, device: str = "cpu"):
    """joint finetune — TwoStageMambaFold with BOTH stages trainable (freeze_stage1=False).

    Loads from `cfg.joint_init_ckpt` (Phase-2 output recommended) into the
    full TwoStage wrapper. Falls back to `stage1_ckpt` (Phase-1 ckpt) +
    fresh Stage 2 if `joint_init_ckpt` is not set — useful for re-running
    Phase 2 + Phase 3 in one go.

    For inference reconstruction, set `cfg.skip_ckpt_load=True` and let the
    caller overwrite the state_dict.
    """
    from mambafold.model.fold import TwoStageMambaFold
    stage1 = _build_stage1_module(cfg, device)
    stage2 = _build_stage2_module(cfg, device)
    model = TwoStageMambaFold(stage1, stage2, freeze_stage1=False).to(torch.device(device))

    if cfg.get("skip_ckpt_load", False):
        return model

    joint_ckpt = cfg.get("joint_init_ckpt")
    if joint_ckpt:
        _load_two_stage_into(model, joint_ckpt, device)
        return model

    s1_ckpt = cfg.get("stage1_ckpt")
    if not s1_ckpt:
        raise ValueError(
            "joint finetune requires either `joint_init_ckpt` (Phase-2 output) or "
            "`stage1_ckpt` (Phase-1 output) to initialise Stage 1. "
            "Pass `skip_ckpt_load=True` for inference-time reconstruction."
        )
    _load_stage1_into(model.stage1, s1_ckpt, device)
    print("[joint finetune init] Stage 1 from stage1_ckpt; Stage 2 starts fresh.")
    return model


def build_model(cfg: dict, device: str = "cpu"):
    """Build the Stage 1, Stage 2, or joint model."""
    stage = cfg.get("stage", 1)
    if stage in ("1", "stage1"):
        stage = 1
    elif stage in ("2", "stage2"):
        stage = 2
    if stage == 1:
        return _build_model_stage1(cfg, device)
    if stage == 2:
        return _build_model_stage2(cfg, device)
    if stage == "joint":
        return _build_model_joint(cfg, device)
    raise ValueError(f"unknown stage: {stage!r} (expected 1, 2, or joint)")

def cosine_warmup_lr(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    decay_fraction: float = 0.5,
):
    """Linear warmup → constant max-lr → cosine decay to 0.

    The decay starts at `(1 - decay_fraction) · total_steps` and runs to the end.
    With `decay_fraction=0.5` (default): warmup → 50% steps at max lr →
    cosine-decay over the final 50%.
    """
    decay_start = int(total_steps * (1.0 - decay_fraction))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if step < decay_start:
            return 1.0
        progress = (step - decay_start) / max(1, total_steps - decay_start)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(out_dir: Path, step: int, model, ema,
                    optimizer, scheduler, args):
    """DDP-aware 체크포인트 저장 (rank 0에서만 호출).

    ckpt_latest.pt 심볼릭 링크를 최신 파일로 갱신함.
    """
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


def load_from_checkpoint(ckpt_path: str | Path, device: str = "cpu",
                         use_ema: bool = True):
    """Load a trained model from checkpoint (inference only, no optimizer).

    Args:
        ckpt_path: Path to checkpoint .pt file.
        device: Device to load model onto.
        use_ema: If True and EMA weights exist, use them.

    Returns:
        model in eval mode.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    cfg = a if isinstance(a, dict) else vars(a)
    # Wrapped checkpoints (stage=2 or joint) are reconstructed first, then
    # overwritten by the saved full state_dict below.
    stage = cfg.get("stage")
    if stage in (2, "2", "stage2", "joint"):
        cfg = {**cfg, "skip_ckpt_load": True}
    model = build_model(cfg, device)
    key = "ema" if (use_ema and "ema" in ckpt) else "model"
    # strict=False tolerates additive model changes (e.g. T1.1 pair_esm_proj
    # zero-init) when running inference on a config that enables them
    # against a Phase-1 ckpt that predates them.
    missing, unexpected = model.load_state_dict(ckpt[key], strict=False)
    if missing:
        print(f"  [infer-load] freshly-initialized (missing in ckpt): {missing}")
    if unexpected:
        print(f"  [infer-load] dropped (unexpected in ckpt): {unexpected}")
    model.eval()
    step = ckpt.get("step", "?")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded checkpoint step={step} ({key}): {ckpt_path}  [{n_params:.1f}M params]")
    return model


def load_checkpoint(path: Path, model, ema, optimizer, scheduler, device) -> int:
    """DDP-aware 체크포인트 로드. 재개할 step 번호를 반환.

    Model load uses strict=False so newly added layers (e.g. `plm_norm` in v3a)
    get fresh init when resuming a pre-existing ckpt. EMA load is left strict —
    if it fails on a structurally-changed model, the caller should pass
    --reset_optimizer (which currently bypasses load_checkpoint) or re-init EMA
    explicitly at the call site.
    """
    raw_model = model.module if isinstance(model, DDP) else model
    ckpt = torch.load(path, map_location=device, weights_only=False)
    missing, unexpected = raw_model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"  [resume] freshly-initialized (missing in ckpt): {missing}", flush=True)
    if unexpected:
        print(f"  [resume] dropped (unexpected in ckpt): {unexpected}", flush=True)
    ema.load_state_dict(ckpt["ema"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    step = ckpt["step"]
    print(f"Resumed from step {step}: {path}", flush=True)
    return step
