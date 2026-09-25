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
        d_res=cfg.get("d_res", 256),
        n_trunk=cfg.get("n_trunk", 6),
        d_res_type=cfg.get("d_res_type", 32),
        d_res_pos=cfg.get("d_res_pos", 64),
        d_plm=cfg.get("d_plm", 1536),
        d_plm_proj=cfg.get("d_plm_proj", 256),
        d_temb=cfg.get("d_temb", 128),
        d_ca_emb=cfg.get("d_ca_emb", 128),
        use_plm=cfg.get("use_plm", False),
        mimo_rank=cfg.get("mimo_rank", 4),
        d_state=cfg.get("d_state", 64),
        expand=cfg.get("expand", 2),
        headdim=cfg.get("headdim", 64),
        self_conditioning=cfg.get("self_conditioning", False),
        bimamba_share=cfg.get("bimamba_share", False),
        d_atom=cfg.get("d_atom", 128),
        n_atom_layers=cfg.get("n_atom_layers", 4),
        atom_mixer=cfg.get("atom_mixer", "mamba"),
        atom_d_state=cfg.get("atom_d_state", 64),
        atom_mimo_rank=cfg.get("atom_mimo_rank", 2),
        n_atom_cross_layers=cfg.get("n_atom_cross_layers", 1),
        n_backbone_streams=cfg.get("n_backbone_streams", 5),
    ).to(torch.device(device))


def linear_warmup_lr(
    optimizer,
    warmup_steps: int,
    max_lr: float,
    min_lr: float = 1e-6,
    total_steps: int = 0,
    cooldown_steps: int = 0,
):
    """SimpleFold's `LinearWarmup`, plus an optional cosine cooldown.

    The reference schedule ramps min_lr → max_lr and then holds max_lr for the
    whole run, with no decay; `cooldown_steps=0` reproduces that exactly. A
    declared deviation is available because the trunk is not a transformer:
    attention tolerates a flat rate to the last step, while a selective SSM
    carries learned discretisation parameters that keep moving at 1e-4, and
    with EMA 0.999 the reported model is an average over only the final ~1000
    steps — i.e. an average taken at the noise floor. The cooldown costs
    nothing if the flat rate was fine and rescues the ending if it was not.

    There is no decay phase. The reference schedule
    (`utils/lr_scheduler.py::LinearWarmup`) interpolates linearly from `min_lr`
    to `max_lr` across `warmup_steps` and returns `max_lr` for every step after,
    for the whole run. A cosine tail would be a different training recipe, and
    this project changes the trunk and nothing else.
    """
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if not (0.0 < min_lr <= max_lr):
        raise ValueError("require 0 < min_lr <= max_lr")

    # LambdaLR multiplies the optimizer's own lr, which is max_lr, so express the
    # reference schedule as a fraction of it.
    floor = min_lr / max_lr

    if cooldown_steps < 0 or cooldown_steps > max(0, total_steps - warmup_steps):
        raise ValueError("cooldown_steps must fit between warmup_steps and total_steps")
    decay_start = total_steps - cooldown_steps if cooldown_steps else None

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return floor + (1.0 - floor) * step / max(1, warmup_steps)
        if decay_start is None or step < decay_start:
            return 1.0
        progress = min(1.0, (step - decay_start) / max(1, cooldown_steps))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def prewarm_sequence_kernels(model, batch_size: int, max_length: int, length_bin: int,
                             device: str, min_length: int = 64) -> list[int]:
    """Compile every sequence shape training can produce, before step 1.

    The Mamba-3 TileLang kernels specialize on sequence length and are compiled
    on first use, tens of seconds each. The collator rounds every batch up to a
    multiple of `length_bin`, so the set of shapes is small and known ahead of
    time — 16 of them at bin 64 and max_length 1024. Paying for them here turns
    a scatter of multi-second stalls across the first few hundred steps, each
    one holding every DDP rank inside a collective, into one bounded cost at
    startup that also shows up honestly in the log.

    Runs a real backward, not just a forward: the SSD kernels compile separate
    forward and backward programs, so a `no_grad` pass would leave every
    backward kernel to compile inside the first training step it is needed —
    exactly the stall this exists to remove. The loss is a sum over the output,
    which touches every parameter without needing targets. Returns the lengths
    it warmed.
    """
    import torch

    from mambafold.data.constants import MAX_ATOMS_PER_RES

    if length_bin <= 0:
        raise ValueError("prewarm requires a positive length_bin; see the config note")
    lengths = list(range(length_bin, max_length + 1, length_bin))
    if not lengths:
        return []
    torch_device = torch.device(device)
    d_plm = getattr(model, "d_plm", None) or getattr(getattr(model, "module", None), "d_plm", 2560)
    was_training = model.training
    model.train()
    warmed = []
    for length in lengths:
        batch = _synthetic_batch(batch_size, length, MAX_ATOMS_PER_RES, d_plm, torch_device)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch)
            out["v_atom"].float().sum().backward()
            model.zero_grad(set_to_none=True)
            warmed.append(length)
        except torch.cuda.OutOfMemoryError:
            # A shape training itself could not run either. Stop rather than
            # pretend the rest are warm.
            print(f"[prewarm] out of memory at L={length}; stopping", flush=True)
            break
        finally:
            del batch
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    if not was_training:
        model.eval()
    return warmed


def _synthetic_batch(batch_size: int, length: int, n_atoms: int, d_plm: int, device):
    """A dense batch of the right shape. Only shapes matter here, not values."""
    import torch

    from mambafold.data.types import ProteinBatch

    res = (batch_size, length)
    atom = (batch_size, length, n_atoms)
    ones_res = torch.ones(res, dtype=torch.bool, device=device)
    zeros_res = torch.zeros(res, dtype=torch.long, device=device)
    coords = torch.zeros(batch_size, length, n_atoms, 3, device=device)
    return ProteinBatch(
        res_type=zeros_res,
        res_seq_nums=torch.arange(length, device=device).expand(batch_size, length).contiguous(),
        atom_type=torch.zeros(atom, dtype=torch.long, device=device),
        pair_type=torch.zeros(atom, dtype=torch.long, device=device),
        res_mask=ones_res,
        atom_mask=torch.ones(atom, dtype=torch.bool, device=device),
        valid_mask=torch.ones(atom, dtype=torch.bool, device=device),
        ca_mask=ones_res,
        chain_id=zeros_res,
        entity_id=zeros_res,
        sym_id=zeros_res,
        is_nterm=torch.zeros(res, dtype=torch.bool, device=device),
        is_cterm=torch.zeros(res, dtype=torch.bool, device=device),
        x_clean=coords,
        x_t=coords.clone(),
        eps=coords.clone(),
        t=torch.full((batch_size, 1, 1, 1), 0.5, device=device),
        esm=torch.zeros(1, length, d_plm, device=device),
    )


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
    try:
        import wandb
    except ImportError:
        wandb_run_id = None
    else:
        active_run = getattr(wandb, "run", None)
        wandb_run_id = getattr(active_run, "id", None)

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
            "wandb_run_id": wandb_run_id,
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
