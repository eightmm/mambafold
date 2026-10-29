"""Model construction, LR scheduler, checkpoint I/O, and seeding."""

import gc
import math
import os
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP


@contextmanager
def freeze_training_setup():
    """Keep persistent setup objects out of full GC scans during training.

    New batch/graph objects still participate in normal cyclic collection.
    The standalone training entrypoint owns the process's permanent generation;
    all frozen objects are thawed on exit, including interpreter startup roots.
    """
    gc.collect()
    gc.freeze()
    try:
        yield gc.get_freeze_count()
    finally:
        gc.unfreeze()


def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_refinement_loops(step: int, seed: int) -> int:
    """Rank-independent K=1/2/3 draw (1:2:1) without consuming training RNG.

    Identical optimizer-step counters and run seeds give every rank and all
    accumulated microbatches the same budget, including after checkpoint resume.
    """
    draw = random.Random(f"refinement-depth:{seed}:{step}").randrange(4)
    return (1, 2, 2, 3)[draw]


def sample_recycle_loops(step: int, seed: int) -> int:
    """Sample K=1..16 with probability proportional to 1/K, without shared RNG.

    Sampling once per optimizer step gives every rank and accumulated
    microbatch the same budget, including after checkpoint resume.
    """
    if type(step) is not int or step < 0 or type(seed) is not int:
        raise ValueError("recycle sampling requires a nonnegative integer step and integer seed")
    depths = tuple(range(1, 17))
    return random.Random(f"atom-recycle-depth:{seed}:{step}").choices(
        depths, weights=tuple(1 / depth for depth in depths), k=1,
    )[0]


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
        latent_compression=cfg.get("latent_compression", 0),
        d_latent=cfg.get("d_latent", 256),
        n_latent_layers=cfg.get("n_latent_layers", 2),
        latent_loops=cfg.get("latent_loops", 2),
        latent_loop_mode=cfg.get("latent_loop_mode", "conditioned"),
        latent_position=cfg.get("latent_position", "post"),
        latent_supervision=cfg.get("latent_supervision", "final"),
        residue_refiner=cfg.get("residue_refiner", False),
        atom_recurrence=cfg.get("atom_recurrence", False),
        atom_recycling=cfg.get("atom_recycling", "none"),
        recycle_policy=cfg.get("recycle_policy", "legacy"),
        shared_trunk_layers=cfg.get("shared_trunk_layers", 0),
        latent_hierarchy=cfg.get("latent_hierarchy", False),
        latent_coarse_width=cfg.get("latent_coarse_width", 0),
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



def initialize_residue_refiner(model, source_state: dict, source_config: dict,
                               target_config: dict) -> dict:
    """Copy the complete learned backbone into a fresh additive refiner.

    The reference function is the source folding backbone with its old latent
    branch bypassed. That replaced branch is explicitly excluded, not reshaped.
    This is a weights-only initialization and never restores optimizer state.
    """
    from mambafold.model.fold.latent_mamba import ResidueLatentRefiner

    if not isinstance(model.latent_residual, ResidueLatentRefiner):
        raise ValueError("target must enable the residue refiner")
    shared_config = (
        "d_res", "n_trunk", "d_temb", "d_ca_emb", "d_res_type", "d_res_pos",
        "d_plm", "d_plm_proj", "use_plm", "d_state", "mimo_rank", "expand", "headdim",
        "bimamba_share", "d_atom", "n_atom_layers", "atom_mixer", "atom_d_state",
        "atom_mimo_rank", "n_atom_cross_layers", "n_backbone_streams", "self_conditioning",
    )
    mismatches = [key for key in shared_config
                  if key in source_config and key in target_config
                  and source_config[key] != target_config[key]]
    if mismatches:
        raise ValueError("backbone configuration differs: " + ", ".join(mismatches))
    prefix = "latent_residual."
    own = model.state_dict()
    source_keys = {key for key in source_state if not key.startswith(prefix)}
    target_keys = {key for key in own if not key.startswith(prefix)}
    if source_keys != target_keys:
        raise ValueError("backbone tensors differ; truncation or partial migration is forbidden")
    if any(own[key].shape != source_state[key].shape for key in target_keys):
        raise ValueError("backbone tensor shapes differ")
    if any(layer.stable for layer in model.residue_trunk.layers):
        raise ValueError("pretrained trunk normalization must stay unchanged")
    count = model.residue_trunk.n_layers
    if model.plm_inject_at != tuple(sorted({count // 3, 2 * count // 3})):
        raise ValueError("pretrained PLM reinjection schedule differs")
    if torch.count_nonzero(model.latent_residual.output_proj.weight).item():
        raise ValueError("refiner readout must be fresh and zero-initialized")
    atom_refiner = model.latent_residual.atom_refiner
    if atom_refiner is not None and any(
        torch.count_nonzero(layer.weight).item()
        for layer in (atom_refiner.atom_out, atom_refiner.res_out)
    ):
        raise ValueError("atom refiner readouts must be fresh and zero-initialized")
    # Complete validation precedes copying: failed admission cannot partially
    # replace a caller's model. No learned suffix is remapped into a new norm.
    own.update({key: source_state[key] for key in target_keys})
    model.load_state_dict(own, strict=True)
    return {
        "reference": "source backbone with old latent branch bypassed",
        "copied": sorted(target_keys),
        "initialized": sorted(set(own) - target_keys),
        "excluded_source": sorted(set(source_state) - source_keys),
        "plm_inject_at": list(model.plm_inject_at),
        "optimizer": "fresh; weights-only initialization",
    }


def initialize_refiner_from_checkpoint(model, path, config: dict) -> dict:
    """Load only EMA backbone weights; legacy optimizer/data/EMA state is unused."""
    import hashlib

    path = Path(path)
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    from mambafold.model.bimamba3 import mamba3_runtime_info

    source_runtime = state.get("mamba3_runtime")
    current_runtime = mamba3_runtime_info()
    if not isinstance(source_runtime, dict) or any(
        source_runtime.get(key) != current_runtime[key]
        for key in ("revision", "module_sha256", "decay")
    ):
        raise ValueError("refiner initialization requires the same recorded Mamba3 runtime")
    receipt = initialize_residue_refiner(model, state["ema"], state["args"], config)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    receipt.update(checkpoint_step=int(state["step"]), weight_source="ema",
                   checkpoint_sha256=digest.hexdigest())
    return receipt

def linear_warmup_lr(
    optimizer,
    warmup_steps: int,
    max_lr: float,
    min_lr: float = 1e-6,
    total_steps: int = 0,
    cooldown_steps: int = 0,
):
    """SimpleFold's `LinearWarmup`, plus an optional cosine cooldown.

    With a shared peak LR, cooldown_steps=0 reproduces the reference's linear
    ramp from min_lr to max_lr followed by a constant rate. The optional cosine
    tail is a separate recipe choice whose folding benefit is unmeasured.
    Each group uses its own peak LR times the same schedule multiplier, so a
    separate Muon LR retains its ratio to AdamW throughout the run. The group's
    warmup start and cooldown floor are its peak LR times min_lr / max_lr.
    """
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if not (0.0 < min_lr <= max_lr):
        raise ValueError("require 0 < min_lr <= max_lr")

    # Apply the same multiplier to each group's peak LR. Separate Muon/AdamW
    # rates keep their ratio through warmup and cooldown; min_lr anchors AdamW.
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
    if max_length > 0 and max_length % length_bin:
        lengths.append(max_length)
    if not lengths:
        return []
    torch_device = torch.device(device)
    d_plm = getattr(model, "d_plm", None) or getattr(getattr(model, "module", None), "d_plm", 2560)
    was_training = model.training
    model.train()
    warmed = []
    for length in lengths:
        batch = _synthetic_batch(batch_size, length, MAX_ATOMS_PER_RES, d_plm, torch_device)
        out = predictions = None
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch)
            predictions = out.get('v_atom_steps', (out['v_atom'],))
            sum(value.float().sum() for value in predictions).backward()
            model.zero_grad(set_to_none=True)
            warmed.append(length)
        except torch.cuda.OutOfMemoryError:
            # A shape training itself could not run either. Stop rather than
            # pretend the rest are warm.
            print(f"[prewarm] out of memory at L={length}; stopping", flush=True)
            break
        finally:
            # LiFT's intermediate graphs must not survive into the next shape.
            del out, predictions, batch
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
    from mambafold.model.bimamba3 import mamba3_runtime_info

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
            "mamba3_runtime": mamba3_runtime_info(raw_model),
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
