#!/usr/bin/env python
"""Train the standalone pLDDT head from frozen offline rollout latents."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist
import yaml
from scipy.stats import spearmanr
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from mambafold.confidence import (
    PLDDTHead,
    PLDDTHeadConfig,
    expected_lddt,
    macro_per_protein_cross_entropy,
    save_plddt_checkpoint,
    soft_adjacent_bin_labels,
)
from mambafold.confidence.rollout import (
    PLDDTRolloutDataset,
    ShardGroupedSampler,
    collate_plddt_rollouts,
    expand_manifest_paths,
    load_rollout_manifest,
    sha256_file,
)
from mambafold.data.esm import ESMC_6B_DIM, ESMC_6B_REVISION

_ROLLOUT_SAMPLER_KEYS = (
    "sampler",
    "n_steps",
    "sde_tau",
    "sde_eps",
    "sde_w_cutoff",
    "sde_log_timesteps",
    "geometry_guidance",
    "label",
    "coordinate_unit",
)
_ROLLOUT_CONDITIONING_KEYS = ("model", "revision", "embedding_dimensions")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-manifest", action="append", default=None)
    parser.add_argument("--val-manifest", action="append", default=None)
    parser.add_argument("--out-dir", default=None)
    return parser


def _setup_distributed(seed: int) -> tuple[bool, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = world_size > 1
    if is_distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if is_distributed else "cuda")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)
    return is_distributed, rank, world_size, device


def _run_rank0_checked(
    action: Callable[[], None],
    *,
    rank: int,
    world_size: int,
    description: str,
) -> None:
    """Run one filesystem mutation on rank 0 and propagate failures to all ranks."""
    error: str | None = None
    if rank == 0:
        try:
            action()
        except Exception as exc:  # propagated, not hidden
            error = f"{type(exc).__name__}: {exc}"
    if world_size > 1:
        error_box = [error]
        dist.broadcast_object_list(error_box, src=0)
        error = error_box[0]
    if error is not None:
        raise RuntimeError(f"{description} failed on rank 0: {error}")


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _rollout_contract_and_split_inventory(
    train_manifests: list[Path],
    val_manifests: list[Path],
) -> dict[str, Any]:
    """Require one folding/sampler contract and disjoint canonical sequences."""
    consensus: dict[str, Any] | None = None
    sequence_sets: dict[str, set[str]] = {"train": set(), "val": set()}
    split_manifests: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for split, paths in (("train", train_manifests), ("val", val_manifests)):
        for path in paths:
            manifest = load_rollout_manifest(path)
            provenance = manifest.get("provenance")
            if not isinstance(provenance, dict):
                raise ValueError(f"manifest lacks provenance: {path}")
            checkpoint = provenance.get("checkpoint")
            sampler_config = provenance.get("config")
            conditioning = provenance.get("conditioning")
            split_config = provenance.get("split")
            provenance_parts = (checkpoint, sampler_config, conditioning, split_config)
            if not all(isinstance(value, dict) for value in provenance_parts):
                raise ValueError(f"manifest has malformed rollout provenance: {path}")
            assert isinstance(checkpoint, dict)
            assert isinstance(sampler_config, dict)
            assert isinstance(conditioning, dict)
            assert isinstance(split_config, dict)
            checkpoint_keys = ("basename", "sha256", "step", "weights")
            missing_checkpoint = [key for key in checkpoint_keys if key not in checkpoint]
            missing_sampler = [key for key in _ROLLOUT_SAMPLER_KEYS if key not in sampler_config]
            missing_conditioning = [
                key for key in _ROLLOUT_CONDITIONING_KEYS if key not in conditioning
            ]
            if missing_checkpoint or missing_sampler or missing_conditioning:
                raise ValueError(
                    f"manifest rollout contract is incomplete: {path}; "
                    f"checkpoint_missing={missing_checkpoint}, sampler_missing={missing_sampler}, "
                    f"conditioning_missing={missing_conditioning}"
                )
            checkpoint_hash = checkpoint["sha256"]
            file_list_hash = split_config.get("file_list_sha256")
            for name, value in (
                ("checkpoint.sha256", checkpoint_hash),
                ("split.file_list_sha256", file_list_hash),
            ):
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise ValueError(f"manifest has invalid {name}: {path}")
            file_list_basename = split_config.get("file_list_basename")
            if not isinstance(file_list_basename, str) or not file_list_basename:
                raise ValueError(f"manifest lacks split file-list basename: {path}")
            if type(checkpoint["step"]) is not int or checkpoint["step"] < 1:
                raise ValueError(f"manifest has invalid folding checkpoint step: {path}")
            if checkpoint["weights"] not in {"ema", "model"}:
                raise ValueError(f"manifest has invalid folding weight selector: {path}")
            if type(sampler_config["n_steps"]) is not int or sampler_config["n_steps"] < 1:
                raise ValueError(f"manifest has invalid sampler n_steps: {path}")
            if (
                conditioning["model"] != "biohub/ESMC-6B"
                or conditioning["revision"] != ESMC_6B_REVISION
                or conditioning["embedding_dimensions"] != ESMC_6B_DIM
            ):
                raise ValueError(f"manifest has incompatible ESMC-6B conditioning: {path}")
            current = {
                "folding_checkpoint": {key: checkpoint[key] for key in checkpoint_keys},
                "conditioning": {
                    key: conditioning[key] for key in _ROLLOUT_CONDITIONING_KEYS
                },
                **{key: sampler_config[key] for key in _ROLLOUT_SAMPLER_KEYS},
            }
            if consensus is None:
                consensus = current
            elif current != consensus:
                raise ValueError(
                    "refusing mixed folding/sampler rollout contracts; "
                    f"first={consensus}, mismatch={path.name}:{current}"
                )
            sequences = set(manifest["sequence_sha256s"])
            sequence_sets[split].update(sequences)
            split_manifests[split].append(
                {
                    "basename": path.name,
                    "sha256": sha256_file(path),
                    "file_list_basename": split_config.get("file_list_basename"),
                    "file_list_sha256": split_config.get("file_list_sha256"),
                    "num_sequences": len(sequences),
                }
            )
    overlap = sequence_sets["train"] & sequence_sets["val"]
    if overlap:
        examples = sorted(overlap)[:5]
        raise ValueError(
            "confidence train/validation canonical sequence overlap: "
            f"count={len(overlap)} examples={examples}"
        )
    if consensus is None:
        raise ValueError("no rollout manifests were supplied")
    return {**consensus, "split_manifests": split_manifests}


def _cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
):
    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _loader(
    dataset: PLDDTRolloutDataset,
    *,
    batch_size: int,
    num_workers: int,
    sampler: ShardGroupedSampler,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_plddt_rollouts,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def _to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, ...]:
    latent = batch["trunk_latent"].to(device=device, dtype=torch.float32, non_blocking=True)
    residue_mask = batch["residue_mask"].to(device=device, non_blocking=True)
    target = batch["plddt_target"].to(device=device, non_blocking=True)
    target_mask = batch["target_mask"].to(device=device, non_blocking=True)
    return latent, residue_mask, target, target_mask


def _safe_spearman(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.size < 2 or np.ptp(prediction) == 0 or np.ptp(target) == 0:
        return 0.0
    result = spearmanr(prediction, target)
    value = float(result.statistic)
    return value if math.isfinite(value) else 0.0


@torch.no_grad()
def _validate(
    model,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    n_bins: int,
    ece_bins: int,
    max_spearman_residues: int,
    rank: int,
    world_size: int,
) -> dict[str, float]:
    model.eval()
    # [loss_sum, proteins, residues, abs_error, squared_error,
    #  sum_pred, sum_target, sum_pred2, sum_target2, sum_product]
    stats = torch.zeros(10, dtype=torch.float64, device=device)
    ece_count = torch.zeros(ece_bins, dtype=torch.float64, device=device)
    ece_prediction = torch.zeros_like(ece_count)
    ece_target = torch.zeros_like(ece_count)
    local_cap = math.ceil(max_spearman_residues / world_size)
    per_batch_cap = max(1, math.ceil(local_cap / max(1, len(loader))))
    sample_prediction_parts: list[np.ndarray] = []
    sample_target_parts: list[np.ndarray] = []
    rng = np.random.default_rng(730_019 + rank)
    for batch in loader:
        latent, residue_mask, target, target_mask = _to_device(batch, device)
        labels = soft_adjacent_bin_labels(target, target_mask, n_bins)
        autocast = (
            torch.amp.autocast("cuda", dtype=amp_dtype)
            if amp_enabled
            else nullcontext()
        )
        with autocast:
            logits = model(latent, residue_mask)
            loss = macro_per_protein_cross_entropy(logits, labels, target_mask)
        prediction = expected_lddt(logits.float())
        valid = target_mask
        valid_proteins = int(valid.any(dim=1).sum())
        valid_prediction = prediction[valid].double()
        valid_target = target[valid].double()
        error = valid_prediction - valid_target
        stats[0] += loss.detach().double() * valid_proteins
        stats[1] += valid_proteins
        stats[2] += valid_prediction.numel()
        stats[3] += error.abs().sum()
        stats[4] += error.square().sum()
        stats[5] += valid_prediction.sum()
        stats[6] += valid_target.sum()
        stats[7] += valid_prediction.square().sum()
        stats[8] += valid_target.square().sum()
        stats[9] += (valid_prediction * valid_target).sum()

        bin_ids = (valid_prediction * ece_bins).long().clamp(max=ece_bins - 1)
        ece_count += torch.bincount(bin_ids, minlength=ece_bins)
        ece_prediction.scatter_add_(0, bin_ids, valid_prediction)
        ece_target.scatter_add_(0, bin_ids, valid_target)

        if valid_prediction.numel() and local_cap:
            prediction_cpu = valid_prediction.float().cpu().numpy()
            target_cpu = valid_target.float().cpu().numpy()
            take = min(per_batch_cap, prediction_cpu.size)
            if take < prediction_cpu.size:
                chosen = rng.choice(prediction_cpu.size, size=take, replace=False)
                prediction_cpu = prediction_cpu[chosen]
                target_cpu = target_cpu[chosen]
            sample_prediction_parts.append(prediction_cpu)
            sample_target_parts.append(target_cpu)

    if world_size > 1:
        dist.all_reduce(stats)
        dist.all_reduce(ece_count)
        dist.all_reduce(ece_prediction)
        dist.all_reduce(ece_target)
    residue_count = int(stats[2].item())
    if residue_count == 0:
        raise RuntimeError("validation produced no labeled residues")

    local_prediction = (
        np.concatenate(sample_prediction_parts).astype(np.float32, copy=False)
        if sample_prediction_parts
        else np.empty(0, dtype=np.float32)
    )
    local_target = (
        np.concatenate(sample_target_parts).astype(np.float32, copy=False)
        if sample_target_parts
        else np.empty(0, dtype=np.float32)
    )
    if local_prediction.size > local_cap:
        chosen = rng.choice(local_prediction.size, size=local_cap, replace=False)
        local_prediction = local_prediction[chosen]
        local_target = local_target[chosen]
    local_sample = (local_prediction.tolist(), local_target.tolist())
    if world_size > 1:
        gathered_samples = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_sample, gathered_samples, dst=0)
    else:
        gathered_samples = [local_sample]

    metrics: dict[str, float] | None = None
    if rank == 0:
        assert gathered_samples is not None
        sampled_prediction = np.asarray(
            [value for prediction_values, _ in gathered_samples for value in prediction_values],
            dtype=np.float64,
        )
        sampled_target = np.asarray(
            [value for _, target_values in gathered_samples for value in target_values],
            dtype=np.float64,
        )
        if sampled_prediction.size > max_spearman_residues:
            chosen = rng.choice(
                sampled_prediction.size,
                size=max_spearman_residues,
                replace=False,
            )
            sampled_prediction = sampled_prediction[chosen]
            sampled_target = sampled_target[chosen]

        count = stats[2]
        covariance = count * stats[9] - stats[5] * stats[6]
        pred_variance = (count * stats[7] - stats[5].square()).clamp_min(0)
        target_variance = (count * stats[8] - stats[6].square()).clamp_min(0)
        pearson_denom = torch.sqrt(pred_variance * target_variance)
        pearson = float(covariance / pearson_denom) if pearson_denom > 0 else 0.0
        occupied = ece_count > 0
        calibration_gap = torch.zeros_like(ece_count)
        calibration_gap[occupied] = (
            ece_prediction[occupied] / ece_count[occupied]
            - ece_target[occupied] / ece_count[occupied]
        ).abs()
        ece = float((calibration_gap * ece_count).sum() / count)
        metrics = {
            "val_loss": float(stats[0] / stats[1].clamp_min(1)),
            "val_mae": float(stats[3] / count),
            "val_rmse": float(torch.sqrt(stats[4] / count)),
            "val_pearson": pearson,
            "val_spearman_sampled": _safe_spearman(sampled_prediction, sampled_target),
            "val_ece": ece,
            "val_labeled_residues": float(residue_count),
            "val_spearman_residues": float(sampled_prediction.size),
        }
    if world_size > 1:
        metrics_box = [metrics]
        dist.broadcast_object_list(metrics_box, src=0)
        metrics = metrics_box[0]
    assert metrics is not None
    return metrics


def _manifest_provenance(paths: list[Path]) -> list[dict[str, str]]:
    return [{"basename": path.name, "sha256": sha256_file(path)} for path in paths]


def main() -> None:
    args = _parser().parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("pLDDT config schema_version must be 1")
    data_cfg = dict(config.get("data") or {})
    train_cfg = dict(config.get("training") or {})
    head_config = PLDDTHeadConfig(**dict(config.get("head") or {}))

    train_values = args.train_manifest or data_cfg.get("train_manifests") or []
    val_values = args.val_manifest or data_cfg.get("val_manifests") or []
    train_manifests = expand_manifest_paths(train_values)
    val_manifests = expand_manifest_paths(val_values)
    rollout_contract = _rollout_contract_and_split_inventory(
        train_manifests,
        val_manifests,
    )
    out_dir = Path(args.out_dir or train_cfg.get("out_dir") or "outputs/plddt")

    seed = int(train_cfg.get("seed", 0))
    is_distributed, rank, world_size, device = _setup_distributed(seed)
    is_main = rank == 0
    _run_rank0_checked(
        lambda: out_dir.mkdir(parents=True, exist_ok=False),
        rank=rank,
        world_size=world_size,
        description="create fresh pLDDT output directory",
    )
    batch_size = int(train_cfg.get("batch_size", 16))
    num_workers = int(train_cfg.get("num_workers", 4))
    verify_hashes = bool(data_cfg.get("verify_hashes", True))
    verify_eagerly = bool(data_cfg.get("verify_eagerly", False))
    train_dataset = PLDDTRolloutDataset(
        train_manifests,
        verify_hashes=verify_hashes,
        verify_eagerly=verify_eagerly,
        cache_size=int(data_cfg.get("shard_cache_size", 2)),
    )
    val_dataset = PLDDTRolloutDataset(
        val_manifests,
        verify_hashes=verify_hashes,
        verify_eagerly=verify_eagerly,
        cache_size=int(data_cfg.get("shard_cache_size", 2)),
    )
    sample_width = int(train_dataset[0]["trunk_latent"].shape[-1])
    if sample_width != head_config.d_model:
        raise ValueError(
            f"rollout latent width {sample_width} does not match head.d_model={head_config.d_model}"
        )

    train_sampler = ShardGroupedSampler(
        train_dataset,
        shuffle=True,
        seed=seed,
        rank=rank,
        world_size=world_size,
        even_divisible=True,
    )
    val_sampler = ShardGroupedSampler(
        val_dataset,
        shuffle=False,
        seed=seed,
        rank=rank,
        world_size=world_size,
        even_divisible=False,
    )
    train_loader = _loader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=device.type == "cuda",
    )
    val_loader = _loader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=val_sampler,
        pin_memory=device.type == "cuda",
    )

    head = PLDDTHead(head_config).to(device)
    model = (
        DistributedDataParallel(
            head,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
        if is_distributed
        else head
    )
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(train_cfg.get("lr", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        fused=device.type == "cuda",
    )
    total_steps = int(train_cfg.get("total_steps", 20_000))
    warmup_steps = int(train_cfg.get("warmup_steps", 1_000))
    scheduler = _cosine_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )
    amp_dtype_name = str(train_cfg.get("amp_dtype", "bf16"))
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype is torch.float16)
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_interval = int(train_cfg.get("log_interval", 50))
    eval_interval = int(train_cfg.get("eval_interval", 500))
    ckpt_interval = int(train_cfg.get("ckpt_interval", eval_interval))
    ece_bins = int(train_cfg.get("ece_bins", 10))
    max_spearman_residues = int(train_cfg.get("max_spearman_residues", 200_000))
    if (
        total_steps < 1
        or eval_interval < 1
        or ckpt_interval < 1
        or max_spearman_residues < 1
    ):
        raise ValueError(
            "total_steps, eval_interval, ckpt_interval, and max_spearman_residues "
            "must be positive"
        )

    provenance = {
        "config": {"basename": config_path.name, "sha256": sha256_file(config_path)},
        "train_manifests": _manifest_provenance(train_manifests),
        "val_manifests": _manifest_provenance(val_manifests),
        "rollout_contract": rollout_contract,
        "folding_backbone": "frozen-offline-latents",
        "label": "hard_lDDT-Ca",
    }
    resolved_run_config = {
        "schema_version": 1,
        "source_config": provenance["config"],
        "rollout_contract": rollout_contract,
        "head": head_config.to_dict(),
        "data": {
            "train_manifests": provenance["train_manifests"],
            "val_manifests": provenance["val_manifests"],
            "verify_hashes": verify_hashes,
            "verify_eagerly": verify_eagerly,
            "shard_cache_size": int(data_cfg.get("shard_cache_size", 2)),
        },
        "training": {
            "seed": seed,
            "world_size": world_size,
            "batch_size_per_rank": batch_size,
            "num_workers_per_rank": num_workers,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "lr": float(train_cfg.get("lr", 2e-4)),
            "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
            "grad_clip": grad_clip,
            "amp": amp_enabled,
            "amp_dtype": amp_dtype_name,
            "log_interval": log_interval,
            "eval_interval": eval_interval,
            "ckpt_interval": ckpt_interval,
            "ece_bins": ece_bins,
            "max_spearman_residues": max_spearman_residues,
        },
    }
    _run_rank0_checked(
        lambda: _write_json_exclusive(out_dir / "run_config.json", resolved_run_config),
        rank=rank,
        world_size=world_size,
        description="write portable pLDDT run config",
    )
    if is_main:
        params = sum(parameter.numel() for parameter in head.parameters())
        print(
            f"[setup] device={device} world_size={world_size} train={len(train_dataset)} "
            f"val={len(val_dataset)} head_params={params:,}",
            flush=True,
        )

    step = 0
    epoch = 0
    best_mae = math.inf
    last_metrics: dict[str, float] = {}
    while step < total_steps:
        train_sampler.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            latent, residue_mask, target, target_mask = _to_device(batch, device)
            labels = soft_adjacent_bin_labels(target, target_mask, head_config.n_bins)
            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.amp.autocast("cuda", dtype=amp_dtype)
                if amp_enabled
                else nullcontext()
            )
            with autocast:
                logits = model(latent, residue_mask)
                loss = macro_per_protein_cross_entropy(logits, labels, target_mask)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1

            if is_main and (step == 1 or step % log_interval == 0):
                print(
                    f"[train] step={step}/{total_steps} loss={float(loss.detach()):.6f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e}",
                    flush=True,
                )

            should_eval = step % eval_interval == 0 or step == total_steps
            if should_eval:
                last_metrics = _validate(
                    model,
                    val_loader,
                    device=device,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    n_bins=head_config.n_bins,
                    ece_bins=ece_bins,
                    max_spearman_residues=max_spearman_residues,
                    rank=rank,
                    world_size=world_size,
                )
                if is_main:
                    summary = " ".join(
                        f"{key}={value:.6f}"
                        for key, value in last_metrics.items()
                        if key not in {"val_labeled_residues", "val_spearman_residues"}
                    )
                    print(f"[val] step={step} {summary}", flush=True)
                is_best = last_metrics["val_mae"] < best_mae
                if is_best:
                    best_mae = last_metrics["val_mae"]
                    _run_rank0_checked(
                        lambda: save_plddt_checkpoint(
                            out_dir / "plddt_head_best.pt",
                            head,
                            step=step,
                            metrics=last_metrics,
                            provenance=provenance,
                        ),
                        rank=rank,
                        world_size=world_size,
                        description="save best pLDDT head",
                    )
                model.train()

            if step % ckpt_interval == 0 or step == total_steps:
                checkpoint_metrics = dict(last_metrics)
                checkpoint_metrics["train_loss"] = float(loss.detach())
                _run_rank0_checked(
                    lambda: save_plddt_checkpoint(
                        out_dir / "plddt_head_latest.pt",
                        head,
                        step=step,
                        metrics=checkpoint_metrics,
                        provenance=provenance,
                    ),
                    rank=rank,
                    world_size=world_size,
                    description="save latest pLDDT head",
                )
            if step >= total_steps:
                break
        epoch += 1

    if is_main:
        print(f"[done] step={step} artifacts={out_dir}", flush=True)
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
