#!/usr/bin/env python3
"""Validate and fingerprint the completed checkpoint used for geometry fine-tuning.

The fine-tune is deliberately initialized from one immutable source artifact:
the completed step-170,000 ESMC-6B checkpoint.  This gate rejects partial,
rotated, architecture-incompatible, or non-latest checkpoints before any GPU
training state is created, then writes an atomic provenance record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

CHECKPOINT_VERSION = 2
REQUIRED_FIELDS = {
    "checkpoint_version",
    "step",
    "model",
    "ema",
    "optimizer",
    "scheduler",
    "args",
    "wandb_run_id",
    "rng_states",
    "data_state",
}

# Values in these groups must be identical in the completed source checkpoint,
# its source YAML, and the fine-tune YAML.  Loss and optimizer settings are
# intentionally excluded because those are the controlled fine-tune changes.
ARCHITECTURE_KEYS = (
    "max_length",
    "d_res",
    "n_trunk",
    "d_res_type",
    "d_res_pos",
    "d_plm",
    "d_plm_proj",
    "d_ca_emb",
    "d_state",
    "mimo_rank",
    "expand",
    "headdim",
    "bimamba_share",
    "trunk_attn_layers",
    "trunk_attn_every",
    "n_attn_heads",
    "trunk_time_film",
    "trunk_adaln_zero",
    "use_pair_stack",
    "d_pair",
    "n_pair_blocks",
    "n_pair_heads",
    "pair_mult_c",
    "pair_use_cueq",
    "d_atom",
    "n_atom_layers",
    "use_plm",
)
DATA_KEYS = (
    "data_dir",
    "val_data_dir",
    "file_list",
    "val_file_list",
    "train_sources",
    "batch_size",
    "grad_accum_steps",
    "copies_per_protein",
    "single_chain_only",
    "extract_monomer_chains",
    "esm_dir",
    "length_bin",
    "length_balanced_sampling",
    "metadata_path",
    "length_balance_mode",
    "length_balance_exponent",
    "length_balance_clip_min",
    "length_balance_clip_max",
    "length_bucketing",
    "crop_schedule",
    "seed",
)

# These model-path defaults are supplied by argparse rather than written in the
# active YAML.  Pin them here so a checkpoint made with a CLI-only architecture
# override cannot slip through an otherwise matching source config.
ARCHITECTURE_DEFAULTS = {
    "self_conditioning": False,
    "pairfree_aux_heads": False,
}

DEFAULT_CODE_PATHS = (
    "scripts/train.py",
    "scripts/train.sh",
    "scripts/validate_geometry_finetune_source.py",
    "scripts/slurm_smoke_geometry_finetune_source.sh",
    "scripts/slurm_finetune_geometry_esmc6b_ada_8gpu.sh",
    "src/mambafold",
)
CODE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".py",
    ".sh",
    ".toml",
}


class SourceValidationError(RuntimeError):
    """Raised when the source checkpoint is not the exact admitted artifact."""


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise SourceValidationError(f"YAML root must be a mapping: {path}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceValidationError(f"{label} must be a mapping")
    return value


def _normalise(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_normalise(item) for item in value]
    if isinstance(value, list):
        return [_normalise(item) for item in value]
    return value


def _require_matching_contract(
    checkpoint_args: Mapping[str, Any],
    source_config: Mapping[str, Any],
    finetune_config: Mapping[str, Any],
    keys: Sequence[str],
    *,
    label: str,
) -> dict[str, Any]:
    admitted: dict[str, Any] = {}
    for key in keys:
        if key not in source_config:
            raise SourceValidationError(f"source config is missing {label} key {key!r}")
        if key not in checkpoint_args:
            raise SourceValidationError(f"checkpoint args are missing {label} key {key!r}")
        if key not in finetune_config:
            raise SourceValidationError(f"fine-tune config is missing {label} key {key!r}")
        source_value = _normalise(source_config[key])
        checkpoint_value = _normalise(checkpoint_args[key])
        finetune_value = _normalise(finetune_config[key])
        if checkpoint_value != source_value:
            raise SourceValidationError(
                f"checkpoint/source {label} mismatch for {key}: "
                f"checkpoint={checkpoint_value!r} source={source_value!r}"
            )
        if finetune_value != source_value:
            raise SourceValidationError(
                f"fine-tune/source {label} mismatch for {key}: "
                f"fine-tune={finetune_value!r} source={source_value!r}"
            )
        admitted[key] = source_value
    return admitted


def _validate_model_ema(model: Mapping[str, Any], ema: Mapping[str, Any]) -> dict[str, Any]:
    if not model:
        raise SourceValidationError("checkpoint model state is empty")
    model_keys = set(model)
    ema_keys = set(ema)
    if model_keys != ema_keys:
        missing = sorted(model_keys - ema_keys)
        extra = sorted(ema_keys - model_keys)
        raise SourceValidationError(
            f"model/EMA key mismatch: ema_missing={missing[:5]} ema_extra={extra[:5]}"
        )

    parameter_count = 0
    dtype_counts: dict[str, int] = {}
    for key in sorted(model_keys):
        model_value = model[key]
        ema_value = ema[key]
        if not isinstance(model_value, torch.Tensor) or not isinstance(ema_value, torch.Tensor):
            raise SourceValidationError(f"model/EMA entry {key!r} is not a tensor")
        if model_value.shape != ema_value.shape:
            raise SourceValidationError(
                f"model/EMA shape mismatch for {key}: "
                f"model={tuple(model_value.shape)} ema={tuple(ema_value.shape)}"
            )
        if model_value.dtype != ema_value.dtype:
            raise SourceValidationError(
                f"model/EMA dtype mismatch for {key}: "
                f"model={model_value.dtype} ema={ema_value.dtype}"
            )
        parameter_count += model_value.numel()
        dtype = str(model_value.dtype)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + model_value.numel()

    return {
        "state_key_count": len(model_keys),
        "state_value_count": parameter_count,
        "dtype_value_counts": dtype_counts,
    }


def _validate_scheduler(scheduler: Mapping[str, Any], expected_step: int) -> None:
    last_epoch = scheduler.get("last_epoch")
    if last_epoch != expected_step:
        raise SourceValidationError(
            f"scheduler last_epoch={last_epoch!r}, expected {expected_step}"
        )
    step_count = scheduler.get("_step_count")
    if step_count is not None and step_count != expected_step + 1:
        raise SourceValidationError(
            f"scheduler _step_count={step_count!r}, expected {expected_step + 1}"
        )
    base_lrs = scheduler.get("base_lrs")
    last_lrs = scheduler.get("_last_lr")
    if not isinstance(base_lrs, list) or not base_lrs:
        raise SourceValidationError("scheduler base_lrs must be a non-empty list")
    if not isinstance(last_lrs, list) or len(last_lrs) != len(base_lrs):
        raise SourceValidationError("scheduler _last_lr must match base_lrs")
    for value in [*base_lrs, *last_lrs]:
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise SourceValidationError(f"scheduler contains invalid learning rate {value!r}")


def _validate_data_state(
    data_state: Mapping[str, Any],
    checkpoint_args: Mapping[str, Any],
    rng_states: Any,
    *,
    expected_step: int,
    expected_world_size: int,
) -> dict[str, Any]:
    required = {
        "micro_batches_consumed",
        "world_size",
        "batch_size",
        "grad_accum_steps",
        "batches_per_epoch",
        "dataset_size",
        "sampler_type",
        "seed",
    }
    missing = sorted(required - set(data_state))
    if missing:
        raise SourceValidationError(f"data_state is missing fields: {missing}")
    for key in (
        "micro_batches_consumed",
        "world_size",
        "batch_size",
        "grad_accum_steps",
        "batches_per_epoch",
        "dataset_size",
    ):
        value = data_state[key]
        minimum = 0 if key == "micro_batches_consumed" else 1
        if type(value) is not int or value < minimum:
            raise SourceValidationError(
                f"data_state {key} must be an integer >= {minimum}, got {value!r}"
            )
    if type(data_state["seed"]) is not int:
        raise SourceValidationError(
            f"data_state seed must be an integer, got {data_state['seed']!r}"
        )
    if not isinstance(data_state["sampler_type"], str) or not data_state["sampler_type"]:
        raise SourceValidationError("data_state sampler_type must be a non-empty string")
    for key in ("batch_size", "grad_accum_steps", "seed"):
        if key not in checkpoint_args:
            raise SourceValidationError(f"checkpoint args are missing data key {key!r}")
    if data_state["world_size"] != expected_world_size:
        raise SourceValidationError(
            f"data_state world_size={data_state['world_size']!r}, expected {expected_world_size}"
        )
    for key in ("batch_size", "grad_accum_steps", "seed"):
        if data_state[key] != checkpoint_args.get(key):
            raise SourceValidationError(
                f"data_state/args mismatch for {key}: "
                f"data_state={data_state[key]!r} args={checkpoint_args.get(key)!r}"
            )
    minimum_micro_batches = expected_step * int(checkpoint_args["grad_accum_steps"])
    if data_state["micro_batches_consumed"] < minimum_micro_batches:
        raise SourceValidationError(
            "data_state micro_batches_consumed="
            f"{data_state['micro_batches_consumed']!r}, expected at least "
            f"{minimum_micro_batches}"
        )
    if not isinstance(rng_states, list) or len(rng_states) != expected_world_size:
        raise SourceValidationError(f"rng_states must contain {expected_world_size} rank states")
    return {key: _normalise(data_state[key]) for key in sorted(required)} | {
        "minimum_micro_batches": minimum_micro_batches,
        "extra_consumed_micro_batches": (
            data_state["micro_batches_consumed"] - minimum_micro_batches
        ),
        "rng_rank_count": len(rng_states),
    }


def _collect_code_files(repo_root: Path, code_paths: Sequence[str | Path]) -> list[Path]:
    files: set[Path] = set()
    for raw_path in code_paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = repo_root / path
        if not path.exists():
            raise SourceValidationError(f"code provenance path does not exist: {path}")
        if path.is_file():
            files.add(path.resolve())
            continue
        for candidate in path.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in CODE_SUFFIXES:
                files.add(candidate.resolve())
    if not files:
        raise SourceValidationError("no code files selected for provenance")
    return sorted(files, key=lambda item: item.as_posix())


def _code_hashes(repo_root: Path, code_paths: Sequence[str | Path]) -> tuple[str, dict[str, str]]:
    aggregate = hashlib.sha256()
    per_file: dict[str, str] = {}
    for path in _collect_code_files(repo_root, code_paths):
        try:
            label = path.relative_to(repo_root).as_posix()
        except ValueError:
            label = path.as_posix()
        digest = sha256_file(path)
        per_file[label] = digest
        aggregate.update(label.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return aggregate.hexdigest(), per_file


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    # Normal untracked reporting is enough for a dirty-tree bit and avoids
    # recursively enumerating ignored/large local dataset directories.
    status = run("status", "--porcelain=v1", "--untracked-files=normal")
    return {
        "commit": commit.strip() if commit else None,
        "dirty": bool(status),
        "status_sha256": (
            hashlib.sha256(status.encode("utf-8")).hexdigest() if status is not None else None
        ),
    }


def _display_path(path: Path, repo_root: Path, *, resolve_symlinks: bool = True) -> str:
    displayed = path.resolve() if resolve_symlinks else path.absolute()
    try:
        return displayed.relative_to(repo_root).as_posix()
    except ValueError:
        return displayed.as_posix()


def _atomic_write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise SourceValidationError(f"refusing to overwrite provenance: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def validate_geometry_finetune_source(
    *,
    checkpoint: str | Path,
    latest: str | Path,
    source_config: str | Path,
    finetune_config: str | Path,
    expected_step: int,
    expected_world_size: int,
    provenance_out: str | Path,
    repo_root: str | Path = ".",
    code_paths: Sequence[str | Path] = DEFAULT_CODE_PATHS,
    overwrite_provenance: bool = False,
) -> dict[str, Any]:
    repo_root_path = Path(repo_root).resolve()
    if expected_step <= 0:
        raise SourceValidationError(f"expected_step must be positive, got {expected_step}")
    if expected_world_size <= 0:
        raise SourceValidationError(
            f"expected_world_size must be positive, got {expected_world_size}"
        )
    checkpoint_path = Path(checkpoint)
    latest_path = Path(latest)
    source_config_path = Path(source_config)
    finetune_config_path = Path(finetune_config)
    provenance_path = Path(provenance_out)
    for label, path in (
        ("checkpoint", checkpoint_path),
        ("source config", source_config_path),
        ("fine-tune config", finetune_config_path),
    ):
        if not path.is_file():
            raise SourceValidationError(f"{label} does not exist: {path}")
    expected_name = f"ckpt_{expected_step:07d}.pt"
    if checkpoint_path.name != expected_name:
        raise SourceValidationError(
            f"source checkpoint must be named {expected_name}, got {checkpoint_path.name}"
        )
    if not latest_path.is_symlink():
        raise SourceValidationError(f"latest checkpoint is not a symlink: {latest_path}")
    if latest_path.resolve() != checkpoint_path.resolve():
        raise SourceValidationError(
            f"latest checkpoint resolves to {latest_path.resolve()}, "
            f"expected {checkpoint_path.resolve()}"
        )

    source_stat_before = checkpoint_path.stat()
    latest_link_before = os.readlink(latest_path)
    if latest_link_before != expected_name:
        raise SourceValidationError(
            f"latest symlink target must be {expected_name!r}, got {latest_link_before!r}"
        )
    try:
        checkpoint_data = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except Exception as exc:  # torch emits many format-specific exception classes.
        raise SourceValidationError(f"cannot load source checkpoint: {exc}") from exc
    checkpoint_map = _mapping(checkpoint_data, "checkpoint")
    missing_fields = sorted(REQUIRED_FIELDS - set(checkpoint_map))
    if missing_fields:
        raise SourceValidationError(f"checkpoint is missing fields: {missing_fields}")
    if checkpoint_map["checkpoint_version"] != CHECKPOINT_VERSION:
        raise SourceValidationError(
            f"checkpoint_version={checkpoint_map['checkpoint_version']!r}, "
            f"expected {CHECKPOINT_VERSION}"
        )
    if checkpoint_map["step"] != expected_step:
        raise SourceValidationError(
            f"checkpoint step={checkpoint_map['step']!r}, expected {expected_step}"
        )
    if not isinstance(checkpoint_map["wandb_run_id"], str) or not checkpoint_map["wandb_run_id"]:
        raise SourceValidationError("source checkpoint has no W&B run id")

    checkpoint_args = _mapping(checkpoint_map["args"], "checkpoint args")
    source_cfg = _load_yaml(source_config_path)
    finetune_cfg = _load_yaml(finetune_config_path)
    saved_config_value = checkpoint_args.get("config")
    if not isinstance(saved_config_value, str) or not saved_config_value:
        raise SourceValidationError("checkpoint args have no source config path")
    saved_config_path = Path(saved_config_value)
    if not saved_config_path.is_absolute():
        saved_config_path = repo_root_path / saved_config_path
    if saved_config_path.resolve() != source_config_path.resolve():
        raise SourceValidationError(
            "checkpoint args config path does not match the admitted source config: "
            f"checkpoint={saved_config_value!r} source={source_config_path}"
        )
    saved_out_dir = checkpoint_args.get("out_dir")
    if not isinstance(saved_out_dir, str) or not saved_out_dir:
        raise SourceValidationError("checkpoint args have no source output directory")
    saved_out_path = Path(saved_out_dir)
    if not saved_out_path.is_absolute():
        saved_out_path = repo_root_path / saved_out_path
    if saved_out_path.resolve() != checkpoint_path.parent.resolve():
        raise SourceValidationError(
            "checkpoint args out_dir does not own the source checkpoint: "
            f"checkpoint={saved_out_dir!r} actual={checkpoint_path.parent}"
        )
    if source_cfg.get("total_steps") != expected_step:
        raise SourceValidationError(
            f"source config total_steps={source_cfg.get('total_steps')!r}, expected {expected_step}"
        )
    if checkpoint_args.get("total_steps") != expected_step:
        raise SourceValidationError(
            f"checkpoint args total_steps={checkpoint_args.get('total_steps')!r}, "
            f"expected {expected_step}"
        )

    architecture_contract = _require_matching_contract(
        checkpoint_args,
        source_cfg,
        finetune_cfg,
        ARCHITECTURE_KEYS,
        label="architecture",
    )
    for key, expected_value in ARCHITECTURE_DEFAULTS.items():
        checkpoint_value = checkpoint_args.get(key, expected_value)
        source_value = source_cfg.get(key, expected_value)
        finetune_value = finetune_cfg.get(key, expected_value)
        if not (checkpoint_value == source_value == finetune_value == expected_value):
            raise SourceValidationError(
                f"architecture default {key} must remain {expected_value!r}: "
                f"checkpoint={checkpoint_value!r} source={source_value!r} "
                f"fine-tune={finetune_value!r}"
            )
        architecture_contract[key] = expected_value
    data_contract = _require_matching_contract(
        checkpoint_args,
        source_cfg,
        finetune_cfg,
        DATA_KEYS,
        label="data",
    )

    model_summary = _validate_model_ema(
        _mapping(checkpoint_map["model"], "model state"),
        _mapping(checkpoint_map["ema"], "EMA state"),
    )
    optimizer = _mapping(checkpoint_map["optimizer"], "optimizer state")
    if not isinstance(optimizer.get("param_groups"), list) or not optimizer["param_groups"]:
        raise SourceValidationError("optimizer state has no parameter groups")
    _validate_scheduler(_mapping(checkpoint_map["scheduler"], "scheduler state"), expected_step)
    data_state_summary = _validate_data_state(
        _mapping(checkpoint_map["data_state"], "data state"),
        checkpoint_args,
        checkpoint_map["rng_states"],
        expected_step=expected_step,
        expected_world_size=expected_world_size,
    )

    checkpoint_digest = sha256_file(checkpoint_path)
    code_digest, code_file_hashes = _code_hashes(repo_root_path, code_paths)
    source_stat_after = checkpoint_path.stat()
    if (
        source_stat_before.st_dev,
        source_stat_before.st_ino,
        source_stat_before.st_size,
        source_stat_before.st_mtime_ns,
    ) != (
        source_stat_after.st_dev,
        source_stat_after.st_ino,
        source_stat_after.st_size,
        source_stat_after.st_mtime_ns,
    ):
        raise SourceValidationError("source checkpoint changed during validation")
    if (
        not latest_path.is_symlink()
        or os.readlink(latest_path) != latest_link_before
        or latest_path.resolve() != checkpoint_path.resolve()
    ):
        raise SourceValidationError("ckpt_latest.pt changed during validation")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "checkpoint": _display_path(checkpoint_path, repo_root_path),
            "checkpoint_sha256": checkpoint_digest,
            "checkpoint_size_bytes": source_stat_after.st_size,
            "checkpoint_version": CHECKPOINT_VERSION,
            "step": expected_step,
            "wandb_run_id": checkpoint_map["wandb_run_id"],
            "latest": _display_path(latest_path, repo_root_path, resolve_symlinks=False),
            "latest_link": latest_link_before,
            "source_config": _display_path(source_config_path, repo_root_path),
            "source_config_sha256": sha256_file(source_config_path),
        },
        "finetune": {
            "config": _display_path(finetune_config_path, repo_root_path),
            "config_sha256": sha256_file(finetune_config_path),
            "initial_weights": "ema",
            "optimizer_scheduler": "fresh",
            "start_step": 0,
        },
        "contract": {
            "architecture": architecture_contract,
            "data": data_contract,
            "expected_world_size": expected_world_size,
            "model_ema": model_summary,
            "data_state": data_state_summary,
        },
        "code": {
            "aggregate_sha256": code_digest,
            "files": code_file_hashes,
            "git": _git_metadata(repo_root_path),
        },
    }
    _atomic_write_json(provenance_path, payload, overwrite=overwrite_provenance)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latest", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--finetune-config", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=170_000)
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--provenance-out", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--code-path",
        action="append",
        default=None,
        help="File or directory included in the code hash (repeatable).",
    )
    parser.add_argument("--overwrite-provenance", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        payload = validate_geometry_finetune_source(
            checkpoint=args.checkpoint,
            latest=args.latest,
            source_config=args.source_config,
            finetune_config=args.finetune_config,
            expected_step=args.expected_step,
            expected_world_size=args.expected_world_size,
            provenance_out=args.provenance_out,
            repo_root=args.repo_root,
            code_paths=args.code_path or DEFAULT_CODE_PATHS,
            overwrite_provenance=args.overwrite_provenance,
        )
    except SourceValidationError as exc:
        raise SystemExit(f"geometry fine-tune source validation failed: {exc}") from exc
    print(
        "geometry fine-tune source validated: "
        f"step={payload['source']['step']} "
        f"sha256={payload['source']['checkpoint_sha256']} "
        f"provenance={args.provenance_out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
