#!/usr/bin/env python3
"""Merge independently produced self-avoidance sweep condition shards."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CONDITION_METADATA = ("checkpoint_sha256", "n_steps", "seed", "ids_file")


@dataclass(frozen=True)
class _ConditionSource:
    name: str
    directory: Path


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing manifest: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON manifest: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"manifest must contain a JSON object: {path}")
    return value


def _require_condition_name(value: Any, *, manifest: Path) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid condition name in {manifest}: {value!r}")
    if Path(value).parts != (value,) or value in {".", ".."}:
        raise ValueError(f"condition name must be one path component: {value!r}")
    return value


def _require_metadata(manifest: dict[str, Any], path: Path) -> dict[str, Any]:
    missing = [key for key in _CONDITION_METADATA if key not in manifest]
    if missing:
        raise ValueError(f"condition manifest missing {missing}: {path}")
    return {key: manifest[key] for key in _CONDITION_METADATA}


def _target_ids(manifest: dict[str, Any], target_count: int, path: Path) -> frozenset[str]:
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"condition manifest rows must be a list: {path}")
    if len(rows) != target_count:
        raise ValueError(
            f"condition target count mismatch in {path}: rows={len(rows)} expected={target_count}"
        )
    target_ids: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("pdb_id"), str):
            raise ValueError(f"invalid row {index} in condition manifest: {path}")
        target_ids.append(row["pdb_id"])
    if len(set(target_ids)) != len(target_ids):
        raise ValueError(f"duplicate target IDs in condition manifest: {path}")
    return frozenset(target_ids)


def _validate_shards(
    shards: list[Path],
) -> tuple[list[_ConditionSource], dict[str, Any]]:
    sources: list[_ConditionSource] = []
    seen_conditions: set[str] = set()
    reference_metadata: dict[str, Any] | None = None
    reference_targets: frozenset[str] | None = None
    reference_target_count: int | None = None
    reference_schema: Any = None
    reference_experiment: Any = None

    for shard_index, shard in enumerate(shards):
        sweep_path = shard / "sweep_manifest.json"
        sweep = _read_object(sweep_path)
        conditions = sweep.get("conditions")
        if not isinstance(conditions, list) or not conditions:
            raise ValueError(f"sweep conditions must be a non-empty list: {sweep_path}")
        names = [_require_condition_name(value, manifest=sweep_path) for value in conditions]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate conditions within shard manifest: {sweep_path}")

        target_count = sweep.get("target_count")
        if not isinstance(target_count, int) or isinstance(target_count, bool) or target_count < 0:
            raise ValueError(f"invalid target_count in {sweep_path}: {target_count!r}")
        if reference_target_count is None:
            reference_target_count = target_count
        elif target_count != reference_target_count:
            raise ValueError(
                f"target_count mismatch: {sweep_path} has {target_count}, "
                f"expected {reference_target_count}"
            )

        expected_paths = [str(Path("inference") / name / "manifest.json") for name in names]
        if sweep.get("condition_manifests") != expected_paths:
            raise ValueError(f"condition_manifests does not match conditions: {sweep_path}")

        schema = sweep.get("schema_version")
        experiment = sweep.get("experiment")
        if shard_index == 0:
            reference_schema = schema
            reference_experiment = experiment
        elif schema != reference_schema or experiment != reference_experiment:
            raise ValueError(f"sweep schema/experiment mismatch: {sweep_path}")

        for name in names:
            if name in seen_conditions:
                raise ValueError(f"duplicate condition across shards: {name}")
            seen_conditions.add(name)
            condition_dir = shard / "inference" / name
            if not condition_dir.is_dir():
                raise ValueError(f"missing condition directory: {condition_dir}")
            condition_path = condition_dir / "manifest.json"
            condition = _read_object(condition_path)
            if condition.get("condition") != name:
                raise ValueError(
                    f"condition field mismatch in {condition_path}: "
                    f"{condition.get('condition')!r} != {name!r}"
                )

            metadata = _require_metadata(condition, condition_path)
            if reference_metadata is None:
                reference_metadata = metadata
            elif metadata != reference_metadata:
                mismatched = [
                    key for key in _CONDITION_METADATA if metadata[key] != reference_metadata[key]
                ]
                raise ValueError(f"condition metadata mismatch for {mismatched}: {condition_path}")

            targets = _target_ids(condition, target_count, condition_path)
            if reference_targets is None:
                reference_targets = targets
            elif targets != reference_targets:
                raise ValueError(f"condition target IDs differ: {condition_path}")
            sources.append(_ConditionSource(name=name, directory=condition_dir))

    if reference_metadata is None or reference_target_count is None:
        raise ValueError("no condition manifests found")
    unified = {
        "schema_version": reference_schema,
        "experiment": reference_experiment,
        "single_process_model_load": False,
        "parallel_shard_merge": True,
        "conditions": [source.name for source in sources],
        "target_count": reference_target_count,
        **reference_metadata,
        "condition_manifests": [
            str(Path("inference") / source.name / "manifest.json") for source in sources
        ],
        "source_shards": [str(shard) for shard in shards],
    }
    return sources, unified


def _refuse_overwrite(root: Path) -> None:
    if root.exists() and not root.is_dir():
        raise FileExistsError(f"merge root exists and is not a directory: {root}")
    for output in (root / "inference", root / "sweep_manifest.json"):
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing merge output: {output}")


def merge_sweep_shards(root: Path, shards: list[Path]) -> dict[str, Any]:
    """Validate and atomically merge condition directories from two or more shards."""
    root = Path(root)
    shards = [Path(shard) for shard in shards]
    if len(shards) < 2:
        raise ValueError("at least two --shards are required")
    _refuse_overwrite(root)
    sources, unified = _validate_shards(shards)

    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.merge-", dir=root.parent))
    staged_inference = staging / "inference"
    staged_manifest = staging / "sweep_manifest.json"
    committed_inference = False
    committed_manifest = False
    try:
        staged_inference.mkdir()
        for source in sources:
            shutil.copytree(source.directory, staged_inference / source.name)
        staged_manifest.write_text(json.dumps(unified, indent=2) + "\n")

        root.mkdir(parents=True, exist_ok=True)
        _refuse_overwrite(root)
        staged_inference.rename(root / "inference")
        committed_inference = True
        staged_manifest.rename(root / "sweep_manifest.json")
        committed_manifest = True
    except Exception:
        if committed_manifest:
            (root / "sweep_manifest.json").unlink(missing_ok=True)
        if committed_inference:
            shutil.rmtree(root / "inference", ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return unified


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    if len(args.shards) < 2:
        parser.error("--shards requires at least two directories")
    try:
        manifest = merge_sweep_shards(args.root, args.shards)
    except (FileExistsError, OSError, ValueError) as exc:
        raise SystemExit(f"merge failed: {exc}") from exc
    print(
        f"[done] merged {len(args.shards)} shards and "
        f"{len(manifest['conditions'])} conditions into {args.root}"
    )


if __name__ == "__main__":
    main()
