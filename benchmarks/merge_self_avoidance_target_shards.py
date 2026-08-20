#!/usr/bin/env python3
"""Atomically merge target-sharded self-avoidance inference outputs."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

_DEFAULT_EXPECTED_CONDITIONS = ("split_local_control", "steric_1")
_CONDITION_METADATA = (
    "checkpoint_sha256",
    "n_steps",
    "seed",
    "sampler",
    "sde_tau",
    "sde_eps",
    "sde_w_cutoff",
    "sde_log_timesteps",
    "cuda_device_name",
    "autocast_dtype",
    "geometry_guidance",
)
_TARGET_FILE_SUFFIXES = (
    "gt.pdb",
    "gt.cif",
    "pred.pdb",
    "pred.cif",
    "pred_seed0.pdb",
    "pred_seed0.cif",
)


@dataclass(frozen=True)
class _MergePlan:
    schema_version: Any
    experiment: Any
    conditions: tuple[str, ...]
    source_shards: tuple[Path, ...]
    source_ids_files: tuple[str, ...]
    source_target_ids: tuple[tuple[str, ...], ...]
    condition_templates: dict[str, dict[str, Any]]
    condition_rows: dict[str, dict[str, dict[str, Any]]]
    condition_sources: dict[str, dict[str, Path]]


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


def _require_path_component(value: Any, *, label: str, source: Path) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid {label} in {source}: {value!r}")
    if Path(value).parts != (value,) or value in {".", ".."}:
        raise ValueError(f"{label} must be one path component in {source}: {value!r}")
    return value


def _read_canonical_ids(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise ValueError(f"missing canonical IDs file: {path}")
    try:
        raw_ids = path.read_text().split()
    except OSError as exc:
        raise ValueError(f"cannot read canonical IDs file: {path}: {exc}") from exc
    target_ids = tuple(
        _require_path_component(value, label="target ID", source=path) for value in raw_ids
    )
    if not target_ids:
        raise ValueError(f"canonical IDs file is empty: {path}")
    if len(set(target_ids)) != len(target_ids):
        raise ValueError(f"duplicate target IDs in canonical IDs file: {path}")
    return target_ids


def _condition_metadata(manifest: dict[str, Any], path: Path) -> dict[str, Any]:
    missing = [key for key in _CONDITION_METADATA if key not in manifest]
    if missing:
        raise ValueError(f"condition manifest missing {missing}: {path}")
    return {key: manifest[key] for key in _CONDITION_METADATA}


def _condition_rows(
    manifest: dict[str, Any], target_count: int, path: Path
) -> tuple[tuple[str, ...], dict[str, dict[str, Any]]]:
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"condition manifest rows must be a list: {path}")
    if len(rows) != target_count:
        raise ValueError(
            f"condition target count mismatch in {path}: rows={len(rows)} expected={target_count}"
        )

    target_ids: list[str] = []
    rows_by_target: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"invalid row {index} in condition manifest: {path}")
        target_id = _require_path_component(
            row.get("pdb_id"), label=f"row {index} target ID", source=path
        )
        if target_id in rows_by_target:
            raise ValueError(f"duplicate target ID {target_id!r} in condition manifest: {path}")
        target_ids.append(target_id)
        rows_by_target[target_id] = deepcopy(row)
    return tuple(target_ids), rows_by_target


def _validate_target_files(condition_dir: Path, target_ids: tuple[str, ...]) -> None:
    for target_id in target_ids:
        for suffix in _TARGET_FILE_SUFFIXES:
            source = condition_dir / f"{target_id}_{suffix}"
            if not source.is_file():
                raise ValueError(f"missing canonical target file: {source}")


def _validate_expected_conditions(values: Sequence[str]) -> tuple[str, ...]:
    expected_conditions = tuple(values)
    if not expected_conditions:
        raise ValueError("expected conditions must not be empty")
    for condition in expected_conditions:
        if not isinstance(condition, str) or not condition:
            raise ValueError(f"invalid expected condition: {condition!r}")
        if Path(condition).parts != (condition,) or condition in {".", ".."}:
            raise ValueError(f"expected condition must be one path component: {condition!r}")
    if len(set(expected_conditions)) != len(expected_conditions):
        raise ValueError(f"duplicate expected conditions: {expected_conditions!r}")
    return expected_conditions


def _validate_shards(
    canonical_ids: tuple[str, ...],
    shards: list[Path],
    expected_conditions: tuple[str, ...],
) -> _MergePlan:
    reference_schema: Any = None
    reference_experiment: Any = None
    reference_metadata: dict[str, dict[str, Any]] = {}
    condition_templates: dict[str, dict[str, Any]] = {}
    condition_rows = {condition: {} for condition in expected_conditions}
    condition_sources = {condition: {} for condition in expected_conditions}
    source_ids_files: list[str] = []
    source_target_ids: list[tuple[str, ...]] = []
    all_targets: set[str] = set()

    for shard_index, shard in enumerate(shards):
        sweep_path = shard / "sweep_manifest.json"
        sweep = _read_object(sweep_path)
        if "schema_version" not in sweep or "experiment" not in sweep:
            raise ValueError(f"sweep manifest missing schema_version or experiment: {sweep_path}")

        raw_conditions = sweep.get("conditions")
        if not isinstance(raw_conditions, list):
            raise ValueError(f"sweep conditions must be a list: {sweep_path}")
        conditions = tuple(
            _require_path_component(value, label="condition", source=sweep_path)
            for value in raw_conditions
        )
        if conditions != expected_conditions:
            raise ValueError(
                f"unexpected conditions or order in {sweep_path}: {conditions!r}; "
                f"expected {expected_conditions!r}"
            )

        expected_manifests = [
            str(Path("inference") / condition / "manifest.json") for condition in conditions
        ]
        if sweep.get("condition_manifests") != expected_manifests:
            raise ValueError(f"condition_manifests does not match conditions: {sweep_path}")

        target_count = sweep.get("target_count")
        if not isinstance(target_count, int) or isinstance(target_count, bool) or target_count < 1:
            raise ValueError(f"invalid target_count in {sweep_path}: {target_count!r}")

        schema = sweep["schema_version"]
        experiment = sweep["experiment"]
        if shard_index == 0:
            reference_schema = schema
            reference_experiment = experiment
        elif schema != reference_schema or experiment != reference_experiment:
            raise ValueError(f"sweep schema/experiment mismatch: {sweep_path}")

        shard_targets: tuple[str, ...] | None = None
        shard_ids_file: str | None = None
        for condition in conditions:
            condition_dir = shard / "inference" / condition
            if not condition_dir.is_dir():
                raise ValueError(f"missing condition directory: {condition_dir}")
            manifest_path = condition_dir / "manifest.json"
            manifest = _read_object(manifest_path)
            if manifest.get("condition") != condition:
                raise ValueError(
                    f"condition field mismatch in {manifest_path}: "
                    f"{manifest.get('condition')!r} != {condition!r}"
                )
            if manifest.get("schema_version") != schema:
                raise ValueError(f"condition schema mismatch: {manifest_path}")

            metadata = _condition_metadata(manifest, manifest_path)
            if condition not in reference_metadata:
                reference_metadata[condition] = metadata
                condition_templates[condition] = deepcopy(manifest)
            elif metadata != reference_metadata[condition]:
                mismatched = [
                    key
                    for key in _CONDITION_METADATA
                    if metadata[key] != reference_metadata[condition][key]
                ]
                raise ValueError(
                    f"condition metadata mismatch for {condition} fields {mismatched}: "
                    f"{manifest_path}"
                )

            ids_file = manifest.get("ids_file")
            if not isinstance(ids_file, str) or not ids_file:
                raise ValueError(f"invalid ids_file in condition manifest: {manifest_path}")
            if shard_ids_file is None:
                shard_ids_file = ids_file
            elif ids_file != shard_ids_file:
                raise ValueError(f"condition ids_file differs within shard: {manifest_path}")

            targets, rows = _condition_rows(manifest, target_count, manifest_path)
            if shard_targets is None:
                shard_targets = targets
            elif targets != shard_targets:
                raise ValueError(
                    f"condition target IDs or order differ within shard: {manifest_path}"
                )
            _validate_target_files(condition_dir, targets)
            for target_id, row in rows.items():
                condition_rows[condition][target_id] = row
                condition_sources[condition][target_id] = condition_dir

        assert shard_targets is not None
        assert shard_ids_file is not None
        overlap = all_targets.intersection(shard_targets)
        if overlap:
            raise ValueError(f"target IDs overlap across shards: {sorted(overlap)}")
        all_targets.update(shard_targets)
        source_ids_files.append(shard_ids_file)
        source_target_ids.append(shard_targets)

    canonical_set = set(canonical_ids)
    missing = canonical_set - all_targets
    extra = all_targets - canonical_set
    if missing or extra:
        raise ValueError(
            f"target union differs from canonical IDs: missing={sorted(missing)} "
            f"extra={sorted(extra)}"
        )
    for condition in expected_conditions:
        if set(condition_rows[condition]) != canonical_set:
            raise ValueError(f"condition target union differs from canonical IDs: {condition}")

    return _MergePlan(
        schema_version=reference_schema,
        experiment=reference_experiment,
        conditions=expected_conditions,
        source_shards=tuple(shards),
        source_ids_files=tuple(source_ids_files),
        source_target_ids=tuple(source_target_ids),
        condition_templates=condition_templates,
        condition_rows=condition_rows,
        condition_sources=condition_sources,
    )


def _refuse_overwrite(root: Path) -> None:
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise FileExistsError(f"merge root exists and is not a directory: {root}")
    for output in (root / "inference", root / "sweep_manifest.json"):
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing merge output: {output}")


def _source_records(plan: _MergePlan) -> list[dict[str, Any]]:
    return [
        {
            "shard": str(shard),
            "ids_file": ids_file,
            "target_ids": list(target_ids),
        }
        for shard, ids_file, target_ids in zip(
            plan.source_shards,
            plan.source_ids_files,
            plan.source_target_ids,
            strict=True,
        )
    ]


def _stage_merge(
    staging: Path,
    canonical_ids_path: Path,
    canonical_ids: tuple[str, ...],
    plan: _MergePlan,
) -> dict[str, Any]:
    staged_inference = staging / "inference"
    staged_inference.mkdir()
    source_records = _source_records(plan)

    for condition in plan.conditions:
        destination = staged_inference / condition
        destination.mkdir()
        for target_id in canonical_ids:
            source_dir = plan.condition_sources[condition][target_id]
            for suffix in _TARGET_FILE_SUFFIXES:
                source = source_dir / f"{target_id}_{suffix}"
                target = destination / source.name
                if target.exists() or target.is_symlink():
                    raise ValueError(f"target file collision while merging: {target}")
                shutil.copy2(source, target)

        manifest = deepcopy(plan.condition_templates[condition])
        manifest.update(
            {
                "ids_file": str(canonical_ids_path),
                "rows": [plan.condition_rows[condition][target_id] for target_id in canonical_ids],
                "parallel_target_shard_merge": True,
                "source_shards": [str(shard) for shard in plan.source_shards],
                "source_ids_files": list(plan.source_ids_files),
                "source_target_shards": source_records,
            }
        )
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    unified = {
        "schema_version": plan.schema_version,
        "experiment": plan.experiment,
        "single_process_model_load": False,
        "parallel_target_shard_merge": True,
        "conditions": list(plan.conditions),
        "target_count": len(canonical_ids),
        "ids_file": str(canonical_ids_path),
        "condition_manifests": [
            str(Path("inference") / condition / "manifest.json") for condition in plan.conditions
        ],
        "source_shards": [str(shard) for shard in plan.source_shards],
        "source_ids_files": list(plan.source_ids_files),
        "source_target_shards": source_records,
    }
    (staging / "sweep_manifest.json").write_text(json.dumps(unified, indent=2) + "\n")
    return unified


def merge_target_shards(
    root: Path,
    canonical_ids_path: Path,
    shards: list[Path],
    *,
    expected_conditions: Sequence[str] = _DEFAULT_EXPECTED_CONDITIONS,
) -> dict[str, Any]:
    """Validate and atomically merge target shards in the expected condition order."""
    root = Path(root)
    canonical_ids_path = Path(canonical_ids_path)
    shards = [Path(shard) for shard in shards]
    expected_conditions = _validate_expected_conditions(expected_conditions)
    if len(shards) < 2:
        raise ValueError("at least two --shards are required")
    _refuse_overwrite(root)
    canonical_ids = _read_canonical_ids(canonical_ids_path)
    plan = _validate_shards(canonical_ids, shards, expected_conditions)

    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.target-merge-", dir=root.parent))
    root_preexisted = root.exists()
    committed_inference = False
    committed_manifest = False
    try:
        unified = _stage_merge(staging, canonical_ids_path, canonical_ids, plan)
        root.mkdir(parents=True, exist_ok=True)
        _refuse_overwrite(root)
        (staging / "inference").rename(root / "inference")
        committed_inference = True
        (staging / "sweep_manifest.json").rename(root / "sweep_manifest.json")
        committed_manifest = True
    except Exception:
        if committed_manifest:
            (root / "sweep_manifest.json").unlink(missing_ok=True)
        if committed_inference:
            shutil.rmtree(root / "inference", ignore_errors=True)
        if not root_preexisted and root.is_dir():
            try:
                root.rmdir()
            except OSError:
                pass
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return unified


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--canonical-ids", type=Path, required=True)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--expected-conditions",
        nargs="+",
        default=list(_DEFAULT_EXPECTED_CONDITIONS),
        help="ordered condition names required in every shard manifest",
    )
    args = parser.parse_args()
    if len(args.shards) < 2:
        parser.error("--shards requires at least two directories")
    try:
        manifest = merge_target_shards(
            args.root,
            args.canonical_ids,
            args.shards,
            expected_conditions=args.expected_conditions,
        )
    except (FileExistsError, OSError, ValueError) as exc:
        raise SystemExit(f"merge failed: {exc}") from exc
    print(
        f"[done] merged {len(args.shards)} target shards, "
        f"{manifest['target_count']} targets, and {len(manifest['conditions'])} conditions "
        f"into {args.root}"
    )


if __name__ == "__main__":
    main()
