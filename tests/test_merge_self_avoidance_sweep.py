"""Tests for atomic self-avoidance sweep shard merging."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from benchmarks.merge_self_avoidance_sweep import merge_sweep_shards

_BASE_METADATA = {
    "checkpoint_sha256": "abc123",
    "n_steps": 500,
    "seed": 0,
    "ids_file": "benchmarks/sets/casp14_self_overlap2.txt",
}
_TARGETS = ("t1036s1", "t1043")


def _write_shard(
    path: Path,
    conditions: list[str],
    *,
    metadata: dict[str, object] | None = None,
    target_count: int = 2,
) -> Path:
    metadata = {**_BASE_METADATA, **(metadata or {})}
    condition_paths = []
    for condition in conditions:
        condition_dir = path / "inference" / condition
        condition_dir.mkdir(parents=True)
        rows = [{"pdb_id": target} for target in _TARGETS[:target_count]]
        manifest = {
            "schema_version": 1,
            "condition": condition,
            **metadata,
            "rows": rows,
        }
        (condition_dir / "manifest.json").write_text(json.dumps(manifest))
        (condition_dir / f"{condition}.marker").write_text(condition)
        condition_paths.append(str(Path("inference") / condition / "manifest.json"))
    sweep = {
        "schema_version": 1,
        "experiment": "self_overlap_guidance_v1",
        "single_process_model_load": True,
        "conditions": conditions,
        "target_count": target_count,
        "condition_manifests": condition_paths,
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "sweep_manifest.json").write_text(json.dumps(sweep))
    return path


def test_merge_shards_copies_conditions_and_writes_unified_manifest(tmp_path: Path):
    shard_a = _write_shard(tmp_path / "shard-a", ["baseline", "split_local_control"])
    shard_b = _write_shard(tmp_path / "shard-b", ["steric_0p1", "steric_0p2"])
    root = tmp_path / "merged"

    result = merge_sweep_shards(root, [shard_a, shard_b])

    assert result["conditions"] == [
        "baseline",
        "split_local_control",
        "steric_0p1",
        "steric_0p2",
    ]
    assert result["target_count"] == 2
    assert result["checkpoint_sha256"] == "abc123"
    assert result["parallel_shard_merge"] is True
    assert result["single_process_model_load"] is False
    assert json.loads((root / "sweep_manifest.json").read_text()) == result
    for condition in result["conditions"]:
        assert (root / "inference" / condition / "manifest.json").is_file()
        assert (root / "inference" / condition / f"{condition}.marker").read_text() == condition


def test_merge_rejects_duplicate_conditions_without_partial_output(tmp_path: Path):
    shard_a = _write_shard(tmp_path / "shard-a", ["baseline"])
    shard_b = _write_shard(tmp_path / "shard-b", ["baseline"])
    root = tmp_path / "merged"

    with pytest.raises(ValueError, match="duplicate condition"):
        merge_sweep_shards(root, [shard_a, shard_b])

    assert not (root / "inference").exists()
    assert not (root / "sweep_manifest.json").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint_sha256", "different"),
        ("n_steps", 100),
        ("seed", 9),
        ("ids_file", "different_ids.txt"),
    ],
)
def test_merge_rejects_condition_metadata_mismatch(tmp_path: Path, field: str, value: object):
    shard_a = _write_shard(tmp_path / "shard-a", ["baseline"])
    shard_b = _write_shard(
        tmp_path / "shard-b",
        ["steric_0p1"],
        metadata={field: value},
    )
    root = tmp_path / "merged"

    with pytest.raises(ValueError, match="metadata mismatch"):
        merge_sweep_shards(root, [shard_a, shard_b])

    assert not (root / "inference").exists()


def test_merge_rejects_target_count_mismatch(tmp_path: Path):
    shard_a = _write_shard(tmp_path / "shard-a", ["baseline"], target_count=2)
    shard_b = _write_shard(tmp_path / "shard-b", ["steric_0p1"], target_count=1)
    root = tmp_path / "merged"

    with pytest.raises(ValueError, match="target_count mismatch"):
        merge_sweep_shards(root, [shard_a, shard_b])

    assert not (root / "inference").exists()


@pytest.mark.parametrize("existing", ["inference", "sweep_manifest.json"])
def test_merge_refuses_to_overwrite_existing_outputs(tmp_path: Path, existing: str):
    shard_a = _write_shard(tmp_path / "shard-a", ["baseline"])
    shard_b = _write_shard(tmp_path / "shard-b", ["steric_0p1"])
    root = tmp_path / "merged"
    root.mkdir()
    output = root / existing
    if existing == "inference":
        output.mkdir()
    else:
        output.write_text("keep")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        merge_sweep_shards(root, [shard_a, shard_b])

    assert output.exists()


def test_copy_failure_leaves_no_partial_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    shard_a = _write_shard(tmp_path / "shard-a", ["baseline"])
    shard_b = _write_shard(tmp_path / "shard-b", ["steric_0p1"])
    root = tmp_path / "merged"
    real_copytree = shutil.copytree
    calls = 0

    def fail_second_copy(source: Path, destination: Path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated copy failure")
        return real_copytree(source, destination)

    monkeypatch.setattr(shutil, "copytree", fail_second_copy)

    with pytest.raises(OSError, match="simulated copy failure"):
        merge_sweep_shards(root, [shard_a, shard_b])

    assert not (root / "inference").exists()
    assert not (root / "sweep_manifest.json").exists()
    assert not list(tmp_path.glob(".merged.merge-*"))


def test_manifest_commit_failure_rolls_back_committed_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    shard_a = _write_shard(tmp_path / "shard-a", ["baseline"])
    shard_b = _write_shard(tmp_path / "shard-b", ["steric_0p1"])
    root = tmp_path / "merged"
    real_rename = Path.rename

    def fail_manifest_rename(source: Path, destination: Path):
        if source.name == "sweep_manifest.json":
            raise OSError("simulated manifest commit failure")
        return real_rename(source, destination)

    monkeypatch.setattr(Path, "rename", fail_manifest_rename)

    with pytest.raises(OSError, match="simulated manifest commit failure"):
        merge_sweep_shards(root, [shard_a, shard_b])

    assert not (root / "inference").exists()
    assert not (root / "sweep_manifest.json").exists()
    assert not list(tmp_path.glob(".merged.merge-*"))


def test_merge_requires_at_least_two_shards(tmp_path: Path):
    shard = _write_shard(tmp_path / "shard", ["baseline"])

    with pytest.raises(ValueError, match="at least two"):
        merge_sweep_shards(tmp_path / "merged", [shard])
