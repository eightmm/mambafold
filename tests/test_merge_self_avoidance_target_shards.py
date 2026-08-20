"""Tests for atomic self-avoidance target-shard merging."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from benchmarks.merge_self_avoidance_target_shards import main, merge_target_shards

_CONDITIONS = ("split_local_control", "steric_1")
_PHYSICS_CONDITIONS = (
    "steric_1",
    "steric_1_vdw",
    "steric_1_segment",
    "steric_1_vdw_segment",
)
_FILE_SUFFIXES = (
    "gt.pdb",
    "gt.cif",
    "pred.pdb",
    "pred.cif",
    "pred_seed0.pdb",
    "pred_seed0.cif",
)


def _metadata(condition: str) -> dict[str, object]:
    return {
        "checkpoint_sha256": "abc123",
        "n_steps": 500,
        "seed": 0,
        "sampler": "sde",
        "sde_tau": 0.01,
        "sde_eps": 0.01,
        "sde_w_cutoff": 0.99,
        "sde_log_timesteps": True,
        "cuda_device_name": "NVIDIA RTX 6000 Ada Generation",
        "autocast_dtype": "bfloat16",
        "geometry_guidance": {
            "scale": 0.03,
            "steric_scale": 0.0 if condition == "split_local_control" else 1.0,
        },
    }


def _write_canonical(path: Path, target_ids: list[str]) -> Path:
    path.write_text("\n".join(target_ids) + "\n")
    return path


def _write_shard(
    path: Path,
    target_ids: list[str],
    *,
    conditions: tuple[str, ...] = _CONDITIONS,
    schema_version: int = 1,
    experiment: str = "self_overlap_guidance_v1",
    ids_file: str | None = None,
    metadata_overrides: dict[str, dict[str, object]] | None = None,
    target_overrides: dict[str, list[str]] | None = None,
    ids_file_overrides: dict[str, str] | None = None,
) -> Path:
    ids_file = ids_file or f"{path.name}.ids"
    metadata_overrides = metadata_overrides or {}
    target_overrides = target_overrides or {}
    ids_file_overrides = ids_file_overrides or {}
    condition_paths = []
    for condition in conditions:
        condition_dir = path / "inference" / condition
        condition_dir.mkdir(parents=True)
        condition_targets = target_overrides.get(condition, target_ids)
        rows = [
            {"pdb_id": target, "L": 100 + index, "runtime_s": float(index + 1)}
            for index, target in enumerate(condition_targets)
        ]
        manifest = {
            "schema_version": schema_version,
            "condition": condition,
            "checkpoint": "checkpoint.pt",
            **_metadata(condition),
            **metadata_overrides.get(condition, {}),
            "ids_file": ids_file_overrides.get(condition, ids_file),
            "rows": rows,
        }
        (condition_dir / "manifest.json").write_text(json.dumps(manifest))
        for target in condition_targets:
            for suffix in _FILE_SUFFIXES:
                (condition_dir / f"{target}_{suffix}").write_text(
                    f"{path.name}:{condition}:{target}:{suffix}"
                )
        condition_paths.append(str(Path("inference") / condition / "manifest.json"))
    sweep = {
        "schema_version": schema_version,
        "experiment": experiment,
        "single_process_model_load": True,
        "conditions": list(conditions),
        "target_count": len(target_ids),
        "condition_manifests": condition_paths,
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "sweep_manifest.json").write_text(json.dumps(sweep))
    return path


def _assert_no_partial_output(root: Path) -> None:
    assert not (root / "inference").exists()
    assert not (root / "sweep_manifest.json").exists()
    assert not list(root.parent.glob(f".{root.name}.target-merge-*"))


def test_merge_target_shards_orders_rows_and_files_by_canonical_ids(tmp_path: Path):
    canonical = _write_canonical(tmp_path / "canonical.txt", ["t1", "t2", "t3"])
    shard_a = _write_shard(tmp_path / "shard-a", ["t3", "t1"])
    shard_b = _write_shard(tmp_path / "shard-b", ["t2"])
    root = tmp_path / "merged"

    result = merge_target_shards(root, canonical, [shard_a, shard_b])

    assert result["conditions"] == list(_CONDITIONS)
    assert result["target_count"] == 3
    assert result["ids_file"] == str(canonical)
    assert result["parallel_target_shard_merge"] is True
    assert result["single_process_model_load"] is False
    assert result["source_shards"] == [str(shard_a), str(shard_b)]
    assert result["source_ids_files"] == ["shard-a.ids", "shard-b.ids"]
    assert [record["target_ids"] for record in result["source_target_shards"]] == [
        ["t3", "t1"],
        ["t2"],
    ]
    assert json.loads((root / "sweep_manifest.json").read_text()) == result

    for condition in _CONDITIONS:
        condition_dir = root / "inference" / condition
        manifest = json.loads((condition_dir / "manifest.json").read_text())
        assert manifest["ids_file"] == str(canonical)
        assert [row["pdb_id"] for row in manifest["rows"]] == ["t1", "t2", "t3"]
        assert manifest["source_ids_files"] == ["shard-a.ids", "shard-b.ids"]
        for target in ("t1", "t2", "t3"):
            for suffix in _FILE_SUFFIXES:
                output = condition_dir / f"{target}_{suffix}"
                assert output.is_file()
                expected_shard = "shard-b" if target == "t2" else "shard-a"
                assert output.read_text().startswith(f"{expected_shard}:{condition}:{target}:")


def test_cli_merges_four_expected_physics_conditions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    canonical = _write_canonical(tmp_path / "canonical.txt", ["t1", "t2"])
    shard_a = _write_shard(tmp_path / "shard-a", ["t2"], conditions=_PHYSICS_CONDITIONS)
    shard_b = _write_shard(tmp_path / "shard-b", ["t1"], conditions=_PHYSICS_CONDITIONS)
    root = tmp_path / "merged"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_self_avoidance_target_shards.py",
            "--root",
            str(root),
            "--canonical-ids",
            str(canonical),
            "--shards",
            str(shard_a),
            str(shard_b),
            "--expected-conditions",
            *_PHYSICS_CONDITIONS,
        ],
    )

    main()

    manifest = json.loads((root / "sweep_manifest.json").read_text())
    assert manifest["conditions"] == list(_PHYSICS_CONDITIONS)
    assert manifest["target_count"] == 2
    assert "4 conditions" in capsys.readouterr().out
    for condition in _PHYSICS_CONDITIONS:
        rows = json.loads((root / "inference" / condition / "manifest.json").read_text())["rows"]
        assert [row["pdb_id"] for row in rows] == ["t1", "t2"]


def test_merge_rejects_custom_expected_condition_order_mismatch(tmp_path: Path):
    canonical = _write_canonical(tmp_path / "canonical.txt", ["t1", "t2"])
    shard_a = _write_shard(tmp_path / "shard-a", ["t1"], conditions=_PHYSICS_CONDITIONS)
    mismatched = (
        "steric_1",
        "steric_1_segment",
        "steric_1_vdw",
        "steric_1_vdw_segment",
    )
    shard_b = _write_shard(tmp_path / "shard-b", ["t2"], conditions=mismatched)
    root = tmp_path / "merged"

    with pytest.raises(ValueError, match="unexpected conditions or order"):
        merge_target_shards(
            root,
            canonical,
            [shard_a, shard_b],
            expected_conditions=_PHYSICS_CONDITIONS,
        )

    _assert_no_partial_output(root)


def test_merge_rejects_condition_order_mismatch(tmp_path: Path):
    canonical = _write_canonical(tmp_path / "canonical.txt", ["t1", "t2"])
    shard_a = _write_shard(tmp_path / "shard-a", ["t1"])
    shard_b = _write_shard(tmp_path / "shard-b", ["t2"], conditions=tuple(reversed(_CONDITIONS)))
    root = tmp_path / "merged"

    with pytest.raises(ValueError, match="unexpected conditions or order"):
        merge_target_shards(root, canonical, [shard_a, shard_b])

    _assert_no_partial_output(root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint_sha256", "different"),
        ("n_steps", 100),
        ("seed", 9),
        ("sampler", "ode"),
        ("sde_tau", 0.02),
        ("sde_eps", 0.02),
        ("sde_w_cutoff", 0.8),
        ("sde_log_timesteps", False),
        ("cuda_device_name", "NVIDIA RTX 2080 Ti"),
        ("autocast_dtype", "float16"),
        ("geometry_guidance", {"scale": 9.0, "steric_scale": 9.0}),
    ],
)
def test_merge_rejects_condition_metadata_mismatch(tmp_path: Path, field: str, value: object):
    canonical = _write_canonical(tmp_path / "canonical.txt", ["t1", "t2"])
    shard_a = _write_shard(tmp_path / "shard-a", ["t1"])
    shard_b = _write_shard(
        tmp_path / "shard-b",
        ["t2"],
        metadata_overrides={"steric_1": {field: value}},
    )
    root = tmp_path / "merged"

    with pytest.raises(ValueError, match="condition metadata mismatch"):
        merge_target_shards(root, canonical, [shard_a, shard_b])

    _assert_no_partial_output(root)


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", 2), ("experiment", "different")],
)
def test_merge_rejects_sweep_contract_mismatch(tmp_path: Path, field: str, value: object):
    canonical = _write_canonical(tmp_path / "canonical.txt", ["t1", "t2"])
    shard_a = _write_shard(tmp_path / "shard-a", ["t1"])
    kwargs = {field: value}
    shard_b = _write_shard(tmp_path / "shard-b", ["t2"], **kwargs)  # type: ignore[arg-type]
    root = tmp_path / "merged"

    with pytest.raises(ValueError, match="sweep schema/experiment mismatch"):
        merge_target_shards(root, canonical, [shard_a, shard_b])

    _assert_no_partial_output(root)


def test_merge_rejects_condition_target_mismatch_within_shard(tmp_path: Path):
    canonical = _write_canonical(tmp_path / "canonical.txt", ["t1", "t2", "t3"])
    shard_a = _write_shard(tmp_path / "shard-a", ["t1"])
    shard_b = _write_shard(
        tmp_path / "shard-b",
        ["t2", "t3"],
        target_overrides={"steric_1": ["t3", "t2"]},
    )
    root = tmp_path / "merged"

    with pytest.raises(ValueError, match="target IDs or order differ within shard"):
        merge_target_shards(root, canonical, [shard_a, shard_b])

    _assert_no_partial_output(root)


def test_merge_rejects_target_overlap_across_shards(tmp_path: Path):
    canonical = _write_canonical(tmp_path / "canonical.txt", ["t1", "t2", "t3"])
    shard_a = _write_shard(tmp_path / "shard-a", ["t1", "t2"])
    shard_b = _write_shard(tmp_path / "shard-b", ["t2", "t3"])
    root = tmp_path / "merged"

    with pytest.raises(ValueError, match="overlap across shards"):
        merge_target_shards(root, canonical, [shard_a, shard_b])

    _assert_no_partial_output(root)


def test_merge_rejects_target_union_mismatch(tmp_path: Path):
    canonical = _write_canonical(tmp_path / "canonical.txt", ["t1", "t2", "t3"])
    shard_a = _write_shard(tmp_path / "shard-a", ["t1"])
    shard_b = _write_shard(tmp_path / "shard-b", ["t2", "t4"])
    root = tmp_path / "merged"

    with pytest.raises(ValueError, match=r"missing=\['t3'\].*extra=\['t4'\]"):
        merge_target_shards(root, canonical, [shard_a, shard_b])

    _assert_no_partial_output(root)


def test_merge_rejects_missing_canonical_target_file(tmp_path: Path):
    canonical = _write_canonical(tmp_path / "canonical.txt", ["t1", "t2"])
    shard_a = _write_shard(tmp_path / "shard-a", ["t1"])
    shard_b = _write_shard(tmp_path / "shard-b", ["t2"])
    (shard_b / "inference" / "steric_1" / "t2_pred_seed0.cif").unlink()
    root = tmp_path / "merged"

    with pytest.raises(ValueError, match="missing canonical target file"):
        merge_target_shards(root, canonical, [shard_a, shard_b])

    _assert_no_partial_output(root)


def test_merge_rejects_condition_ids_file_mismatch_within_shard(tmp_path: Path):
    canonical = _write_canonical(tmp_path / "canonical.txt", ["t1", "t2"])
    shard_a = _write_shard(
        tmp_path / "shard-a",
        ["t1"],
        ids_file_overrides={"steric_1": "different.ids"},
    )
    shard_b = _write_shard(tmp_path / "shard-b", ["t2"])
    root = tmp_path / "merged"

    with pytest.raises(ValueError, match="ids_file differs within shard"):
        merge_target_shards(root, canonical, [shard_a, shard_b])

    _assert_no_partial_output(root)


@pytest.mark.parametrize("existing", ["inference", "sweep_manifest.json"])
def test_merge_refuses_to_overwrite_existing_outputs(tmp_path: Path, existing: str):
    canonical = _write_canonical(tmp_path / "canonical.txt", ["t1", "t2"])
    shard_a = _write_shard(tmp_path / "shard-a", ["t1"])
    shard_b = _write_shard(tmp_path / "shard-b", ["t2"])
    root = tmp_path / "merged"
    root.mkdir()
    output = root / existing
    if existing == "inference":
        output.mkdir()
    else:
        output.write_text("keep")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        merge_target_shards(root, canonical, [shard_a, shard_b])

    assert output.exists()


def test_copy_failure_leaves_no_partial_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    canonical = _write_canonical(tmp_path / "canonical.txt", ["t1", "t2"])
    shard_a = _write_shard(tmp_path / "shard-a", ["t1"])
    shard_b = _write_shard(tmp_path / "shard-b", ["t2"])
    root = tmp_path / "merged"
    real_copy2 = shutil.copy2
    calls = 0

    def fail_second_copy(source: Path, destination: Path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated copy failure")
        return real_copy2(source, destination)

    monkeypatch.setattr(shutil, "copy2", fail_second_copy)

    with pytest.raises(OSError, match="simulated copy failure"):
        merge_target_shards(root, canonical, [shard_a, shard_b])

    _assert_no_partial_output(root)


def test_manifest_commit_failure_rolls_back_committed_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    canonical = _write_canonical(tmp_path / "canonical.txt", ["t1", "t2"])
    shard_a = _write_shard(tmp_path / "shard-a", ["t1"])
    shard_b = _write_shard(tmp_path / "shard-b", ["t2"])
    root = tmp_path / "merged"
    real_rename = Path.rename

    def fail_manifest_rename(source: Path, destination: Path):
        if source.name == "sweep_manifest.json":
            raise OSError("simulated manifest commit failure")
        return real_rename(source, destination)

    monkeypatch.setattr(Path, "rename", fail_manifest_rename)

    with pytest.raises(OSError, match="simulated manifest commit failure"):
        merge_target_shards(root, canonical, [shard_a, shard_b])

    _assert_no_partial_output(root)
    assert not root.exists()


def test_merge_rejects_duplicate_canonical_ids(tmp_path: Path):
    canonical = _write_canonical(tmp_path / "canonical.txt", ["t1", "t1"])
    shard_a = _write_shard(tmp_path / "shard-a", ["t1"])
    shard_b = _write_shard(tmp_path / "shard-b", ["t2"])
    root = tmp_path / "merged"

    with pytest.raises(ValueError, match="duplicate target IDs in canonical"):
        merge_target_shards(root, canonical, [shard_a, shard_b])

    _assert_no_partial_output(root)


def test_merge_requires_at_least_two_shards(tmp_path: Path):
    canonical = _write_canonical(tmp_path / "canonical.txt", ["t1"])
    shard = _write_shard(tmp_path / "shard", ["t1"])

    with pytest.raises(ValueError, match="at least two"):
        merge_target_shards(tmp_path / "merged", canonical, [shard])
