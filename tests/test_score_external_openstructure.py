"""Focused tests for the active external OpenStructure scorer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.score_external_openstructure import (
    COMPARE_STRUCTURE_ARGS,
    MODEL_NAMES,
    Reference,
    apply_target_filter,
    load_references,
    read_target_filter,
    summarize,
    valid_cached_result,
)

ACTIVE_MODELS = (
    "mambafold_esmc6b_step170000",
    "simplefold_360m",
    "esmfold_v1",
    "dplm2_bit_650m",
)


def _write_casp14_mapping(evaluation_root: Path) -> None:
    inputs = evaluation_root / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "manifest.json").write_text(
        json.dumps(
            {
                "mapping": [
                    {
                        "dataset": "casp14_70",
                        "fasta_id": "t1001",
                        "canonical_id": "seq_first",
                        "length": 101,
                    },
                    {
                        "dataset": "casp14_70",
                        "fasta_id": "t1002",
                        "canonical_id": "seq_second",
                        "length": 202,
                    },
                ]
            }
        )
        + "\n"
    )


def test_active_model_roster_excludes_legacy_and_oom_comparators() -> None:
    assert MODEL_NAMES == ACTIVE_MODELS
    assert all("esm3" not in model.lower() for model in MODEL_NAMES)
    assert all("omega" not in model.lower() for model in MODEL_NAMES)


def test_empty_score_summary_records_failure_without_statistics_error() -> None:
    assert summarize([]) == {
        "n": 0,
        "mean": None,
        "median": None,
        "min": None,
        "max": None,
    }


def test_cached_score_is_bound_to_prediction_and_reference_hashes(tmp_path: Path) -> None:
    result_path = tmp_path / "pair.json"
    identity_path = tmp_path / "pair.inputs.json"
    result = {
        "status": "SUCCESS",
        **{
            metric: 0.5
            for metric in (
                "oligo_gdtts",
                "oligo_gdtha",
                "tm_score",
                "lddt",
                "bb_lddt",
                "rmsd",
            )
        },
    }
    expected_identity = {
        "schema_version": 1,
        "dataset": "casp16",
        "model": "mambafold_esmc6b_step170000",
        "pair_id": "t1208s1",
        "canonical_id": "seq_000001",
        "prediction_sha256": "new-prediction-hash",
        "reference_sha256": "reference-hash",
        "openstructure_version": "OpenStructure 2.9.1",
        "compare_structure_args": list(COMPARE_STRUCTURE_ARGS),
    }
    result_path.write_text(json.dumps(result) + "\n")
    identity_path.write_text(
        json.dumps({**expected_identity, "prediction_sha256": "stale-prediction-hash"}) + "\n"
    )

    assert valid_cached_result(result_path, identity_path, expected_identity) is None

    identity_path.write_text(json.dumps(expected_identity) + "\n")
    assert valid_cached_result(result_path, identity_path, expected_identity) == result

    identity_path.write_text(
        json.dumps({**expected_identity, "openstructure_version": "OpenStructure 2.8.0"}) + "\n"
    )
    assert valid_cached_result(result_path, identity_path, expected_identity) is None


def test_casp14_references_use_explicit_model_independent_directory(tmp_path: Path) -> None:
    evaluation_root = tmp_path / "evaluation"
    _write_casp14_mapping(evaluation_root)
    reference_dir = tmp_path / "official_references"
    reference_dir.mkdir()
    for target in ("t1001", "t1002"):
        (reference_dir / f"{target}_gt.pdb").write_text(f"REMARK {target}\n")

    references = load_references(
        tmp_path,
        evaluation_root,
        "casp14",
        casp14_reference_dir=reference_dir,
    )

    assert [reference.target_id for reference in references] == ["t1001", "t1002"]
    assert [reference.canonical_id for reference in references] == ["seq_first", "seq_second"]
    assert [reference.path.parent for reference in references] == [reference_dir, reference_dir]


def test_casp14_default_reference_directory_is_stable_and_model_independent(
    tmp_path: Path,
) -> None:
    evaluation_root = tmp_path / "evaluation"
    _write_casp14_mapping(evaluation_root)
    reference_dir = evaluation_root / "references/casp14_full70"
    reference_dir.mkdir(parents=True)
    for target in ("t1001", "t1002"):
        (reference_dir / f"{target}_gt.pdb").write_text(f"REMARK {target}\n")

    references = load_references(tmp_path, evaluation_root, "casp14")

    assert {reference.path.parent for reference in references} == {reference_dir}


def test_casp14_missing_reference_directory_explains_migration(tmp_path: Path) -> None:
    evaluation_root = tmp_path / "evaluation"
    _write_casp14_mapping(evaluation_root)

    with pytest.raises(SystemExit, match="--casp14-reference-dir"):
        load_references(tmp_path, evaluation_root, "casp14")


def test_target_filter_is_normalized_and_content_addressed(tmp_path: Path) -> None:
    path = tmp_path / "casp16_clean.txt"
    path.write_text("T1208S1\n# exact match removed below\nt1212  # retained\n")

    identifiers, metadata = read_target_filter(path)

    assert identifiers == {"t1208s1", "t1212"}
    assert metadata["filename"] == "casp16_clean.txt"
    assert metadata["target_ids"] == ["t1208s1", "t1212"]
    assert len(metadata["sha256"]) == 64


def test_target_filter_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.txt"
    path.write_text("T1208S1\nt1208s1\n")

    with pytest.raises(SystemExit, match="duplicate IDs"):
        read_target_filter(path)


def test_target_filter_selects_complete_targets_case_insensitively(tmp_path: Path) -> None:
    path = tmp_path / "admitted.txt"
    path.write_text("T1208S1\n")
    references = [
        Reference("t1208s1", "t1208s1", "seq_1", tmp_path / "a.pdb", 100, "whole"),
        Reference("t1212", "t1212", "seq_2", tmp_path / "b.pdb", 200, "whole"),
    ]

    selected, metadata = apply_target_filter(references, path, "casp16")

    assert selected == references[:1]
    assert metadata["target_ids"] == ["t1208s1"]


def test_target_filter_rejects_unknown_dataset_ids(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("not-a-casp-target\n")

    with pytest.raises(SystemExit, match="absent from casp16"):
        apply_target_filter([], path, "casp16")
