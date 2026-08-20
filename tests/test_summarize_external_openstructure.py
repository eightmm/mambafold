"""Focused tests for the active external comparison report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.summarize_external_openstructure import DATASETS, METRICS, MODELS, build_report


def _write_summary(path: Path, *, target_ids: list[str], value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"target_id": target_id, **{metric: value + index / 100 for metric in METRICS}}
        for index, target_id in enumerate(target_ids)
    ]
    path.write_text(
        json.dumps(
            {
                "evaluation_complete": True,
                "evaluation_complete_for_available_predictions": True,
                "expected_target_count": 2,
                "aggregation": "unweighted target mean",
                "target_rows": rows,
            }
        )
        + "\n"
    )


def _fixture(score_root: Path) -> None:
    for dataset_index, (dataset, _) in enumerate(DATASETS):
        for model_index, (model, _) in enumerate(MODELS):
            _write_summary(
                score_root / dataset / model / "summary.json",
                target_ids=[f"{dataset}_1", f"{dataset}_2"],
                value=0.5 + dataset_index / 10 + model_index / 100,
            )


def test_report_uses_one_full_four_model_comparison(tmp_path: Path) -> None:
    _fixture(tmp_path)

    report, markdown = build_report(tmp_path)

    assert report["schema_version"] == 2
    assert markdown.count("### Full four-model comparison (N=2)") == len(DATASETS)
    assert markdown.count("| Model | N |") == len(DATASETS)
    rendered = json.dumps(report).lower() + markdown.lower()
    assert "esm3" not in rendered
    assert "omega" not in rendered
    for dataset, _ in DATASETS:
        comparison = report["datasets"][dataset]["full_comparison"]
        assert set(report["datasets"][dataset]) == {
            "label",
            "aggregation",
            "full_comparison",
        }
        assert comparison["target_ids"] == [f"{dataset}_1", f"{dataset}_2"]
        assert [row["model"] for row in comparison["rows"]] == [model for model, _ in MODELS]
        assert all(row["n"] == 2 for row in comparison["rows"])
        dplm_row = next(row for row in comparison["rows"] if row["model"] == "dplm2_bit_650m")
        assert dplm_row["lddt"] is None
    assert markdown.count("| DPLM-2 Bit 650M | 2 |") == len(DATASETS)
    assert markdown.count("| NA |") == len(DATASETS)


def test_report_fails_closed_when_an_active_model_is_incomplete(tmp_path: Path) -> None:
    _fixture(tmp_path)
    dataset = DATASETS[0][0]
    model = MODELS[-1][0]
    _write_summary(
        tmp_path / dataset / model / "summary.json",
        target_ids=[f"{dataset}_1"],
        value=0.5,
    )

    with pytest.raises(SystemExit, match="scored 1 targets, expected 2"):
        build_report(tmp_path)


def test_report_fails_closed_on_target_identity_mismatch(tmp_path: Path) -> None:
    _fixture(tmp_path)
    dataset = DATASETS[1][0]
    model = MODELS[1][0]
    _write_summary(
        tmp_path / dataset / model / "summary.json",
        target_ids=[f"{dataset}_1", "unexpected_target"],
        value=0.5,
    )

    with pytest.raises(SystemExit, match="target set mismatch"):
        build_report(tmp_path)
