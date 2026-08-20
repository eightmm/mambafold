"""Synthetic tests for the strict independent-VDW dose report."""

from __future__ import annotations

import json
import sys

import pytest

from benchmarks.summarize_vdw_channel_dose import (
    CONTROL,
    DEFAULT_CONDITIONS,
    DIAGNOSTIC_VDW_METRIC,
    DIRECT_VDW_METRIC,
    EXPECTED_DOSES,
    TREATMENTS,
    main,
    render_markdown,
    summarize,
)

TARGETS = ("t1036s1", "t1040", "t1096")


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _geometry(condition):
    scale, every = EXPECTED_DOSES[condition]
    return {
        "scale": 0.03,
        "start": 0.65,
        "every_n_steps": 2,
        "covalent_weight": 1.0,
        "all_atom_clash_weight": 0.0,
        "all_atom_clash_every_n_steps": 4,
        "vdw_scale": scale,
        "vdw_start": 0.65,
        "vdw_every_n_steps": every,
        "vdw_overlap_tolerance_A": 1.5,
        "vdw_max_step_A": 0.01,
        "steric_scale": 1.0,
        "steric_segment_weight": 0.0,
    }


def _write_condition(root, condition, targets=TARGETS):
    treatment = condition != CONTROL
    runtime_offset = {
        CONTROL: 0.0,
        TREATMENTS[0]: 1.0,
        TREATMENTS[1]: 1.0,
        TREATMENTS[2]: 2.0,
        TREATMENTS[3]: 3.0,
    }[condition]
    _write_json(
        root / "inference" / condition / "manifest.json",
        {
            "schema_version": 1,
            "condition": condition,
            "checkpoint": "frozen.pt",
            "checkpoint_sha256": "abc123",
            "ids_file": "high_clash3.txt",
            "sampler": "sde",
            "n_steps": 500,
            "seed": 0,
            "sde_tau": 0.01,
            "geometry_guidance": _geometry(condition),
            "rows": [
                {
                    "pdb_id": target,
                    "L": 100 + index,
                    "runtime_s": 10.0 + index + runtime_offset,
                    "peak_vram_gib": 2.5,
                    # Deliberately worsens so a passing result proves this
                    # diagnostic metric is not accidentally used as a gate.
                    DIAGNOSTIC_VDW_METRIC: 30.0 if treatment else 20.0,
                    DIRECT_VDW_METRIC: 7.5 if treatment else 10.0,
                }
                for index, target in enumerate(targets)
            ],
        },
    )

    local_rows = []
    ost_rows = []
    for index, target in enumerate(targets):
        local_rows.append(
            {
                "pdb_id": target,
                "pred": {
                    "n_atoms": 1000,
                    "bond_p95_A": 0.024 if treatment else 0.020,
                    "clashes_per_1k_atoms": 9.0 if treatment else 10.0,
                    "nonlocal_ca_clashes_lt_2A": 0,
                    "nonlocal_ca_clashes_lt_3A": 1,
                    "nonlocal_ca_clashes_lt_3p6A": 2,
                    "nonlocal_ca_segment_clashes_lt_2A": 0,
                    "nonlocal_ca_segment_clashes_lt_2p5A": (1 if treatment else 2),
                    "nonlocal_ca_segment_clashes_lt_3A": 3,
                    "nonlocal_ca_segment_penetration_rms_A": 0.2,
                },
                "gt": {"nonlocal_ca_segment_clashes_lt_2p5A": 0},
            }
        )
        ost_rows.append(
            {
                "target": target,
                "oligo_gdtts": 0.80 - 0.01 * index - (0.001 if treatment else 0.0),
                "lddt": 0.75 - 0.01 * index - (0.001 if treatment else 0.0),
                "tm_score": 0.90 - 0.01 * index - (0.001 if treatment else 0.0),
            }
        )
        _write_json(
            root / "scores" / condition / "openstructure" / f"{target}.json",
            {"model_clashes": list(range(8 if treatment else 10))},
        )
    _write_json(root / "scores" / condition / "local_geometry.json", {"rows": local_rows})
    _write_json(
        root / "scores" / condition / "openstructure" / "summary.json",
        {"rows": ost_rows},
    )


def _fixture(root, targets=TARGETS):
    for condition in DEFAULT_CONDITIONS:
        _write_condition(root, condition, targets)


def test_summary_reports_direct_metric_gates_selection_and_writes(tmp_path, monkeypatch):
    _fixture(tmp_path)

    summary = summarize(tmp_path)
    first = summary["aggregates"][TREATMENTS[0]]
    markdown = render_markdown(summary)

    assert summary["coverage"]["paired_complete_3_of_3"] is True
    assert summary["coverage"]["gt_segment_sanity"] is True
    assert first["means"][DIRECT_VDW_METRIC] == pytest.approx(7.5)
    assert first["means"][DIAGNOSTIC_VDW_METRIC] == pytest.approx(30.0)
    assert first["paired_mean_delta_vs_control"][DIRECT_VDW_METRIC] == pytest.approx(-2.5)
    assert first["relative_reduction_vs_control"][DIRECT_VDW_METRIC] == pytest.approx(0.25)
    assert summary["treatment_gates"][TREATMENTS[0]]["all_pass"] is True
    # The first two treatments tie on runtime, so scale is the second key.
    assert summary["selection"]["selected_condition"] == TREATMENTS[0]
    assert summary["selection"]["independent_confirmation_claim"] is False
    target_delta = summary["rows"][0]["conditions"][TREATMENTS[0]]["delta_vs_control"]
    assert target_delta[DIRECT_VDW_METRIC] == pytest.approx(-2.5)
    assert target_delta[DIAGNOSTIC_VDW_METRIC] == pytest.approx(10.0)
    assert target_delta["ost_model_clashes_per_1k_atoms"] == pytest.approx(-2.0)
    assert "high-clash targets" in markdown
    assert "no independent confirmation" in markdown
    assert "Per-target paired deltas" in markdown

    monkeypatch.setattr(
        sys,
        "argv",
        ["summarize_vdw_channel_dose.py", "--root", str(tmp_path)],
    )
    main()
    written = json.loads((tmp_path / "vdw_dose_comparison.json").read_text())
    assert written["conditions"] == list(DEFAULT_CONDITIONS)
    assert (
        (tmp_path / "vdw_dose_comparison.md")
        .read_text()
        .startswith("# Exploratory independent-VDW channel dose probe")
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("steric_scale", 1.1, "mismatch beyond"),
        ("vdw_start", 0.70, "vdw_start.*exactly"),
        ("all_atom_clash_weight", 0.2, "legacy all_atom_clash_weight"),
        ("vdw_scale", 0.05, "expected independent VDW dose"),
        ("vdw_every_n_steps", 4, "expected independent VDW dose"),
    ],
)
def test_summary_fails_closed_on_config_drift(tmp_path, field, value, match):
    _fixture(tmp_path)
    path = tmp_path / "inference" / TREATMENTS[0] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["geometry_guidance"][field] = value
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=match):
        summarize(tmp_path)


@pytest.mark.parametrize("missing_metric", [DIAGNOSTIC_VDW_METRIC, DIRECT_VDW_METRIC])
def test_summary_fails_closed_on_conditions_coverage_and_direct_metric(tmp_path, missing_metric):
    _fixture(tmp_path)
    with pytest.raises(ValueError, match="conditions must be exactly"):
        summarize(tmp_path, DEFAULT_CONDITIONS[:-1])

    _fixture(tmp_path, TARGETS[:2])
    with pytest.raises(ValueError, match="exactly 3 targets"):
        summarize(tmp_path)

    _fixture(tmp_path)
    path = tmp_path / "inference" / TREATMENTS[0] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["rows"][0].pop(missing_metric)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=missing_metric):
        summarize(tmp_path)


def test_summary_requires_complete_zero_event_gt_sanity(tmp_path):
    _fixture(tmp_path)
    path = tmp_path / "scores" / TREATMENTS[0] / "local_geometry.json"
    local = json.loads(path.read_text())
    for row in local["rows"]:
        row["gt"] = None
    path.write_text(json.dumps(local))
    with pytest.raises(ValueError, match="complete GT"):
        summarize(tmp_path)

    _fixture(tmp_path)
    local = json.loads(path.read_text())
    local["rows"][0]["gt"]["nonlocal_ca_segment_clashes_lt_2p5A"] = 1
    path.write_text(json.dumps(local))
    with pytest.raises(ValueError, match="must be 0"):
        summarize(tmp_path)


def test_failed_direct_efficacy_yields_no_promotion(tmp_path):
    _fixture(tmp_path)
    for condition in TREATMENTS:
        path = tmp_path / "inference" / condition / "manifest.json"
        manifest = json.loads(path.read_text())
        for row in manifest["rows"]:
            row[DIRECT_VDW_METRIC] = 8.5  # only 15% below control
        path.write_text(json.dumps(manifest))

    summary = summarize(tmp_path)
    assert summary["selection"]["selected_condition"] is None
    assert summary["selection"]["status"] == "no_promotion"
    for condition in TREATMENTS:
        gate = summary["treatment_gates"][condition]
        assert gate["checks"]["direct_vdw_loss_mean_reduction"]["pass"] is False
        assert gate["all_pass"] is False


def test_each_target_accuracy_gate_is_independent_of_mean_gate(tmp_path):
    _fixture(tmp_path)
    path = tmp_path / "scores" / TREATMENTS[0] / "openstructure" / "summary.json"
    ost = json.loads(path.read_text())
    control_base = (0.80, 0.79, 0.78)
    deltas = (-0.006, 0.003, 0.003)
    for row, base, delta in zip(ost["rows"], control_base, deltas, strict=True):
        row["oligo_gdtts"] = base + delta
    path.write_text(json.dumps(ost))

    checks = summarize(tmp_path)["treatment_gates"][TREATMENTS[0]]["checks"]
    assert checks["gdt_ts_mean_delta"]["pass"] is True
    assert checks["gdt_ts_each_target_delta"]["value"] == pytest.approx(-0.006)
    assert checks["gdt_ts_each_target_delta"]["pass"] is False


def test_zero_control_direct_metric_is_not_treated_as_reduction(tmp_path):
    _fixture(tmp_path)
    for condition in DEFAULT_CONDITIONS:
        path = tmp_path / "inference" / condition / "manifest.json"
        manifest = json.loads(path.read_text())
        for row in manifest["rows"]:
            row[DIRECT_VDW_METRIC] = 0.0
        path.write_text(json.dumps(manifest))

    gate = summarize(tmp_path)["treatment_gates"][TREATMENTS[0]]["checks"][
        "direct_vdw_loss_mean_reduction"
    ]
    assert gate["value"] is None
    assert gate["pass"] is False
