"""Synthetic tests for the full CASP14 physics confirmation report."""

from __future__ import annotations

import json
import sys

import pytest

from benchmarks.summarize_physics_full import main, render_markdown, summarize
from benchmarks.summarize_physics_guidance import CONTROL

TREATMENT = "steric_1_vdw_segment"
TARGETS = ("t1001", "t1002", "t1003", "t1004")


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _write_condition(root, condition):
    treatment = condition == TREATMENT
    geometry = {
        "scale": 0.03,
        "steric_scale": 1.0,
        "all_atom_clash_weight": 0.2 if treatment else 0.0,
        "steric_segment_weight": 0.7 if treatment else 0.0,
    }
    runtime_offset = 1.0 if treatment else 0.0
    _write_json(
        root / "inference" / condition / "manifest.json",
        {
            "schema_version": 1,
            "condition": condition,
            "checkpoint": "frozen.pt",
            "checkpoint_sha256": "abc123",
            "sampler": "sde",
            "n_steps": 500,
            "seed": 0,
            "geometry_guidance": geometry,
            "rows": [
                {
                    "pdb_id": target,
                    "L": 100 + index,
                    "runtime_s": 10.0 + index + runtime_offset,
                    "peak_vram_gib": 2.5,
                }
                for index, target in enumerate(TARGETS)
            ],
        },
    )

    gdt_deltas = (-0.003, -0.002, 0.0, 0.001)
    lddt_deltas = (-0.004, -0.001, 0.001, 0.002)
    control_segments = (4, 4, 2, 0)
    treatment_segments = (2, 2, 1, 0)
    local_rows = []
    ost_rows = []
    for index, target in enumerate(TARGETS):
        segment_count = treatment_segments[index] if treatment else control_segments[index]
        local_rows.append(
            {
                "pdb_id": target,
                "pred": {
                    "n_atoms": 1000,
                    "bond_p95_A": 0.021 if treatment else 0.020,
                    "clashes_per_1k_atoms": 9.0 if treatment else 10.0,
                    "nonlocal_ca_clashes_lt_2A": 0,
                    "nonlocal_ca_clashes_lt_3A": 1 if treatment else 2,
                    "nonlocal_ca_clashes_lt_3p6A": 2 if treatment else 4,
                    "nonlocal_ca_segment_clashes_lt_2A": segment_count // 2,
                    "nonlocal_ca_segment_clashes_lt_2p5A": segment_count,
                    "nonlocal_ca_segment_clashes_lt_3A": segment_count + 1,
                    "nonlocal_ca_segment_penetration_rms_A": (0.2 if treatment else 0.4),
                },
                "gt": {"nonlocal_ca_segment_clashes_lt_2p5A": 0},
            }
        )
        ost_rows.append(
            {
                "target": target,
                "oligo_gdtts": 0.80 - 0.01 * index + (gdt_deltas[index] if treatment else 0.0),
                "lddt": 0.75 - 0.01 * index + (lddt_deltas[index] if treatment else 0.0),
                "tm_score": 0.90 - 0.01 * index + (-0.001 if treatment else 0.0),
            }
        )
        _write_json(
            root / "scores" / condition / "openstructure" / f"{target}.json",
            {"model_clashes": list(range(9 if treatment else 10))},
        )

    _write_json(root / "scores" / condition / "local_geometry.json", {"rows": local_rows})
    _write_json(
        root / "scores" / condition / "openstructure" / "summary.json",
        {"rows": ost_rows},
    )


def _fixture(root):
    _write_condition(root, CONTROL)
    _write_condition(root, TREATMENT)
    selection = root / "selection.txt"
    selection.write_text("# selected during smoke\nt1001\n")
    return selection


def test_full_summary_partitions_bootstraps_and_writes(tmp_path, monkeypatch):
    selection = _fixture(tmp_path)

    summary = summarize(
        tmp_path,
        selection,
        [CONTROL, TREATMENT],
        bootstrap=400,
        seed=17,
    )
    repeated = summarize(
        tmp_path,
        selection,
        [CONTROL, TREATMENT],
        bootstrap=400,
        seed=17,
    )

    assert summary["subsets"]["selection"]["target_ids"] == ["t1001"]
    assert summary["subsets"]["heldout"]["target_ids"] == [
        "t1002",
        "t1003",
        "t1004",
    ]
    assert summary["subsets"]["full"]["n"] == 4
    heldout = summary["subsets"]["heldout"]
    treatment = heldout["conditions"][TREATMENT]
    assert treatment["paired_mean_delta_vs_control"]["gdt_ts"] == pytest.approx(
        (-0.002 + 0.0 + 0.001) / 3
    )
    assert (
        treatment["pooled_nonlocal_ca_segment_counts"]["nonlocal_ca_segment_clashes_lt_2p5A"] == 3
    )
    assert treatment["runtime"]["paired_mean_delta_s_vs_control"] == 1.0
    assert (
        heldout["paired_bootstrap_95_ci"]
        == repeated["subsets"]["heldout"]["paired_bootstrap_95_ci"]
    )
    segment_ci = heldout["paired_bootstrap_95_ci"]["segment_lt_2p5_pooled_reduction"]
    assert segment_ci["estimate"] == pytest.approx(0.5)
    assert 0 < segment_ci["n_defined"] <= 400
    assert summary["primary_gates"]["all_pass"] is True
    markdown = render_markdown(summary)
    assert "Primary held-out safety gates" in markdown
    assert "selection (n=1)" in markdown
    assert "post-CASP14" in markdown
    assert "Independent efficacy is not claimed" in markdown
    assert len(summary["interpretation_limits"]) == 3

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_physics_full.py",
            "--root",
            str(tmp_path),
            "--selection-ids",
            str(selection),
            "--conditions",
            CONTROL,
            TREATMENT,
            "--bootstrap",
            "100",
            "--seed",
            "7",
        ],
    )
    main()
    written = json.loads((tmp_path / "full_confirmation.json").read_text())
    assert written["bootstrap"]["n_resamples"] == 100
    assert (
        (tmp_path / "full_confirmation.md")
        .read_text()
        .startswith("# CASP14 full physics-guidance safety check")
    )


@pytest.mark.parametrize(
    ("contents", "match"),
    [
        ("", "nonempty"),
        ("t1001\nt1001\n", "duplicates"),
        ("t9999\n", "not present"),
        ("\n".join(TARGETS) + "\n", "strict subset"),
    ],
)
def test_full_summary_rejects_invalid_selection_ids(tmp_path, contents, match):
    selection = _fixture(tmp_path)
    selection.write_text(contents)
    with pytest.raises(ValueError, match=match):
        summarize(
            tmp_path,
            selection,
            [CONTROL, TREATMENT],
            bootstrap=20,
            seed=0,
        )


def test_full_summary_rejects_conditions_and_source_mismatch(tmp_path):
    selection = _fixture(tmp_path)
    for conditions in (
        [CONTROL],
        [TREATMENT, CONTROL],
        [CONTROL, TREATMENT, "extra"],
    ):
        with pytest.raises(ValueError, match="exactly"):
            summarize(
                tmp_path,
                selection,
                conditions,
                bootstrap=20,
                seed=0,
            )

    manifest_path = tmp_path / "inference" / TREATMENT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["n_steps"] = 250
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="mismatch beyond"):
        summarize(
            tmp_path,
            selection,
            [CONTROL, TREATMENT],
            bootstrap=20,
            seed=0,
        )


def test_full_summary_reports_failed_heldout_gates(tmp_path):
    selection = _fixture(tmp_path)
    summary_path = tmp_path / "scores" / TREATMENT / "openstructure" / "summary.json"
    ost = json.loads(summary_path.read_text())
    ost["rows"][1]["oligo_gdtts"] -= 0.10
    summary_path.write_text(json.dumps(ost))

    summary = summarize(
        tmp_path,
        selection,
        [CONTROL, TREATMENT],
        bootstrap=100,
        seed=3,
    )
    gates = summary["primary_gates"]
    assert gates["all_pass"] is False
    assert gates["checks"]["gdt_ts_mean_delta"]["pass"] is False
    assert gates["checks"]["gdt_ts_worst_target_delta"]["pass"] is False


def test_full_summary_marks_zero_event_heldout_efficacy_not_testable(tmp_path):
    selection = _fixture(tmp_path)
    for condition in (CONTROL, TREATMENT):
        path = tmp_path / "scores" / condition / "local_geometry.json"
        local = json.loads(path.read_text())
        for row in local["rows"]:
            if row["pdb_id"] != "t1001":
                row["pred"]["nonlocal_ca_segment_clashes_lt_2p5A"] = 0
        path.write_text(json.dumps(local))

    summary = summarize(
        tmp_path,
        selection,
        [CONTROL, TREATMENT],
        bootstrap=100,
        seed=5,
    )
    context = summary["segment_efficacy_context"]

    assert context["subsets"]["heldout"]["testable"] is False
    assert context["subsets"]["heldout"]["status"] == "not_testable_no_control_events"
    assert context["independent_efficacy_confirmed"] is False
    assert summary["primary_gates"]["claim"] == "safety_and_nonregression"
