"""Synthetic tests for the generic physics-guidance CASP14 report."""

from __future__ import annotations

import json
import sys

import pytest

from benchmarks.summarize_physics_guidance import (
    CONTROL,
    DEFAULT_CONDITIONS,
    main,
    render_markdown,
    summarize,
)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _weights(condition):
    return {
        "all_atom_clash_weight": 0.2
        if condition in {"steric_1_vdw", "steric_1_vdw_segment"}
        else 0.0,
        "steric_segment_weight": 0.7
        if condition in {"steric_1_segment", "steric_1_vdw_segment"}
        else 0.0,
    }


def _write_condition(root, condition, condition_index):
    targets = ("t1001", "t1002")
    weights = _weights(condition)
    geometry = {
        "scale": 0.03,
        "steric_scale": 1.0,
        "steric_ca_min_dist_A": 3.6,
        **weights,
    }
    _write_json(
        root / "inference" / condition / "manifest.json",
        {
            "schema_version": 1,
            "condition": condition,
            "checkpoint": "frozen.pt",
            "checkpoint_sha256": "abc123",
            "ids_file": "casp14.txt",
            "sampler": "sde",
            "n_steps": 500,
            "seed": 0,
            "sde_tau": 0.01,
            "geometry_guidance": geometry,
            "rows": [
                {
                    "pdb_id": target,
                    "L": 100 + target_index,
                    "runtime_s": 10.0 + condition_index + target_index,
                    "peak_vram_gib": 2.0 + condition_index / 10,
                }
                for target_index, target in enumerate(targets)
            ],
        },
    )

    local_rows = []
    ost_rows = []
    for target_index, target in enumerate(targets):
        improvement = float(condition_index)
        local_rows.append(
            {
                "pdb_id": target,
                "pred": {
                    "n_atoms": 1000 + 100 * target_index,
                    "bond_p95_A": 0.02 + 0.001 * improvement,
                    "clashes_per_1k_atoms": 10.0 - improvement,
                    "nonlocal_ca_clashes_lt_2A": 4 - condition_index,
                    "nonlocal_ca_clashes_lt_3A": 8 - condition_index,
                    "nonlocal_ca_clashes_lt_3p6A": 12 - condition_index,
                    "nonlocal_ca_segment_clashes_lt_2A": 3 - min(condition_index, 3),
                    "nonlocal_ca_segment_clashes_lt_2p5A": 5 - condition_index,
                    "nonlocal_ca_segment_clashes_lt_3A": 7 - condition_index,
                    "nonlocal_ca_segment_penetration_rms_A": 0.4 - 0.1 * improvement,
                },
                "gt": {"nonlocal_ca_segment_clashes_lt_2p5A": 0},
            }
        )
        ost_rows.append(
            {
                "target": target,
                "oligo_gdtts": 0.80 - 0.001 * improvement - 0.01 * target_index,
                "lddt": 0.75 - 0.002 * improvement - 0.01 * target_index,
                "tm_score": 0.90 - 0.001 * improvement,
            }
        )
        _write_json(
            root / "scores" / condition / "openstructure" / f"{target}.json",
            {
                "model_clashes": list(range(10 - condition_index)),
            },
        )
    _write_json(root / "scores" / condition / "local_geometry.json", {"rows": local_rows})
    _write_json(
        root / "scores" / condition / "openstructure" / "summary.json",
        {"rows": ost_rows},
    )


def _fixture(root):
    for index, condition in enumerate(DEFAULT_CONDITIONS):
        _write_condition(root, condition, index)


def test_summary_outputs_generic_paired_metrics(tmp_path, monkeypatch):
    _fixture(tmp_path)

    summary = summarize(tmp_path)
    combined = summary["aggregates"]["steric_1_vdw_segment"]
    markdown = render_markdown(summary)

    assert summary["coverage"]["paired_complete"] is True
    assert summary["target_count"] == 2
    assert combined["means"]["gdt_ts"] == pytest.approx(0.792)
    assert combined["paired_mean_delta_vs_control"]["lddt"] == pytest.approx(-0.006)
    assert combined["pooled_nonlocal_ca_segment_counts"]["nonlocal_ca_segment_clashes_lt_2p5A"] == 4
    assert combined["runtime"]["total_s"] == 27.0
    assert combined["runtime"]["delta_total_s_vs_control"] == 6.0
    assert combined["vram"]["max_peak_gib"] == pytest.approx(2.3)
    assert summary["worst_target_regressions"]["steric_1_vdw_segment"]["gdt_ts"][0][
        "delta"
    ] == pytest.approx(-0.003)
    # OpenStructure rates use local n_atoms: mean(10/1000, 10/1100)*1000.
    assert summary["aggregates"][CONTROL]["means"][
        "ost_model_clashes_per_1k_atoms"
    ] == pytest.approx((10.0 + 10000.0 / 1100.0) / 2.0)
    assert "Segment <2.5 Å" in markdown
    assert "Worst per-target accuracy deltas" in markdown

    monkeypatch.setattr(
        sys,
        "argv",
        ["summarize_physics_guidance.py", "--root", str(tmp_path)],
    )
    main()
    written = json.loads((tmp_path / "physics_comparison.json").read_text())
    assert written["conditions"] == list(DEFAULT_CONDITIONS)
    assert (
        (tmp_path / "physics_comparison.md")
        .read_text()
        .startswith("# CASP14 physics-guidance comparison")
    )


def test_summary_rejects_target_and_unintended_config_mismatch(tmp_path):
    _fixture(tmp_path)
    manifest_path = tmp_path / "inference" / "steric_1_vdw" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["rows"].pop()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="target mismatch"):
        summarize(tmp_path)

    _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["geometry_guidance"]["steric_scale"] = 1.1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="mismatch beyond"):
        summarize(tmp_path)


def test_summary_rejects_nonfinite_and_nonzero_gt_segment_counts(tmp_path):
    _fixture(tmp_path)
    local_path = tmp_path / "scores" / CONTROL / "local_geometry.json"
    local = json.loads(local_path.read_text())
    local["rows"][0]["pred"]["bond_p95_A"] = float("nan")
    local_path.write_text(json.dumps(local))
    with pytest.raises(ValueError, match="finite"):
        summarize(tmp_path)

    _fixture(tmp_path)
    local = json.loads(local_path.read_text())
    local["rows"][0]["gt"]["nonlocal_ca_segment_clashes_lt_2p5A"] = 1
    local_path.write_text(json.dumps(local))
    with pytest.raises(ValueError, match="must be 0"):
        summarize(tmp_path)


def test_summary_rejects_noop_treatment(tmp_path):
    _fixture(tmp_path)
    manifest_path = tmp_path / "inference" / "steric_1_segment" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["geometry_guidance"]["steric_segment_weight"] = 0.0
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="must enable"):
        summarize(tmp_path)
