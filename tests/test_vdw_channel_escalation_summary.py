"""Synthetic tests for the strict held-out independent-VDW escalation report."""

from __future__ import annotations

import json
import sys

import pytest

from benchmarks.summarize_vdw_channel_escalation import (
    CONTROL,
    DEFAULT_CONDITIONS,
    DIAGNOSTIC_VDW_METRIC,
    DIRECT_VDW_METRIC,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_DOSES,
    EXPECTED_DTYPE,
    EXPECTED_GPU,
    FULL_TARGETS,
    HELDOUT_TARGETS,
    SELECTION_TARGETS,
    TREATMENT,
    main,
    render_markdown,
    summarize,
)


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


def _write_condition(root, condition, targets=FULL_TARGETS):
    treatment = condition == TREATMENT
    manifest_rows = []
    local_rows = []
    ost_rows = []
    for index, target in enumerate(targets):
        selected = target in SELECTION_TARGETS
        if treatment and selected:
            # Deliberately bad selection-set behavior proves that it is reported
            # but never leaks into the held-out decision.
            runtime = 30.0 + index
            direct_vdw = 20.0
            diagnostic_vdw = 30.0
            bond = 0.030
            hard_clashes = 20.0
            segment_count = 4
            gdt_delta = -0.100
            lddt_delta = -0.100
            ost_clashes = 20
        elif treatment:
            runtime = 11.0 + index
            direct_vdw = 7.5
            diagnostic_vdw = 30.0
            bond = 0.0208
            hard_clashes = 9.0
            segment_count = 1
            gdt_delta = -0.001
            lddt_delta = -0.001
            ost_clashes = 8
        else:
            runtime = 10.0 + index
            direct_vdw = 10.0
            diagnostic_vdw = 20.0
            bond = 0.020
            hard_clashes = 10.0
            segment_count = 2
            gdt_delta = 0.0
            lddt_delta = 0.0
            ost_clashes = 10

        manifest_rows.append(
            {
                "pdb_id": target,
                "L": 100 + index,
                "runtime_s": runtime,
                "peak_vram_gib": 2.5,
                DIAGNOSTIC_VDW_METRIC: diagnostic_vdw,
                DIRECT_VDW_METRIC: direct_vdw,
            }
        )
        local_rows.append(
            {
                "pdb_id": target,
                "pred": {
                    "n_atoms": 1000,
                    "bond_p95_A": bond,
                    "clashes_per_1k_atoms": hard_clashes,
                    "nonlocal_ca_clashes_lt_2A": 0,
                    "nonlocal_ca_clashes_lt_3A": 1,
                    "nonlocal_ca_clashes_lt_3p6A": 2,
                    "nonlocal_ca_segment_clashes_lt_2A": 0,
                    "nonlocal_ca_segment_clashes_lt_2p5A": segment_count,
                    "nonlocal_ca_segment_clashes_lt_3A": 3,
                    "nonlocal_ca_segment_penetration_rms_A": 0.2,
                },
                "gt": {"nonlocal_ca_segment_clashes_lt_2p5A": 0},
            }
        )
        ost_rows.append(
            {
                "target": target,
                "oligo_gdtts": 0.80 - 0.005 * index + gdt_delta,
                "lddt": 0.75 - 0.005 * index + lddt_delta,
                "tm_score": 0.90 - 0.005 * index,
            }
        )
        _write_json(
            root / "scores" / condition / "openstructure" / f"{target}.json",
            {"model_clashes": list(range(ost_clashes))},
        )

    _write_json(
        root / "inference" / condition / "manifest.json",
        {
            "schema_version": 1,
            "condition": condition,
            "checkpoint": "frozen.pt",
            "checkpoint_staged": True,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "ids_file": "selection3_plus_heldout5.txt",
            "sampler": "sde",
            "n_steps": 500,
            "seed": 0,
            "sde_tau": 0.01,
            "sde_eps": 0.01,
            "sde_w_cutoff": 0.99,
            "sde_log_timesteps": True,
            "cuda_device_name": EXPECTED_GPU,
            "autocast_dtype": EXPECTED_DTYPE,
            "geometry_guidance": _geometry(condition),
            "rows": manifest_rows,
        },
    )
    _write_json(root / "scores" / condition / "local_geometry.json", {"rows": local_rows})
    _write_json(
        root / "scores" / condition / "openstructure" / "summary.json",
        {"rows": ost_rows},
    )


def _fixture(root, targets=FULL_TARGETS):
    for condition in DEFAULT_CONDITIONS:
        _write_condition(root, condition, targets)


def test_heldout_only_pass_reports_all_splits_and_writes(tmp_path, monkeypatch):
    _fixture(tmp_path)

    summary = summarize(tmp_path)
    heldout = summary["splits"]["heldout"]
    selection = summary["splits"]["selection"]
    gates = summary["heldout_gates"]
    markdown = render_markdown(summary)

    assert summary["coverage"]["paired_complete_8_of_8"] is True
    assert summary["coverage"]["gt_segment_sanity"] is True
    assert selection["target_ids"] == list(SELECTION_TARGETS)
    assert heldout["target_ids"] == list(HELDOUT_TARGETS)
    assert summary["splits"]["full"]["target_ids"] == list(FULL_TARGETS)
    assert selection["aggregates"][TREATMENT]["paired_mean_delta_vs_control"][
        "gdt_ts"
    ] == pytest.approx(-0.1)
    assert heldout["aggregates"][TREATMENT]["relative_reduction_vs_control"][
        DIRECT_VDW_METRIC
    ] == pytest.approx(0.25)
    assert heldout["aggregates"][TREATMENT]["relative_reduction_vs_control"][
        "ost_model_clashes_per_1k_atoms"
    ] == pytest.approx(0.2)
    assert gates["all_pass"] is True
    assert summary["decision"]["gate_scope"] == "heldout_only"
    assert summary["decision"]["selection_metrics_used_for_gate"] is False
    assert summary["decision"]["status"] == "candidate_for_multiseed_confirmation"
    assert "held-out split is the only gate" in markdown
    assert "Selection (3 targets)" in markdown
    assert "Heldout (5 targets)" in markdown
    assert "Full (8 targets)" in markdown

    monkeypatch.setattr(
        sys,
        "argv",
        ["summarize_vdw_channel_escalation.py", "--root", str(tmp_path)],
    )
    main()
    written = json.loads((tmp_path / "vdw_escalation_comparison.json").read_text())
    assert written["decision"]["status"] == "candidate_for_multiseed_confirmation"
    assert (
        (tmp_path / "vdw_escalation_comparison.md")
        .read_text()
        .startswith("# Independent-VDW channel escalation")
    )


def test_failed_heldout_gate_abandons_channel_only_escalation(tmp_path):
    _fixture(tmp_path)
    path = tmp_path / "inference" / TREATMENT / "manifest.json"
    manifest = json.loads(path.read_text())
    for row in manifest["rows"]:
        if row["pdb_id"] in HELDOUT_TARGETS:
            row[DIRECT_VDW_METRIC] = 8.5  # only 15% below held-out control
    path.write_text(json.dumps(manifest))

    summary = summarize(tmp_path)
    check = summary["heldout_gates"]["checks"]["direct_vdw_loss_mean_reduction"]
    assert check["value"] == pytest.approx(0.15)
    assert check["pass"] is False
    assert summary["heldout_gates"]["all_pass"] is False
    assert summary["decision"]["status"] == "abandon_channel_only_escalation"


def test_summary_fails_closed_on_exact_split_and_condition_set(tmp_path):
    _fixture(tmp_path)
    with pytest.raises(ValueError, match="conditions must be exactly"):
        summarize(tmp_path, (CONTROL,))

    _fixture(tmp_path, FULL_TARGETS[:-1])
    with pytest.raises(ValueError, match="target split must be exact"):
        summarize(tmp_path)


@pytest.mark.parametrize(
    ("location", "field", "value", "match"),
    [
        ("manifest", "checkpoint_sha256", "wrong", "checkpoint_sha256 must be exactly"),
        ("manifest", "cuda_device_name", "NVIDIA A5000", "cuda_device_name must be exactly"),
        ("manifest", "n_steps", 499, "n_steps must be exactly"),
        ("guidance", "all_atom_clash_weight", 0.2, "all_atom_clash_weight.*exactly"),
        ("guidance", "vdw_scale", 0.2, "expected independent VDW dose"),
        ("guidance", "vdw_every_n_steps", 2, "expected independent VDW dose"),
        ("guidance", "vdw_max_step_A", 0.02, "vdw_max_step_A.*exactly"),
    ],
)
def test_summary_fails_closed_on_frozen_config_drift(tmp_path, location, field, value, match):
    _fixture(tmp_path)
    path = tmp_path / "inference" / TREATMENT / "manifest.json"
    manifest = json.loads(path.read_text())
    target = manifest if location == "manifest" else manifest["geometry_guidance"]
    target[field] = value
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=match):
        summarize(tmp_path)


def test_summary_rejects_nonvaried_config_mismatch(tmp_path):
    _fixture(tmp_path)
    path = tmp_path / "inference" / TREATMENT / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["geometry_guidance"]["steric_scale"] = 1.1
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="mismatch beyond"):
        summarize(tmp_path)


def test_summary_fails_closed_on_coverage_gt_and_direct_metrics(tmp_path):
    _fixture(tmp_path)
    missing_raw = tmp_path / "scores" / TREATMENT / "openstructure" / f"{HELDOUT_TARGETS[0]}.json"
    missing_raw.unlink()
    with pytest.raises(ValueError, match="missing required artifact"):
        summarize(tmp_path)

    _fixture(tmp_path)
    local_path = tmp_path / "scores" / TREATMENT / "local_geometry.json"
    local = json.loads(local_path.read_text())
    local["rows"][0]["gt"] = None
    local_path.write_text(json.dumps(local))
    with pytest.raises(ValueError, match="GT coverage is incomplete"):
        summarize(tmp_path)

    _fixture(tmp_path)
    manifest_path = tmp_path / "inference" / TREATMENT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["rows"][0].pop(DIAGNOSTIC_VDW_METRIC)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=DIAGNOSTIC_VDW_METRIC):
        summarize(tmp_path)

    _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["rows"][0].pop(DIRECT_VDW_METRIC)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=DIRECT_VDW_METRIC):
        summarize(tmp_path)


def test_each_target_and_runtime_gates_are_heldout_scoped(tmp_path):
    _fixture(tmp_path)
    ost_path = tmp_path / "scores" / TREATMENT / "openstructure" / "summary.json"
    ost = json.loads(ost_path.read_text())
    target = next(row for row in ost["rows"] if row["target"] == HELDOUT_TARGETS[0])
    target["oligo_gdtts"] -= 0.006
    ost_path.write_text(json.dumps(ost))

    summary = summarize(tmp_path)
    check = summary["heldout_gates"]["checks"]["gdt_ts_each_target_delta"]
    assert check["value"] == pytest.approx(-0.007)
    assert check["pass"] is False
    assert summary["decision"]["status"] == "abandon_channel_only_escalation"


def test_zero_control_primary_metric_fails_reduction_gate(tmp_path):
    _fixture(tmp_path)
    for condition in DEFAULT_CONDITIONS:
        path = tmp_path / "inference" / condition / "manifest.json"
        manifest = json.loads(path.read_text())
        for row in manifest["rows"]:
            if row["pdb_id"] in HELDOUT_TARGETS:
                row[DIRECT_VDW_METRIC] = 0.0
        path.write_text(json.dumps(manifest))

    check = summarize(tmp_path)["heldout_gates"]["checks"]["direct_vdw_loss_mean_reduction"]
    assert check["value"] is None
    assert check["pass"] is False
