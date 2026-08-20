"""Focused test for the multi-condition self-overlap report."""

import json

from benchmarks.summarize_self_avoidance_sweep import render_markdown, summarize


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _condition(root, name, *, steric_scale, ca_clashes, gdt=0.7):
    target = "t0001"
    _write(
        root / "inference" / name / "manifest.json",
        {
            "geometry_guidance": {"steric_scale": steric_scale},
            "rows": [
                {
                    "pdb_id": target,
                    "L": 42,
                    "runtime_s": 3.0,
                    "peak_vram_gib": 2.0,
                }
            ],
        },
    )
    _write(
        root / "scores" / name / "local_geometry.json",
        {
            "rows": [
                {
                    "pdb_id": target,
                    "pred": {
                        "bond_mae_A": 0.01,
                        "bond_p95_A": 0.02,
                        "clashes_per_1k_atoms": 2.0,
                        "ca_chirality_wrong_frac": 0.0,
                        "nonlocal_ca_min_A": 1.0 if ca_clashes else 3.2,
                        "nonlocal_ca_clashes_lt_2A": ca_clashes,
                        "nonlocal_ca_clashes_lt_3A": ca_clashes,
                        "nonlocal_ca_clashes_lt_3p6A": ca_clashes,
                        "nonlocal_ca_penetration_rms_A": 1.0 if ca_clashes else 0.0,
                    },
                }
            ]
        },
    )
    _write(
        root / "scores" / name / "openstructure" / "summary.json",
        {"rows": [{"target": target, "oligo_gdtts": gdt, "lddt": 0.8}]},
    )
    _write(
        root / "scores" / name / "openstructure" / f"{target}.json",
        {"model_bad_bonds": [], "model_bad_angles": [], "model_clashes": [1]},
    )


def test_summary_selects_guardrailed_condition_with_fewer_ca_overlaps(tmp_path):
    _write(
        tmp_path / "sweep_manifest.json",
        {"conditions": ["baseline", "split_local_control", "steric_0p1"]},
    )
    _condition(tmp_path, "baseline", steric_scale=0.0, ca_clashes=3)
    _condition(tmp_path, "split_local_control", steric_scale=0.0, ca_clashes=4)
    _condition(tmp_path, "steric_0p1", steric_scale=0.1, ca_clashes=0)

    result = summarize(tmp_path)
    markdown = render_markdown(result)

    assert result["best_guardrailed_condition"] == "steric_0p1"
    assert result["best_preregistered_success_condition"] == "steric_0p1"
    assert result["preregistered_success"] is True
    assert result["aggregates"]["steric_0p1"]["nonlocal_ca_clashes_lt_3A_total"] == 0
    assert (
        result["aggregates"]["steric_0p1"]["nonlocal_ca_clashes_lt_3p6A_reduction_fraction"] == 1.0
    )
    assert "inference/steric_0p1/t0001_pred.pdb" in markdown
