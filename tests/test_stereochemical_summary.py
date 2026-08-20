"""Tests for the paired stereochemical example report."""

import json

from benchmarks.summarize_stereochemical_examples import render_markdown, summarize


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _write_condition(root, condition, offset):
    target = "t0001"
    _write_json(
        root / "inference" / condition / "manifest.json",
        {"rows": [{"pdb_id": target, "L": 42}]},
    )
    _write_json(
        root / "scores" / condition / "local_geometry.json",
        {
            "rows": [
                {
                    "pdb_id": target,
                    "pred": {
                        "bond_mae_A": 0.10 + offset,
                        "bond_p95_A": 0.20 + offset,
                        "clashes_per_1k_atoms": 3.0 + offset,
                        "ca_chirality_wrong_frac": 0.01 + offset,
                        "nonlocal_ca_min_A": 1.5 - offset,
                        "nonlocal_ca_clashes_lt_2A": 2,
                        "nonlocal_ca_clashes_lt_3A": 3,
                        "nonlocal_ca_clashes_lt_3p6A": 4,
                        "nonlocal_ca_penetration_rms_A": 0.5,
                    },
                }
            ]
        },
    )
    _write_json(
        root / "scores" / condition / "openstructure" / "summary.json",
        {
            "rows": [
                {
                    "target": target,
                    "oligo_gdtts": 0.7 + offset,
                    "lddt": 0.8 + offset,
                }
            ]
        },
    )
    _write_json(
        root / "scores" / condition / "openstructure" / f"{target}.json",
        {
            "model_bad_bonds": [1],
            "model_bad_angles": [1, 2],
            "model_clashes": [1, 2, 3],
        },
    )


def test_summary_pairs_conditions_and_renders_structure_links(tmp_path):
    _write_condition(tmp_path, "baseline", 0.0)
    _write_condition(tmp_path, "guided", -0.05)

    result = summarize(tmp_path)
    markdown = render_markdown(result)

    assert result["target_count"] == 1
    assert result["rows"][0]["target"] == "t0001"
    assert result["aggregate"]["bond_p95_A"]["delta_guided_minus_baseline"] < 0
    assert "inference/baseline/t0001_pred.pdb" in markdown
    assert "inference/guided/t0001_pred.cif" in markdown
