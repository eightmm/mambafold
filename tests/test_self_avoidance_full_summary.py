"""Tests for the confirmatory full-CASP14 self-avoidance report."""

import json

import pytest

from benchmarks.summarize_self_avoidance_full import (
    CONTROL,
    GUIDED,
    read_tuning_ids,
    render_markdown,
    summarize,
)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _write_condition(root, condition, *, guided):
    targets = [f"t{index:04d}" for index in range(70)]
    geometry = {
        "scale": 0.03,
        "start": 0.65,
        "every_n_steps": 2,
        "steric_scale": 1.0 if guided else 0.0,
        "steric_ca_min_dist_A": 3.6,
        "steric_ca_seq_sep": 12,
    }
    _write_json(
        root / "inference" / condition / "manifest.json",
        {
            "condition": condition,
            "checkpoint": "checkpoint.pt",
            "checkpoint_sha256": "abc123",
            "sampler": "sde",
            "n_steps": 500,
            "seed": 0,
            "sde_tau": 0.01,
            "sde_eps": 0.01,
            "sde_w_cutoff": 0.99,
            "sde_log_timesteps": True,
            "geometry_guidance": geometry,
            "rows": [
                {
                    "pdb_id": target,
                    "L": 100 + index,
                    "runtime_s": 11.0 if guided else 10.0,
                    "peak_vram_gib": 2.1 if guided else 2.0,
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
                    "bond_mae_A": 0.01,
                    "bond_p95_A": 0.021 if guided else 0.02,
                    "clashes_per_1k_atoms": 9.0 if guided else 10.0,
                    "n_ca_chiral_centres": 100,
                    "ca_chirality_wrong_frac": 0.0,
                    "nonlocal_ca_min_A": 3.1 if guided else 1.0,
                    "nonlocal_ca_clashes_lt_2A": 0 if guided else 1,
                    "nonlocal_ca_clashes_lt_3A": 0 if guided else 2,
                    "nonlocal_ca_clashes_lt_3p6A": 1 if guided else 4,
                    "nonlocal_ca_penetration_rms_A": 0.1 if guided else 0.4,
                },
            }
        )
        ost_rows.append(
            {
                "target": target,
                "oligo_gdtts": 0.701 if guided else 0.7,
                "lddt": 0.801 if guided else 0.8,
            }
        )
        raw = {
            "status": "SUCCESS",
            "model_bad_bonds": list(range(2)),
            "model_bad_angles": list(range(2)),
            "model_clashes": list(range(9 if guided else 10)),
        }
        if index % 2 == 0:
            raw["n_atoms"] = 1000
        _write_json(root / "scores" / condition / "openstructure" / f"{target}.json", raw)

    _write_json(root / "scores" / condition / "local_geometry.json", {"rows": local_rows})
    _write_json(
        root / "scores" / condition / "openstructure" / "summary.json",
        {"rows": ost_rows},
    )


def _fixture(root):
    _write_condition(root, CONTROL, guided=False)
    _write_condition(root, GUIDED, guided=True)
    return ["t0068", "t0069"]


def test_full_summary_separates_tuning_and_heldout_and_passes_contract(tmp_path):
    tuning = _fixture(tmp_path)

    first = summarize(tmp_path, tuning, bootstrap=500, seed=7)
    second = summarize(tmp_path, tuning, bootstrap=500, seed=7)
    markdown = render_markdown(first)

    assert first["subsets"]["heldout"]["n"] == 68
    assert first["subsets"]["tuning"]["n"] == 2
    assert first["subsets"]["full"]["n"] == 70
    reduction = first["subsets"]["heldout"]["paired"]["nonlocal_ca_clashes_lt_3p6A_reduction"]
    assert reduction["fraction"] == 0.75
    assert reduction["paired_target_bootstrap_95pct_ci"] == [0.75, 0.75]
    assert (
        first["subsets"]["heldout"]["conditions"][CONTROL]["means"]["ost_clashes_per_1k_atoms"]
        == 10.0
    )
    assert first["subsets"]["heldout"]["conditions"][CONTROL]["atom_count_sources"] == {
        "openstructure": 34,
        "local_geometry": 34,
    }
    assert first["heldout_decision"]["passed"] is True
    assert first["bootstrap"] == second["bootstrap"]
    assert first["subsets"]["heldout"]["paired"] == second["subsets"]["heldout"]["paired"]
    assert "Held-out decision: **PASS**" in markdown
    assert "Worst held-out accuracy regressions" in markdown
    assert "tuning | 2" in markdown


def test_summary_hard_fails_on_condition_target_mismatch(tmp_path):
    tuning = _fixture(tmp_path)
    manifest_path = tmp_path / "inference" / GUIDED / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["rows"].pop()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="target mismatch"):
        summarize(tmp_path, tuning, bootstrap=10)


def test_summary_fails_catastrophic_heldout_accuracy_regression(tmp_path):
    tuning = _fixture(tmp_path)
    summary_path = tmp_path / "scores" / GUIDED / "openstructure" / "summary.json"
    summary_rows = json.loads(summary_path.read_text())
    summary_rows["rows"][0]["oligo_gdtts"] = 0.5
    summary_path.write_text(json.dumps(summary_rows))

    result = summarize(tmp_path, tuning, bootstrap=200)

    assert result["heldout_decision"]["criteria"]["worst_target_gdt_ts_delta"]["passed"] is False
    assert result["passed"] is False


def test_read_tuning_ids_requires_two_unique_targets(tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("T0068\nt0069\n")
    assert read_tuning_ids(ids) == ["t0068", "t0069"]

    ids.write_text("t0068 t0068\n")
    with pytest.raises(ValueError, match="two unique tuning IDs"):
        read_tuning_ids(ids)
