"""Focused tests for the paired CASP14 geometry fine-tune report."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from benchmarks.summarize_geoft_casp14 import (
    BASE,
    CONDITIONS,
    SPECS,
    main,
    render_markdown,
    summarize,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _condition_values(condition: str, *, no_promotion: bool) -> dict[str, float | int]:
    if condition == BASE:
        return {"clashes": 10, "gdt": 0.700, "lddt": 0.800, "crossings": 8}
    if no_promotion:
        return {"clashes": 6, "gdt": 0.694, "lddt": 0.794, "crossings": 7}
    return {
        "ft250": {"clashes": 6, "gdt": 0.699, "lddt": 0.799, "crossings": 7},
        "ft500": {"clashes": 5, "gdt": 0.696, "lddt": 0.797, "crossings": 5},
        "ft1000": {"clashes": 4, "gdt": 0.697, "lddt": 0.798, "crossings": 4},
        "ft1500": {"clashes": 3, "gdt": 0.694, "lddt": 0.799, "crossings": 3},
        "ft2000": {"clashes": 2, "gdt": 0.698, "lddt": 0.798, "crossings": 2},
    }[condition]


def _fixture(root: Path, *, no_promotion: bool = False) -> None:
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True)
    ids = [f"t{index:04d}" for index in range(70)]
    ids_path = artifacts / "casp14_whole70.txt"
    ids_path.write_text("\n".join(ids) + "\n", encoding="utf-8")

    base_config = artifacts / SPECS[BASE].config_name
    ft_config = artifacts / SPECS["ft250"].config_name
    base_config.write_text("model: esmc6b\ntotal_steps: 170000\n", encoding="utf-8")
    ft_config.write_text("model: esmc6b\ntotal_steps: 2000\nw_ost_clash: 0.5\n", encoding="utf-8")

    checkpoints: dict[str, Path] = {}
    for condition in CONDITIONS:
        checkpoint = artifacts / condition / SPECS[condition].checkpoint_name
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"checkpoint:{condition}".encode())
        checkpoints[condition] = checkpoint

    provenance_path = artifacts / "source_provenance.json"
    provenance = {
        "schema_version": 1,
        "source": {
            "checkpoint": str(checkpoints[BASE]),
            "checkpoint_sha256": _sha(checkpoints[BASE]),
            "source_config": str(base_config),
            "source_config_sha256": _sha(base_config),
            "step": 170000,
        },
        "finetune": {
            "config": str(ft_config),
            "config_sha256": _sha(ft_config),
            "initial_weights": "ema",
            "optimizer_scheduler": "fresh",
            "start_step": 0,
        },
    }
    _write_json(provenance_path, provenance)

    for condition in CONDITIONS:
        config = base_config if condition == BASE else ft_config
        manifest = {
            "condition": condition,
            "dataset": "CASP14 whole70 exact70",
            "checkpoint_step": SPECS[condition].checkpoint_step,
            "checkpoint": str(checkpoints[condition]),
            "checkpoint_sha256": _sha(checkpoints[condition]),
            "config": str(config),
            "config_sha256": _sha(config),
            "source_provenance": None if condition == BASE else str(provenance_path),
            "source_provenance_sha256": None if condition == BASE else _sha(provenance_path),
            "ids_file": str(ids_path),
            "sampler": "sde",
            "n_steps": 500,
            "seed": 0,
            "seeds": [0],
            "sde_tau": 0.01,
            "sde_eps": 0.01,
            "sde_w_cutoff": 0.99,
            "sde_log_timesteps": True,
            "max_length": 1024,
            "output_format": "both",
            "use_ema": True,
            "single_chain_only": True,
            "geometry_guidance_preset": "bond_cleanup",
            "guidance_off": True,
            "geometry_guidance": {
                "scale": 0.0,
                "steric_scale": 0.0,
                "vdw_scale": 0.0,
                "start": 0.5,
                "every_n_steps": 1,
                "bond_weight": 1.0,
                "steric_ca_min_dist_A": 3.6,
            },
            "n_predicted": 70,
            "expected_target_count": 70,
            "rows": [
                {
                    "pdb_id": target,
                    "L": 100 + index,
                    "n_chains": 1,
                    "n_seeds_ok": 1,
                }
                for index, target in enumerate(ids)
            ],
        }
        _write_json(root / "conditions" / condition / "manifest.json", manifest)

        values = _condition_values(condition, no_promotion=no_promotion)
        ost_rows = []
        local_rows = []
        for target in ids:
            raw = {
                "status": "SUCCESS",
                "n_atoms": 1000,
                "oligo_gdtts": values["gdt"],
                "lddt": values["lddt"],
                "bb_lddt": float(values["lddt"]) - 0.01,
                "tm_score": float(values["gdt"]) + 0.02,
                "model_clashes": list(range(int(values["clashes"]))),
            }
            _write_json(
                root / "scores" / condition / "openstructure" / f"{target}.json",
                raw,
            )
            ost_rows.append(
                {
                    "target": target,
                    "oligo_gdtts": raw["oligo_gdtts"],
                    "lddt": raw["lddt"],
                    "bb_lddt": raw["bb_lddt"],
                    "tm_score": raw["tm_score"],
                }
            )
            local_rows.append(
                {
                    "pdb_id": target,
                    "pred": {
                        "n_atoms": 1000,
                        "clashes_per_1k_atoms": float(values["clashes"]) + 1.0,
                        "bond_p95_A": 0.20 - 0.01 * CONDITIONS.index(condition),
                        "nonlocal_ca_segment_clashes_lt_2p5A": values["crossings"],
                    },
                }
            )
        _write_json(
            root / "scores" / condition / "openstructure" / "summary.json",
            {
                "target_count": 70,
                "success_count": 70,
                "openstructure": {
                    "version": "OpenStructure 2.9.1",
                    "command": "ost compare-structures --lddt --bb-lddt --tm-score",
                },
                "rows": ost_rows,
            },
        )
        _write_json(
            root / "scores" / condition / "local_geometry.json",
            {
                "n": 70,
                "clash_threshold_A": 1.5,
                "nonlocal_ca_metric_definition": {
                    "sequence_separation_gt": 12,
                    "point_penetration_floor_A": 3.6,
                    "segment_penetration_floor_A": 2.5,
                    "segment_max_edge_A": 6.0,
                },
                "rows": local_rows,
            },
        )


def test_summary_selects_earliest_eligible_checkpoint_deterministically(tmp_path: Path) -> None:
    _fixture(tmp_path)

    first = summarize(tmp_path, bootstrap=500, seed=7)
    second = summarize(tmp_path, bootstrap=500, seed=7)
    markdown = render_markdown(first)

    assert first == second
    assert first["coverage"]["paired"] == 70
    assert first["decision"] == {
        "status": "promote",
        "selected_condition": "ft500",
        "eligible_conditions": ["ft500", "ft1000", "ft2000"],
        "selection_rule": "earliest eligible fine-tune checkpoint",
    }
    ft500 = first["conditions"]["ft500"]
    assert ft500["gate"]["criteria"]["ost_model_clash_reduction"]["observed"] == 0.5
    assert ft500["deltas_vs_base"]["gdt_ts"]["candidate_minus_base_mean"] == pytest.approx(-0.004)
    assert ft500["deltas_vs_base"]["gdt_ts"]["paired_target_bootstrap_95pct_ci"] == pytest.approx(
        [-0.004, -0.004]
    )
    assert ft500["pooled_ca_crossings"] == 350
    assert "retrospective engineering evidence" in markdown
    assert "Selected condition: **ft500**" in markdown


def test_summary_returns_no_promotion_when_no_checkpoint_passes(tmp_path: Path) -> None:
    _fixture(tmp_path, no_promotion=True)

    result = summarize(tmp_path, bootstrap=50, seed=11)

    assert result["decision"]["status"] == "no_promotion"
    assert result["decision"]["selected_condition"] is None
    assert result["decision"]["eligible_conditions"] == []


def test_summary_fails_closed_on_target_mismatch(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "scores" / "ft1000" / "local_geometry.json"
    local = json.loads(path.read_text())
    local["rows"].pop()
    _write_json(path, local)

    with pytest.raises(ValueError, match="target mismatch"):
        summarize(tmp_path, bootstrap=10)


def test_summary_fails_closed_on_checkpoint_sha(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "conditions" / "ft500" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["checkpoint_sha256"] = "0" * 64
    _write_json(path, manifest)

    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        summarize(tmp_path, bootstrap=10)


def test_summary_fails_closed_when_any_guidance_channel_is_active(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "conditions" / "ft500" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["geometry_guidance"]["vdw_scale"] = 0.1
    manifest["guidance_off"] = False
    _write_json(path, manifest)

    with pytest.raises(ValueError, match="guidance_off"):
        summarize(tmp_path, bootstrap=10)


def test_summary_fails_closed_on_internal_source_provenance(tmp_path: Path) -> None:
    _fixture(tmp_path)
    manifest_path = tmp_path / "conditions" / "ft250" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    provenance_path = Path(manifest["source_provenance"])
    provenance = json.loads(provenance_path.read_text())
    provenance["source"]["checkpoint_sha256"] = "f" * 64
    _write_json(provenance_path, provenance)
    provenance_sha = _sha(provenance_path)
    for condition in CONDITIONS[1:]:
        path = tmp_path / "conditions" / condition / "manifest.json"
        condition_manifest = json.loads(path.read_text())
        condition_manifest["source_provenance_sha256"] = provenance_sha
        _write_json(path, condition_manifest)

    with pytest.raises(ValueError, match="does not match base170k"):
        summarize(tmp_path, bootstrap=10)


def test_cli_writes_fixed_output_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fixture(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["summarize_geoft_casp14.py", "--root", str(tmp_path), "--bootstrap", "10"],
    )

    main()

    assert (tmp_path / "geoft_comparison.json").stat().st_size > 0
    assert (tmp_path / "geoft_comparison.md").stat().st_size > 0
