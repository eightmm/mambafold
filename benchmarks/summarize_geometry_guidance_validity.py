#!/usr/bin/env python3
"""Aggregate paired accuracy and stereochemical validity for guidance runs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

MODELS = (
    ("esm3", "MambaFold-ESM3, step 120,000"),
    ("esmc6b", "MambaFold-ESMC-6B, step 119,500"),
)
CONDITIONS = ("baseline", "guided")
HIGHER_IS_BETTER = ("gdt_ts", "gdt_ha", "tm_score", "lddt", "bb_lddt")
LOWER_IS_BETTER = (
    "rmsd_A",
    "bond_mae_A",
    "bond_p95_A",
    "bond_bad_frac_gt_0p10A",
    "hard_clashes_per_1k_atoms",
    "ost_clashes_per_1k_atoms",
    "ost_bad_bonds_per_1k_atoms",
    "ost_bad_angles_per_1k_atoms",
    "ca_chirality_wrong_frac",
    "ca_chirality_degenerate_frac_lt_0p1",
)
METRICS = HIGHER_IS_BETTER + LOWER_IS_BETTER


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else float("nan")


def load_condition(root: Path, model: str, condition: str) -> dict[str, dict[str, float]]:
    score_dir = root / "scores" / model / condition
    ost_summary = json.loads((score_dir / "openstructure/summary.json").read_text())
    local = json.loads((score_dir / "local_geometry.json").read_text())
    local_rows = {row["pdb_id"]: row["pred"] for row in local["rows"]}
    ost_rows = {row["target"]: row for row in ost_summary["rows"]}
    target_ids = set(local_rows) & set(ost_rows)
    if len(target_ids) != 69 or set(local_rows) != set(ost_rows):
        raise SystemExit(
            f"{model}/{condition}: expected identical 69-target local/OpenStructure rows, "
            f"got local={len(local_rows)}, OpenStructure={len(ost_rows)}"
        )

    rows: dict[str, dict[str, float]] = {}
    for target_id in sorted(target_ids):
        raw = json.loads((score_dir / "openstructure" / f"{target_id}.json").read_text())
        local_row = local_rows[target_id]
        n_atoms = int(local_row["n_atoms"])
        if raw.get("status") != "SUCCESS" or n_atoms < 1:
            raise SystemExit(f"Invalid score record: {model}/{condition}/{target_id}")
        rows[target_id] = {
            "gdt_ts": float(raw["oligo_gdtts"]),
            "gdt_ha": float(raw["oligo_gdtha"]),
            "tm_score": float(raw["tm_score"]),
            "lddt": float(raw["lddt"]),
            "bb_lddt": float(raw["bb_lddt"]),
            "rmsd_A": float(raw["rmsd"]),
            "bond_mae_A": float(local_row["bond_mae_A"]),
            "bond_p95_A": float(local_row["bond_p95_A"]),
            "bond_bad_frac_gt_0p10A": float(local_row["bond_bad_frac_gt_0p10A"]),
            "hard_clashes_per_1k_atoms": float(local_row["clashes_per_1k_atoms"]),
            "ost_clashes_per_1k_atoms": len(raw.get("model_clashes", [])) * 1000.0 / n_atoms,
            "ost_bad_bonds_per_1k_atoms": len(raw.get("model_bad_bonds", [])) * 1000.0 / n_atoms,
            "ost_bad_angles_per_1k_atoms": len(raw.get("model_bad_angles", [])) * 1000.0 / n_atoms,
            "ca_chirality_wrong_frac": float(local_row["ca_chirality_wrong_frac"]),
            "ca_chirality_degenerate_frac_lt_0p1": float(
                local_row["ca_chirality_degenerate_frac_lt_0p1"]
            ),
        }
    return rows


def bootstrap_delta(values: np.ndarray, seed: int = 20260813) -> list[float]:
    rng = np.random.default_rng(seed)
    sample_count = 20_000
    means = np.empty(sample_count, dtype=np.float64)
    for start in range(0, sample_count, 1_000):
        count = min(1_000, sample_count - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[indices].mean(axis=1)
    return [float(value) for value in np.percentile(means, [2.5, 97.5])]


def aggregate_model(
    baseline: dict[str, dict[str, float]], guided: dict[str, dict[str, float]]
) -> dict[str, Any]:
    if set(baseline) != set(guided):
        raise SystemExit("Baseline/guided target sets differ")
    ids = sorted(baseline)
    conditions = {
        "baseline": {
            metric: mean([baseline[target][metric] for target in ids]) for metric in METRICS
        },
        "guided": {metric: mean([guided[target][metric] for target in ids]) for metric in METRICS},
    }
    deltas = {}
    for metric in METRICS:
        paired = np.asarray(
            [guided[target][metric] - baseline[target][metric] for target in ids],
            dtype=np.float64,
        )
        baseline_mean = conditions["baseline"][metric]
        delta = float(paired.mean())
        deltas[metric] = {
            "guided_minus_baseline": delta,
            "paired_bootstrap_95pct_ci": bootstrap_delta(paired),
            "relative_change_percent": (
                delta / baseline_mean * 100.0 if abs(baseline_mean) > 1e-12 else None
            ),
        }
    return {
        "n": len(ids),
        "target_ids": ids,
        "conditions": conditions,
        "deltas": deltas,
        "per_target": {
            target: {"baseline": baseline[target], "guided": guided[target]} for target in ids
        },
    }


def fmt(value: float, percent: bool = False) -> str:
    return f"{value * 100:.2f}%" if percent else f"{value:.4f}"


def render_table(result: dict[str, Any]) -> list[str]:
    labels = (
        ("bond_bad_frac_gt_0p10A", "Bond violations >0.10 Å", True),
        ("bond_mae_A", "Bond MAE (Å)", False),
        ("ost_bad_bonds_per_1k_atoms", "OST bad bonds / 1k atoms", False),
        ("ost_bad_angles_per_1k_atoms", "OST bad angles / 1k atoms", False),
        ("ost_clashes_per_1k_atoms", "OST clashes / 1k atoms", False),
        ("hard_clashes_per_1k_atoms", "Hard clashes / 1k atoms", False),
        ("ca_chirality_wrong_frac", "Wrong Cα chirality", True),
        ("ca_chirality_degenerate_frac_lt_0p1", "Degenerate Cα chirality", True),
        ("gdt_ts", "GDT-TS", False),
        ("tm_score", "TM-score", False),
        ("lddt", "all-atom lDDT", False),
        ("bb_lddt", "backbone lDDT", False),
        ("rmsd_A", "RMSD (Å)", False),
    )
    lines = [
        "| Metric | Unguided | Guided | Guided − unguided | 95% paired CI |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric, label, percent in labels:
        baseline = result["conditions"]["baseline"][metric]
        guided = result["conditions"]["guided"][metric]
        delta = result["deltas"][metric]["guided_minus_baseline"]
        low, high = result["deltas"][metric]["paired_bootstrap_95pct_ci"]
        lines.append(
            f"| {label} | {fmt(baseline, percent)} | {fmt(guided, percent)} | "
            f"{fmt(delta, percent)} | [{fmt(low, percent)}, {fmt(high, percent)}] |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-description",
        default="SDE500 log timesteps, seed 0, geometry guidance off",
    )
    args = parser.parse_args()
    root = args.experiment_root.resolve()

    output: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "experiment": {
            "dataset": "CASP14 full70 minus T1061",
            "target_count": 69,
            "baseline": args.baseline_description,
            "change": (
                "GT-free geometry guidance scale=0.1, start=0.5, every step, "
                "bond=1.0, CA-angle=0.25, CA-clash=0.1"
            ),
            "primary_success": (
                ">=25% mean reduction in first-shell bond violations for both models; "
                "GDT-TS drop <=0.01; OST clash and bad-angle rates do not increase >10%"
            ),
            "selection_note": (
                "Scale 0.1 was selected from prior exploratory T1061; T1061 is excluded here"
            ),
        },
        "models": {},
    }
    md = [
        "# Geometry-guided inference validity",
        "",
        (
            "Paired seed-0 comparison on 69 CASP14 targets; T1061 is excluded "
            "because it selected the guidance scale."
        ),
        "",
        (
            "Guidance is ground-truth-free and changes inference only: scale 0.1, "
            "start time 0.5, every solver step, with bond/CA-angle/CA-clash "
            "weights 1.0/0.25/0.1."
        ),
        "",
    ]
    for model, label in MODELS:
        baseline = load_condition(root, model, "baseline")
        guided = load_condition(root, model, "guided")
        result = aggregate_model(baseline, guided)
        output["models"][model] = {"label": label, **result}
        md.extend((f"## {label}", ""))
        md.extend(render_table(result))
        md.append("")

    pass_by_model = {}
    for model, _ in MODELS:
        result = output["models"][model]
        bond = result["deltas"]["bond_bad_frac_gt_0p10A"]
        gdt = result["deltas"]["gdt_ts"]["guided_minus_baseline"]
        baseline_conditions = result["conditions"]["baseline"]
        guided_conditions = result["conditions"]["guided"]
        bond_reduction = -float(bond["relative_change_percent"] or 0.0)
        clash_increase = (
            guided_conditions["ost_clashes_per_1k_atoms"]
            / max(baseline_conditions["ost_clashes_per_1k_atoms"], 1e-12)
            - 1.0
        ) * 100.0
        angle_increase = (
            guided_conditions["ost_bad_angles_per_1k_atoms"]
            / max(baseline_conditions["ost_bad_angles_per_1k_atoms"], 1e-12)
            - 1.0
        ) * 100.0
        pass_by_model[model] = {
            "passed": (
                bond_reduction >= 25.0
                and gdt >= -0.01
                and clash_increase <= 10.0
                and angle_increase <= 10.0
            ),
            "bond_violation_reduction_percent": bond_reduction,
            "gdt_ts_delta": gdt,
            "ost_clash_change_percent": clash_increase,
            "ost_bad_angle_change_percent": angle_increase,
        }
    output["success_criterion"] = {
        "by_model": pass_by_model,
        "passed_both_models": all(row["passed"] for row in pass_by_model.values()),
    }
    md.extend(("## Pre-registered decision", ""))
    for model, label in MODELS:
        row = pass_by_model[model]
        md.append(
            f"- {label}: {'PASS' if row['passed'] else 'FAIL'} — "
            f"bond violations {row['bond_violation_reduction_percent']:.1f}% lower, "
            f"ΔGDT-TS {row['gdt_ts_delta']:+.4f}, OST clashes "
            f"{row['ost_clash_change_percent']:+.1f}%, bad angles "
            f"{row['ost_bad_angle_change_percent']:+.1f}%."
        )
    md.extend(
        (
            "",
            (
                "OpenStructure issue counts and the local bond/clash/chirality "
                "metrics are validity diagnostics; they are not MolProbity scores."
            ),
            "",
        )
    )

    (root / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    (root / "RESULTS.md").write_text("\n".join(md))
    print(json.dumps(output["success_criterion"], indent=2))


if __name__ == "__main__":
    main()
