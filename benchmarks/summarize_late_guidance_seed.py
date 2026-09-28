#!/usr/bin/env python3
"""Summarize paired baseline/guided CASP15/16 scores for one sampler seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

METRICS = ("lddt", "bb_lddt", "tm_score")
EXPECTED = {"casp15": (19, 28), "casp16": (18, 18)}


def read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def arm(root: Path, cohort: str, seed: int) -> dict:
    base = root / cohort
    rollout = read(base / "rollout/rollout.json")
    scores = read(base / "scores-admitted/summary.json")
    targets = read(
        base / "scores-admitted" / ("target-summary.json" if cohort == "casp15" else "summary.json")
    )
    manifest = read(base / "pairs-admitted/manifest.json")
    expected_targets, expected_pairs = EXPECTED[cohort]
    if not scores["evaluation_complete"] or scores["failures"]:
        raise ValueError(f"incomplete scoring: {base}")
    if scores["successful_targets"] != expected_pairs:
        raise ValueError(f"wrong scored pair count: {base}")
    if manifest["reference_pair_count"] != expected_pairs:
        raise ValueError(f"wrong staged pair count: {base}")
    if rollout["sampled"] != expected_targets or rollout["seed"] != seed:
        raise ValueError(f"wrong rollout count or seed: {base}")
    if not (
        rollout["n_steps"] == 500
        and rollout["method"] == "sde"
        and rollout["sde_tau"] == 0.01
        and rollout["sde_log_timesteps"]
    ):
        raise ValueError(f"sampler settings differ: {base}")
    rows = targets["target_rows" if cohort == "casp15" else "rows"]
    by_target = {row["target_id"].lower(): row for row in rows}
    if len(by_target) != expected_targets or set(by_target) != set(rollout["per_target"]):
        raise ValueError(f"wrong target set: {base}")
    references = {row["pair_id"]: row["reference_sha256"] for row in manifest["rows"]}
    violations: dict[str, dict[str, float]] = {}
    for target in by_target:
        pairs = [row for row in manifest["rows"] if row["target_id"].lower() == target]
        weight_sum = sum(row["weight"] for row in pairs)
        if not pairs or weight_sum <= 0:
            raise ValueError(f"missing pair weights for {target}: {base}")
        raw = {
            row["pair_id"]: read(base / "scores-admitted/raw" / f"{row['pair_id']}.json")
            for row in pairs
        }
        violations[target] = {
            kind: sum(row["weight"] * len(raw[row["pair_id"]][f"model_{kind}"]) for row in pairs)
            / weight_sum
            for kind in ("bad_bonds", "bad_angles", "clashes")
        }
    return {
        "rollout": rollout,
        "by_target": by_target,
        "references": references,
        "scorer": scores["openstructure"],
        "violations": violations,
    }


def summarize(baseline: dict, guided: dict, cohort: str) -> dict:
    if baseline["references"] != guided["references"]:
        raise ValueError(f"reference pairs differ: {cohort}")
    if baseline["scorer"] != guided["scorer"]:
        raise ValueError(f"OpenStructure versions differ: {cohort}")
    b_roll = baseline["rollout"]
    g_roll = guided["rollout"]
    for field in ("checkpoint_sha256", "config_sha256", "seed", "seed_scheme"):
        if b_roll[field] != g_roll[field]:
            raise ValueError(f"different {field}: {cohort}")
    baseline_config = b_roll.get("geometry_guidance")
    if baseline_config is not None and baseline_config["max_step_A"] != 0.0:
        raise ValueError(f"baseline is guided: {cohort}")
    config = g_roll.get("geometry_guidance")
    if config is None or (
        config["start_t"],
        config["every_n_steps"],
        config["max_step_A"],
        config["bond_weight"],
        config["angle_weight"],
        config["clash_weight"],
        config["backbone_scale"],
    ) != (
        0.9,
        10,
        0.02,
        1.0,
        1.0,
        1.0,
        0.1,
    ):
        raise ValueError(f"unexpected guidance settings: {cohort}")
    ids = sorted(baseline["by_target"])
    if ids != sorted(guided["by_target"]):
        raise ValueError(f"unpaired target IDs: {cohort}")
    rows = [
        {
            "target_id": target,
            "baseline": {m: baseline["by_target"][target][m] for m in METRICS},
            "guided": {m: guided["by_target"][target][m] for m in METRICS},
            "baseline_stereo_violations": baseline["violations"][target],
            "guided_stereo_violations": guided["violations"][target],
        }
        for target in ids
    ]
    metrics = {}
    for metric in METRICS:
        differences = [row["guided"][metric] - row["baseline"][metric] for row in rows]
        metrics[metric] = {
            "baseline_mean": mean(row["baseline"][metric] for row in rows),
            "guided_mean": mean(row["guided"][metric] for row in rows),
            "paired_mean_delta": mean(differences),
            "improved": sum(value > 0 for value in differences),
            "worsened": sum(value < 0 for value in differences),
            "unchanged": sum(value == 0 for value in differences),
            "worst_delta": min(differences),
        }
    violations = {
        kind: {
            "baseline_mean": mean(row["baseline_stereo_violations"][kind] for row in rows),
            "guided_mean": mean(row["guided_stereo_violations"][kind] for row in rows),
        }
        for kind in ("bad_bonds", "bad_angles", "clashes")
    }
    return {
        "target_count": len(rows),
        "reference_pair_count": len(baseline["references"]),
        "checkpoint_sha256": b_roll["checkpoint_sha256"],
        "config_sha256": b_roll["config_sha256"],
        "reference_sha256_by_pair": baseline["references"],
        "guided_settings": {k: v for k, v in config.items() if k != "props"},
        "metrics": metrics,
        "stereo_violation_means": violations,
        "target_rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--guided-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "seed": args.seed,
        "description": (
            "Paired admitted-target comparison with frozen references and "
            "default OpenStructure scoring."
        ),
        "cohorts": {},
    }
    for cohort in EXPECTED:
        baseline = arm(args.baseline_root, cohort, args.seed)
        guided = arm(args.guided_root, cohort, args.seed)
        result["cohorts"][cohort] = summarize(baseline, guided, cohort)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
