#!/usr/bin/env python3
"""Evaluate one selected physics-guidance treatment on full CASP14.

The generic physics-guidance summary owns all source-artifact validation.  This
module adds the confirmatory analysis contract: the targets used to select the
treatment are reported separately from held-out targets, and only the held-out
partition determines the primary gates.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Sequence

try:
    from benchmarks.summarize_physics_guidance import (
        CONTROL,
        MEAN_METRICS,
        POINT_COUNT_METRICS,
        SEGMENT_COUNT_METRICS,
    )
    from benchmarks.summarize_physics_guidance import (
        summarize as summarize_sources,
    )
except ModuleNotFoundError:  # direct ``python benchmarks/...py`` execution
    from summarize_physics_guidance import (  # type: ignore[no-redef]
        CONTROL,
        MEAN_METRICS,
        POINT_COUNT_METRICS,
        SEGMENT_COUNT_METRICS,
    )
    from summarize_physics_guidance import (
        summarize as summarize_sources,
    )


SEGMENT_REDUCTION_METRIC = "nonlocal_ca_segment_clashes_lt_2p5A"
CA_LT_2_METRIC = "nonlocal_ca_clashes_lt_2A"
ACCURACY_METRICS = ("gdt_ts", "lddt")
WORST_LIMIT = 5


def _read_selection_ids(path: Path) -> list[str]:
    """Read a strict one-target-per-line selection file."""
    try:
        text = path.read_text()
    except FileNotFoundError as exc:
        raise ValueError(f"missing selection IDs file: {path}") from exc

    target_ids: list[str] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 1:
            raise ValueError(f"{path}:{line_number}: expected exactly one target ID")
        target_ids.append(fields[0])

    if not target_ids:
        raise ValueError("selection IDs must be nonempty")
    if len(set(target_ids)) != len(target_ids):
        duplicates = sorted(target for target in set(target_ids) if target_ids.count(target) > 1)
        raise ValueError(f"selection IDs contain duplicates: {duplicates}")
    return target_ids


def _validate_conditions(conditions: Sequence[str]) -> tuple[str, str]:
    values = list(conditions)
    if len(values) != 2 or values[0] != CONTROL or values[1] == CONTROL:
        raise ValueError(f"conditions must be exactly {CONTROL!r} followed by one treatment")
    return values[0], values[1]


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of no values")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _ci_record(
    estimate: float | None,
    samples: list[float],
    *,
    n_resamples: int,
) -> dict[str, float | int | None]:
    if estimate is None or not samples:
        return {
            "estimate": estimate,
            "lower": None,
            "upper": None,
            "n_defined": len(samples),
            "n_resamples": n_resamples,
        }
    return {
        "estimate": estimate,
        "lower": _percentile(samples, 0.025),
        "upper": _percentile(samples, 0.975),
        "n_defined": len(samples),
        "n_resamples": n_resamples,
    }


def _bootstrap_intervals(
    control_rows: dict[str, dict[str, float | int]],
    treatment_rows: dict[str, dict[str, float | int]],
    target_ids: Sequence[str],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, dict[str, float | int | None]]:
    if n_resamples <= 0:
        raise ValueError("bootstrap must be a positive integer")
    if not target_ids:
        raise ValueError("cannot bootstrap an empty target subset")

    rng = random.Random(seed)
    accuracy_samples = {metric: [] for metric in ACCURACY_METRICS}
    reduction_samples: list[float] = []
    n = len(target_ids)
    for _ in range(n_resamples):
        sampled = [target_ids[rng.randrange(n)] for _ in range(n)]
        for metric in ACCURACY_METRICS:
            accuracy_samples[metric].append(
                statistics.fmean(
                    float(treatment_rows[target][metric]) - float(control_rows[target][metric])
                    for target in sampled
                )
            )

        control_total = sum(
            int(control_rows[target][SEGMENT_REDUCTION_METRIC]) for target in sampled
        )
        if control_total > 0:
            treatment_total = sum(
                int(treatment_rows[target][SEGMENT_REDUCTION_METRIC]) for target in sampled
            )
            reduction_samples.append(1.0 - treatment_total / control_total)

    result: dict[str, dict[str, float | int | None]] = {}
    for metric in ACCURACY_METRICS:
        estimate = statistics.fmean(
            float(treatment_rows[target][metric]) - float(control_rows[target][metric])
            for target in target_ids
        )
        result[f"{metric}_mean_delta"] = _ci_record(
            estimate, accuracy_samples[metric], n_resamples=n_resamples
        )

    control_total = sum(
        int(control_rows[target][SEGMENT_REDUCTION_METRIC]) for target in target_ids
    )
    treatment_total = sum(
        int(treatment_rows[target][SEGMENT_REDUCTION_METRIC]) for target in target_ids
    )
    reduction_estimate = 1.0 - treatment_total / control_total if control_total > 0 else None
    result["segment_lt_2p5_pooled_reduction"] = _ci_record(
        reduction_estimate, reduction_samples, n_resamples=n_resamples
    )
    return result


def _relative_change(control: float, treatment: float) -> float | None:
    if control == 0.0:
        return None
    return treatment / control - 1.0


def _worst_deltas(
    control_rows: dict[str, dict[str, float | int]],
    treatment_rows: dict[str, dict[str, float | int]],
    target_ids: Sequence[str],
    metric: str,
) -> list[dict[str, float | int | str]]:
    rows = [
        {
            "target": target,
            "length": int(control_rows[target]["length"]),
            "control": float(control_rows[target][metric]),
            "treatment": float(treatment_rows[target][metric]),
            "delta": float(treatment_rows[target][metric]) - float(control_rows[target][metric]),
        }
        for target in target_ids
    ]
    return sorted(rows, key=lambda row: (float(row["delta"]), str(row["target"])))[:WORST_LIMIT]


def _aggregate_subset(
    rows_by_condition: dict[str, dict[str, dict[str, float | int]]],
    conditions: Sequence[str],
    target_ids: Sequence[str],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    control, treatment = conditions
    control_rows = rows_by_condition[control]
    treatment_rows = rows_by_condition[treatment]
    condition_aggregates: dict[str, Any] = {}

    for condition in conditions:
        rows = rows_by_condition[condition]
        means = {
            metric: statistics.fmean(float(rows[target][metric]) for target in target_ids)
            for metric in MEAN_METRICS
        }
        control_means = {
            metric: statistics.fmean(float(control_rows[target][metric]) for target in target_ids)
            for metric in MEAN_METRICS
        }
        total_runtime = sum(float(rows[target]["runtime_s"]) for target in target_ids)
        control_runtime = sum(float(control_rows[target]["runtime_s"]) for target in target_ids)
        condition_aggregates[condition] = {
            "n": len(target_ids),
            "means": means,
            "paired_mean_delta_vs_control": {
                metric: means[metric] - control_means[metric] for metric in MEAN_METRICS
            },
            "pooled_nonlocal_ca_point_counts": {
                metric: sum(int(rows[target][metric]) for target in target_ids)
                for metric in POINT_COUNT_METRICS
            },
            "pooled_nonlocal_ca_segment_counts": {
                metric: sum(int(rows[target][metric]) for target in target_ids)
                for metric in SEGMENT_COUNT_METRICS
            },
            "runtime": {
                "total_s": total_runtime,
                "mean_s": total_runtime / len(target_ids),
                "paired_mean_delta_s_vs_control": (total_runtime - control_runtime)
                / len(target_ids),
                "delta_total_s_vs_control": total_runtime - control_runtime,
                "relative_total_change_vs_control": _relative_change(
                    control_runtime, total_runtime
                ),
            },
        }

    return {
        "n": len(target_ids),
        "target_ids": list(target_ids),
        "conditions": condition_aggregates,
        "paired_bootstrap_95_ci": _bootstrap_intervals(
            control_rows,
            treatment_rows,
            target_ids,
            n_resamples=n_resamples,
            seed=seed,
        ),
        "worst_target_deltas": {
            metric: _worst_deltas(control_rows, treatment_rows, target_ids, metric)
            for metric in ACCURACY_METRICS
        },
    }


def _clash_gate(control: float, treatment: float) -> tuple[float | None, bool]:
    relative = _relative_change(control, treatment)
    if relative is None:
        return None, treatment == 0.0
    return relative, relative <= 0.10


def _segment_efficacy_context(
    subsets: dict[str, Any],
    conditions: Sequence[str],
) -> dict[str, Any]:
    """Describe where segment repair is observable without overstating evidence."""
    control, treatment = conditions
    records: dict[str, Any] = {}
    for subset_name in ("selection", "heldout", "full"):
        subset = subsets[subset_name]
        control_total = int(
            subset["conditions"][control]["pooled_nonlocal_ca_segment_counts"][
                SEGMENT_REDUCTION_METRIC
            ]
        )
        treatment_total = int(
            subset["conditions"][treatment]["pooled_nonlocal_ca_segment_counts"][
                SEGMENT_REDUCTION_METRIC
            ]
        )
        testable = control_total > 0
        records[subset_name] = {
            "control_events": control_total,
            "treatment_events": treatment_total,
            "pooled_reduction": (1.0 - treatment_total / control_total if testable else None),
            "testable": testable,
            "status": (
                "descriptive_repair_observed"
                if testable and treatment_total < control_total
                else "descriptive_no_repair"
                if testable
                else "not_testable_no_control_events"
            ),
        }
    heldout = records["heldout"]
    return {
        "selection_role": "treatment_selection_and_mechanistic_repair_check",
        "heldout_role": "safety_and_nonregression_check",
        "independent_efficacy_confirmed": False,
        "independent_efficacy_reason": (
            "held-out control has no segment events"
            if not heldout["testable"]
            else "this contract preregistered held-out safety, not an efficacy threshold"
        ),
        "subsets": records,
    }


def _primary_gates(
    source_summary: dict[str, Any],
    heldout: dict[str, Any],
    conditions: Sequence[str],
) -> dict[str, Any]:
    control, treatment = conditions
    control_row = heldout["conditions"][control]
    treatment_row = heldout["conditions"][treatment]
    bootstrap = heldout["paired_bootstrap_95_ci"]

    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, passed: bool, **details: Any) -> None:
        checks[name] = {"pass": bool(passed), **details}

    source_coverage = source_summary["coverage"]
    add(
        "coverage_complete",
        bool(source_coverage["paired_complete"]),
        paired_complete=bool(source_coverage["paired_complete"]),
    )
    add("heldout_nonempty", heldout["n"] > 0, n=heldout["n"])

    gt_by_condition = source_coverage["by_condition"]
    gt_sane = bool(source_summary["gt_segment_validation"]["all_present_gt_totals_zero"])
    gt_sane = gt_sane and all(
        bool(gt_by_condition[condition]["gt_present"])
        and int(gt_by_condition[condition]["gt_segment_lt_2p5_total"]) == 0
        for condition in conditions
    )
    add("gt_segment_sanity", gt_sane)

    for metric in ACCURACY_METRICS:
        delta = float(treatment_row["paired_mean_delta_vs_control"][metric])
        ci_lower = bootstrap[f"{metric}_mean_delta"]["lower"]
        add(
            f"{metric}_mean_delta",
            delta >= -0.005,
            value=delta,
            minimum=-0.005,
        )
        add(
            f"{metric}_ci_lower",
            ci_lower is not None and float(ci_lower) >= -0.01,
            value=ci_lower,
            minimum=-0.01,
        )
        worst_delta = min(float(row["delta"]) for row in heldout["worst_target_deltas"][metric])
        add(
            f"{metric}_worst_target_delta",
            worst_delta >= -0.05,
            value=worst_delta,
            minimum=-0.05,
        )

    bond_delta = float(treatment_row["paired_mean_delta_vs_control"]["bond_p95_A"])
    add(
        "bond_p95_delta",
        bond_delta <= 0.005,
        value=bond_delta,
        maximum=0.005,
    )

    control_ca_lt_2 = int(control_row["pooled_nonlocal_ca_point_counts"][CA_LT_2_METRIC])
    treatment_ca_lt_2 = int(treatment_row["pooled_nonlocal_ca_point_counts"][CA_LT_2_METRIC])
    add(
        "ca_lt_2_remains_zero",
        control_ca_lt_2 == 0 and treatment_ca_lt_2 == 0,
        control=control_ca_lt_2,
        treatment=treatment_ca_lt_2,
    )

    for name, metric in (
        ("hard_clash_relative_increase", "hard_clashes_per_1k_atoms"),
        ("ost_clash_relative_increase", "ost_model_clashes_per_1k_atoms"),
    ):
        relative, passed = _clash_gate(
            float(control_row["means"][metric]),
            float(treatment_row["means"][metric]),
        )
        add(name, passed, value=relative, maximum=0.10)

    runtime_relative = treatment_row["runtime"]["relative_total_change_vs_control"]
    add(
        "runtime_relative_increase",
        runtime_relative is not None and float(runtime_relative) <= 0.10,
        value=runtime_relative,
        maximum=0.10,
    )

    control_segments = int(
        control_row["pooled_nonlocal_ca_segment_counts"][SEGMENT_REDUCTION_METRIC]
    )
    treatment_segments = int(
        treatment_row["pooled_nonlocal_ca_segment_counts"][SEGMENT_REDUCTION_METRIC]
    )
    add(
        "segment_lt_2p5_not_increased",
        treatment_segments <= control_segments,
        control=control_segments,
        treatment=treatment_segments,
    )

    return {
        "subset": "heldout",
        "claim": "safety_and_nonregression",
        "checks": checks,
        "all_pass": all(bool(check["pass"]) for check in checks.values()),
    }


def summarize(
    root: Path,
    selection_ids_path: Path,
    conditions: Sequence[str],
    *,
    bootstrap: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Validate source artifacts and build a selection/held-out/full report."""
    condition_pair = _validate_conditions(conditions)
    if bootstrap <= 0:
        raise ValueError("bootstrap must be a positive integer")

    # This call is intentionally the only artifact loader: its strict target,
    # score-status, finite-value, GT, and manifest-config checks remain the
    # single source of truth for both the generic and confirmatory reports.
    source_summary = summarize_sources(root, condition_pair)
    full_ids = list(source_summary["target_ids"])
    selection_ids = _read_selection_ids(selection_ids_path)
    selection_set = set(selection_ids)
    full_set = set(full_ids)
    unknown = sorted(selection_set - full_set)
    if unknown:
        raise ValueError(f"selection IDs are not present in full targets: {unknown}")
    if selection_set == full_set:
        raise ValueError("selection IDs must be a strict subset of full target IDs")

    selection_sorted = sorted(selection_set)
    heldout_ids = sorted(full_set - selection_set)
    if not heldout_ids:
        raise ValueError("held-out target set must be nonempty")

    rows_by_condition: dict[str, dict[str, dict[str, float | int]]] = {
        condition: {} for condition in condition_pair
    }
    for target_row in source_summary["rows"]:
        target = str(target_row["target"])
        for condition in condition_pair:
            rows_by_condition[condition][target] = target_row["conditions"][condition]

    partition_ids = {
        "selection": selection_sorted,
        "heldout": heldout_ids,
        "full": full_ids,
    }
    subsets = {
        name: _aggregate_subset(
            rows_by_condition,
            condition_pair,
            target_ids,
            n_resamples=bootstrap,
            seed=seed + offset,
        )
        for offset, (name, target_ids) in enumerate(partition_ids.items())
    }
    gates = _primary_gates(source_summary, subsets["heldout"], condition_pair)
    efficacy_context = _segment_efficacy_context(subsets, condition_pair)

    return {
        "schema_version": 1,
        "experiment": "physics_guidance_casp14_full_safety_check",
        "control_condition": condition_pair[0],
        "treatment_condition": condition_pair[1],
        "conditions": list(condition_pair),
        "selection_ids_file": str(selection_ids_path),
        "target_count": len(full_ids),
        "selection_count": len(selection_sorted),
        "heldout_count": len(heldout_ids),
        "bootstrap": {
            "n_resamples": bootstrap,
            "seed": seed,
            "confidence_level": 0.95,
            "unit": "target",
            "paired": True,
        },
        "interpretation_limits": [
            "This is a paired inference-time guidance ablation on one frozen checkpoint, "
            "not a leakage-free estimate of de novo folding accuracy.",
            "The ESMC sequence model was pretrained on database snapshots newer than "
            "CASP14; only treatment-minus-control effects are used for the guidance claim.",
            "Treatment selection used the declared selection targets; primary gates use "
            "only the disjoint held-out targets.",
        ],
        "source_validation": {
            "coverage": source_summary["coverage"],
            "gt_segment_validation": source_summary["gt_segment_validation"],
        },
        "subsets": subsets,
        "segment_efficacy_context": efficacy_context,
        "primary_gates": gates,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _ci_text(record: dict[str, Any]) -> str:
    if record["lower"] is None:
        return "NA"
    return f"[{record['lower']:+.4f}, {record['upper']:+.4f}]"


def render_markdown(summary: dict[str, Any]) -> str:
    control = summary["control_condition"]
    treatment = summary["treatment_condition"]
    lines = [
        "# CASP14 full physics-guidance safety check",
        "",
        f"Comparison: `{control}` vs `{treatment}`. Primary decisions use only "
        f"the **{summary['heldout_count']} held-out targets**; "
        f"{summary['selection_count']} selection targets are reported separately.",
        "",
        "Interpretation: this is a paired inference-time ablation on one frozen "
        "checkpoint. ESMC pretraining used post-CASP14 database snapshots, so absolute "
        "CASP14 accuracy is not treated as a leakage-free benchmark claim; the guidance "
        "claim uses treatment-minus-control effects, with decisions made on the disjoint "
        "held-out subset.",
        "",
        "## Primary held-out safety gates",
        "",
        f"Overall safety: **{'PASS' if summary['primary_gates']['all_pass'] else 'FAIL'}**",
        "",
        "| Gate | Result | Value/details |",
        "|---|---:|---|",
    ]
    for name, check in summary["primary_gates"]["checks"].items():
        details = ", ".join(f"{key}={_fmt(value)}" for key, value in check.items() if key != "pass")
        lines.append(f"| {name} | {'PASS' if check['pass'] else 'FAIL'} | {details} |")

    efficacy = summary["segment_efficacy_context"]
    selection = efficacy["subsets"]["selection"]
    heldout = efficacy["subsets"]["heldout"]
    lines.extend(
        [
            "",
            "## Segment-efficacy scope",
            "",
            f"Selection repair (tuned/mechanistic): {selection['control_events']} -> "
            f"{selection['treatment_events']} events below 2.5 Å. Held-out efficacy: "
            f"{heldout['status']} ({heldout['control_events']} -> "
            f"{heldout['treatment_events']}). **Independent efficacy is not claimed**; "
            "the held-out decision is safety/nonregression only.",
        ]
    )

    for subset_name in ("selection", "heldout", "full"):
        subset = summary["subsets"][subset_name]
        lines.extend(
            [
                "",
                f"## {subset_name} (n={subset['n']})",
                "",
                "| Condition | GDT-TS (Δ) | lDDT (Δ) | TM (Δ) | Bond p95 Å (Δ) | "
                "Hard clash/1k (Δ) | OST clash/1k (Δ) | Segment penetration Å (Δ) | "
                "Runtime total s (mean, Δ mean) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for condition in summary["conditions"]:
            row = subset["conditions"][condition]
            means = row["means"]
            deltas = row["paired_mean_delta_vs_control"]

            def cell(metric: str) -> str:
                return f"{_fmt(means[metric])} ({deltas[metric]:+.4f})"

            runtime = row["runtime"]
            lines.append(
                f"| {condition} | {cell('gdt_ts')} | {cell('lddt')} | "
                f"{cell('tm_score')} | {cell('bond_p95_A')} | "
                f"{cell('hard_clashes_per_1k_atoms')} | "
                f"{cell('ost_model_clashes_per_1k_atoms')} | "
                f"{cell('nonlocal_ca_segment_penetration_rms_A')} | "
                f"{runtime['total_s']:.2f} ({runtime['mean_s']:.2f}, "
                f"{runtime['paired_mean_delta_s_vs_control']:+.2f}) |"
            )

        lines.extend(
            [
                "",
                "| Condition | Point <2 Å | Point <3 Å | Point <3.6 Å | "
                "Segment <2 Å | Segment <2.5 Å | Segment <3 Å |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for condition in summary["conditions"]:
            row = subset["conditions"][condition]
            point = row["pooled_nonlocal_ca_point_counts"]
            segment = row["pooled_nonlocal_ca_segment_counts"]
            lines.append(
                f"| {condition} | {point['nonlocal_ca_clashes_lt_2A']} | "
                f"{point['nonlocal_ca_clashes_lt_3A']} | "
                f"{point['nonlocal_ca_clashes_lt_3p6A']} | "
                f"{segment['nonlocal_ca_segment_clashes_lt_2A']} | "
                f"{segment['nonlocal_ca_segment_clashes_lt_2p5A']} | "
                f"{segment['nonlocal_ca_segment_clashes_lt_3A']} |"
            )

        bootstrap = subset["paired_bootstrap_95_ci"]
        lines.extend(
            [
                "",
                f"Paired bootstrap 95% CI: GDT-TS Δ "
                f"{_ci_text(bootstrap['gdt_ts_mean_delta'])}; lDDT Δ "
                f"{_ci_text(bootstrap['lddt_mean_delta'])}; segment <2.5 Å pooled "
                f"reduction {_ci_text(bootstrap['segment_lt_2p5_pooled_reduction'])}.",
                "",
                "Worst per-target deltas:",
                "",
                "| Metric | Target | L | Control | Treatment | Δ |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for metric in ACCURACY_METRICS:
            for row in subset["worst_target_deltas"][metric]:
                lines.append(
                    f"| {metric} | {row['target']} | {row['length']} | "
                    f"{row['control']:.4f} | {row['treatment']:.4f} | "
                    f"{row['delta']:+.4f} |"
                )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection-ids", type=Path, required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        required=True,
        help="Exactly steric_1 followed by one selected treatment",
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary = summarize(
        args.root,
        args.selection_ids,
        args.conditions,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    json_path = args.root / "full_confirmation.json"
    markdown_path = args.root / "full_confirmation.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    markdown_path.write_text(render_markdown(summary))
    print(json.dumps(summary["primary_gates"], indent=2))


if __name__ == "__main__":
    main()
