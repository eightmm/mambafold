#!/usr/bin/env python3
"""Summarize the controlled self-overlap guidance scale sweep."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

try:
    from benchmarks.summarize_stereochemical_examples import _condition_rows
except ModuleNotFoundError:  # direct ``python benchmarks/...py`` execution
    from summarize_stereochemical_examples import _condition_rows


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _mean(rows: dict[str, dict[str, float | int]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows.values())


def summarize(root: Path) -> dict[str, Any]:
    sweep = _read(root / "sweep_manifest.json")
    conditions = list(sweep["conditions"])
    rows_by_condition = {condition: _condition_rows(root, condition) for condition in conditions}
    target_set = set(rows_by_condition[conditions[0]])
    for condition, rows in rows_by_condition.items():
        if set(rows) != target_set:
            raise ValueError(f"{condition}: target set differs from baseline")

    aggregates: dict[str, dict[str, float | int | bool]] = {}
    for condition in conditions:
        rows = rows_by_condition[condition]
        manifest = _read(root / "inference" / condition / "manifest.json")
        timing = {row["pdb_id"]: row for row in manifest["rows"]}
        aggregates[condition] = {
            "steric_scale": float(manifest["geometry_guidance"]["steric_scale"]),
            "gdt_ts_mean": _mean(rows, "gdt_ts"),
            "lddt_mean": _mean(rows, "lddt"),
            "bond_p95_A_mean": _mean(rows, "bond_p95_A"),
            "hard_clashes_per_1k_mean": _mean(rows, "hard_clashes_per_1k_atoms"),
            "ost_clashes_total": sum(int(row["ost_clashes"]) for row in rows.values()),
            "nonlocal_ca_min_A_min": min(float(row["nonlocal_ca_min_A"]) for row in rows.values()),
            "nonlocal_ca_clashes_lt_2A_total": sum(
                int(row["nonlocal_ca_clashes_lt_2A"]) for row in rows.values()
            ),
            "nonlocal_ca_clashes_lt_3A_total": sum(
                int(row["nonlocal_ca_clashes_lt_3A"]) for row in rows.values()
            ),
            "nonlocal_ca_clashes_lt_3p6A_total": sum(
                int(row["nonlocal_ca_clashes_lt_3p6A"]) for row in rows.values()
            ),
            "nonlocal_ca_penetration_rms_A_mean": _mean(rows, "nonlocal_ca_penetration_rms_A"),
            "runtime_s_total": sum(float(row["runtime_s"]) for row in timing.values()),
            "peak_vram_gib_max": max(float(row["peak_vram_gib"]) for row in timing.values()),
        }

    control = aggregates["split_local_control"]
    control_clashes_3p6 = int(control["nonlocal_ca_clashes_lt_3p6A_total"])
    for condition, aggregate in aggregates.items():
        aggregate["guardrail_pass"] = bool(
            float(aggregate["gdt_ts_mean"]) - float(control["gdt_ts_mean"]) >= -0.005
            and float(aggregate["lddt_mean"]) - float(control["lddt_mean"]) >= -0.005
            and float(aggregate["bond_p95_A_mean"]) - float(control["bond_p95_A_mean"]) <= 0.01
        )
        clashes_3p6 = int(aggregate["nonlocal_ca_clashes_lt_3p6A_total"])
        aggregate["nonlocal_ca_clashes_lt_3p6A_reduction_fraction"] = (
            (control_clashes_3p6 - clashes_3p6) / control_clashes_3p6
            if control_clashes_3p6 > 0
            else float(clashes_3p6 == 0)
        )
        aggregate["preregistered_success"] = bool(
            condition.startswith("steric_")
            and aggregate["guardrail_pass"]
            and float(aggregate["nonlocal_ca_clashes_lt_3p6A_reduction_fraction"]) >= 0.75
            and int(aggregate["nonlocal_ca_clashes_lt_2A_total"]) == 0
        )

    guardrailed_candidates = [
        condition
        for condition in conditions
        if condition.startswith("steric_") and aggregates[condition]["guardrail_pass"]
    ]
    best_guardrailed = min(
        guardrailed_candidates,
        key=lambda condition: (
            int(aggregates[condition]["nonlocal_ca_clashes_lt_3p6A_total"]),
            float(aggregates[condition]["nonlocal_ca_penetration_rms_A_mean"]),
            int(aggregates[condition]["nonlocal_ca_clashes_lt_3A_total"]),
            int(aggregates[condition]["nonlocal_ca_clashes_lt_2A_total"]),
            int(aggregates[condition]["ost_clashes_total"]),
            float(aggregates[condition]["steric_scale"]),
        ),
        default=None,
    )
    successful_candidates = [
        condition
        for condition in guardrailed_candidates
        if aggregates[condition]["preregistered_success"]
    ]
    best_success = min(
        successful_candidates,
        key=lambda condition: float(aggregates[condition]["steric_scale"]),
        default=None,
    )

    per_target = []
    for target in sorted(target_set):
        per_target.append(
            {
                "target": target,
                "length": int(rows_by_condition["baseline"][target]["length"]),
                "conditions": {
                    condition: rows_by_condition[condition][target] for condition in conditions
                },
            }
        )
    return {
        "schema_version": 1,
        "experiment": "self_overlap_guidance_v1",
        "selection": "Exploratory CASP14 targets with known gross self-overlap",
        "conditions": conditions,
        "target_count": len(target_set),
        "aggregates": aggregates,
        "effect_control_condition": "split_local_control",
        "best_guardrailed_condition": best_guardrailed,
        "best_preregistered_success_condition": best_success,
        "preregistered_success": best_success is not None,
        "rows": per_target,
    }


def _fmt(value: float | int) -> str:
    return str(value) if isinstance(value, int) else f"{value:.4f}"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Self-overlap guidance sweep",
        "",
        "Controlled ESMC-6B step-132000 EMA, seed-0, SDE-500 comparison. "
        "The model and kernels are loaded once; only inference guidance changes.",
        "",
        "## Aggregate",
        "",
        "| Condition | Steric scale | GDT-TS | lDDT | Bond p95 Å | OST clashes | "
        "min nonlocal Cα Å | Cα <2 Å | Cα <3 Å | Cα <3.6 Å | reduction | "
        "penetration RMS Å | Time s | Peak GiB | Guardrail | Success |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for condition in summary["conditions"]:
        row = summary["aggregates"][condition]
        lines.append(
            f"| {condition} | {_fmt(row['steric_scale'])} | "
            f"{_fmt(row['gdt_ts_mean'])} | {_fmt(row['lddt_mean'])} | "
            f"{_fmt(row['bond_p95_A_mean'])} | {row['ost_clashes_total']} | "
            f"{_fmt(row['nonlocal_ca_min_A_min'])} | "
            f"{row['nonlocal_ca_clashes_lt_2A_total']} | "
            f"{row['nonlocal_ca_clashes_lt_3A_total']} | "
            f"{row['nonlocal_ca_clashes_lt_3p6A_total']} | "
            f"{100.0 * row['nonlocal_ca_clashes_lt_3p6A_reduction_fraction']:.1f}% | "
            f"{_fmt(row['nonlocal_ca_penetration_rms_A_mean'])} | "
            f"{_fmt(row['runtime_s_total'])} | {_fmt(row['peak_vram_gib_max'])} | "
            f"{'PASS' if row['guardrail_pass'] else 'FAIL'} | "
            f"{'PASS' if row['preregistered_success'] else 'FAIL'} |"
        )

    lines.extend(
        [
            "",
            f"Best guardrailed condition: `{summary['best_guardrailed_condition']}`.",
            f"Pre-registered success: `{summary['preregistered_success']}`; "
            f"lowest successful scale: `{summary['best_preregistered_success_condition']}`.",
            "",
            "## Per target",
            "",
            "| Target | L | Condition | GDT-TS | lDDT | Bond p95 Å | OST clashes | "
            "min nonlocal Cα Å | Cα <3 Å | Cα <3.6 Å | penetration RMS Å |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for target_row in summary["rows"]:
        for condition in summary["conditions"]:
            row = target_row["conditions"][condition]
            lines.append(
                f"| {target_row['target']} | {target_row['length']} | {condition} | "
                f"{_fmt(row['gdt_ts'])} | {_fmt(row['lddt'])} | "
                f"{_fmt(row['bond_p95_A'])} | {row['ost_clashes']} | "
                f"{_fmt(row['nonlocal_ca_min_A'])} | "
                f"{row['nonlocal_ca_clashes_lt_3A']} | "
                f"{row['nonlocal_ca_clashes_lt_3p6A']} | "
                f"{_fmt(row['nonlocal_ca_penetration_rms_A'])} |"
            )

    lines.extend(["", "## Structures", ""])
    for target_row in summary["rows"]:
        target = target_row["target"]
        lines.append(f"### {target}")
        lines.append("")
        for condition in summary["conditions"]:
            lines.append(
                f"- {condition}: [PDB](inference/{condition}/{target}_pred.pdb), "
                f"[CIF](inference/{condition}/{target}_pred.cif)"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.root)
    (args.root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.root / "comparison.md").write_text(render_markdown(summary) + "\n")
    print(json.dumps(summary["aggregates"], indent=2))


if __name__ == "__main__":
    main()
