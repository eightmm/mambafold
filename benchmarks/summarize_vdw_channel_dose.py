#!/usr/bin/env python3
"""Strict report for the exploratory independent-VDW channel dose probe.

Two targets were selected for high clash rates and one as an adverse-response
control from the prior VDW run. Consequently, this report may nominate a dose
for a later confirmatory run, but it never makes an independent efficacy claim.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

try:
    from benchmarks.summarize_physics_guidance import (
        MANIFEST_PROVENANCE_KEYS,
        _count,
        _finite,
        _index_rows,
        _load_condition,
    )
except ModuleNotFoundError:  # direct ``python benchmarks/...py`` execution
    from summarize_physics_guidance import (  # type: ignore[no-redef]
        MANIFEST_PROVENANCE_KEYS,
        _count,
        _finite,
        _index_rows,
        _load_condition,
    )


CONTROL = "steric_1"
TREATMENTS = (
    "steric_1_vdw_sep_s0p03_e8",
    "steric_1_vdw_sep_s0p10_e8",
    "steric_1_vdw_sep_s0p03_e2",
    "steric_1_vdw_sep_s0p10_e2",
)
DEFAULT_CONDITIONS = (CONTROL, *TREATMENTS)

EXPECTED_DOSES = {
    CONTROL: (0.0, 8),
    "steric_1_vdw_sep_s0p03_e8": (0.03, 8),
    "steric_1_vdw_sep_s0p10_e8": (0.10, 8),
    "steric_1_vdw_sep_s0p03_e2": (0.03, 2),
    "steric_1_vdw_sep_s0p10_e2": (0.10, 2),
}
VARIED_GUIDANCE_FIELDS = ("vdw_scale", "vdw_every_n_steps")
DIRECT_VDW_METRIC = "final_vdw_loss_tol_1p5"
DIAGNOSTIC_VDW_METRIC = "final_vdw_loss_tol_0p6"
VDW_METRICS = (DIAGNOSTIC_VDW_METRIC, DIRECT_VDW_METRIC)

MEAN_METRICS = (
    "gdt_ts",
    "lddt",
    "tm_score",
    "bond_p95_A",
    "hard_clashes_per_1k_atoms",
    "ost_model_clashes_per_1k_atoms",
    "runtime_s",
    *VDW_METRICS,
)

_REQUIRED_TREATMENT_CONSTANTS = {
    "vdw_start": 0.65,
    "vdw_overlap_tolerance_A": 1.5,
    "vdw_max_step_A": 0.01,
}
_EPS = 1e-12


def _guidance(manifest: dict[str, Any], condition: str) -> dict[str, Any]:
    value = manifest.get("geometry_guidance")
    if not isinstance(value, dict):
        raise ValueError(f"{condition}: missing geometry_guidance object")
    return value


def _validate_dose(manifest: dict[str, Any], condition: str) -> dict[str, float | int]:
    """Validate exact condition semantics and fixed independent-VDW settings."""
    guidance = _guidance(manifest, condition)
    expected_scale, expected_every = EXPECTED_DOSES[condition]
    scale = _finite(
        guidance.get("vdw_scale"),
        label=f"{condition} geometry_guidance.vdw_scale",
        minimum=0.0,
    )
    every = _count(
        guidance.get("vdw_every_n_steps"),
        label=f"{condition} geometry_guidance.vdw_every_n_steps",
    )
    if scale != expected_scale or every != expected_every:
        raise ValueError(
            f"{condition}: expected independent VDW dose "
            f"vdw_scale={expected_scale}, vdw_every_n_steps={expected_every}; "
            f"got {scale}, {every}"
        )

    legacy_weight = _finite(
        guidance.get("all_atom_clash_weight"),
        label=f"{condition} geometry_guidance.all_atom_clash_weight",
        minimum=0.0,
    )
    if legacy_weight != 0.0:
        raise ValueError(f"{condition}: legacy all_atom_clash_weight must be exactly 0")

    for field, expected in _REQUIRED_TREATMENT_CONSTANTS.items():
        value = _finite(
            guidance.get(field),
            label=f"{condition} geometry_guidance.{field}",
            minimum=0.0,
        )
        if value != expected:
            raise ValueError(
                f"{condition}: geometry_guidance.{field} must be exactly {expected}, got {value}"
            )
    return {"vdw_scale": scale, "vdw_every_n_steps": every}


def _comparison_config(manifest: dict[str, Any], condition: str) -> dict[str, Any]:
    """Remove only the two preregistered dose fields before comparison."""
    guidance = dict(_guidance(manifest, condition))
    for field in VARIED_GUIDANCE_FIELDS:
        if field not in guidance:
            raise ValueError(f"{condition}: missing geometry_guidance.{field}")
        guidance.pop(field)
    config = {
        key: value
        for key, value in manifest.items()
        if key not in MANIFEST_PROVENANCE_KEYS and key != "geometry_guidance"
    }
    config["geometry_guidance"] = guidance
    return config


def _load(
    root: Path, condition: str
) -> tuple[
    dict[str, dict[str, float | int]],
    dict[str, Any],
    dict[str, int | bool],
]:
    rows, manifest, coverage = _load_condition(root, condition)
    manifest_rows = _index_rows(manifest.get("rows"), key="pdb_id", label=f"{condition} manifest")
    for target, row in rows.items():
        for metric in VDW_METRICS:
            row[metric] = _finite(
                manifest_rows[target].get(metric),
                label=f"{condition}/{target} {metric}",
                minimum=0.0,
            )
    return rows, manifest, coverage


def _mean(rows: dict[str, dict[str, float | int]], metric: str) -> float:
    return statistics.fmean(float(row[metric]) for row in rows.values())


def _reduction_fraction(control: float, treatment: float) -> float | None:
    if control <= 0.0:
        return None
    return (control - treatment) / control


def _check(value: Any, threshold: str, passed: bool) -> dict[str, Any]:
    return {"value": value, "threshold": threshold, "pass": bool(passed)}


def _treatment_gates(
    *,
    treatment: str,
    rows: dict[str, dict[str, dict[str, float | int]]],
    aggregates: dict[str, dict[str, Any]],
    coverage_complete: bool,
    gt_sane: bool,
) -> dict[str, Any]:
    control_means = aggregates[CONTROL]["means"]
    treatment_row = aggregates[treatment]
    means = treatment_row["means"]
    deltas = treatment_row["paired_mean_delta_vs_control"]
    direct_reduction = _reduction_fraction(
        float(control_means[DIRECT_VDW_METRIC]),
        float(means[DIRECT_VDW_METRIC]),
    )
    ost_reduction = _reduction_fraction(
        float(control_means["ost_model_clashes_per_1k_atoms"]),
        float(means["ost_model_clashes_per_1k_atoms"]),
    )
    target_gdt_deltas = {
        target: float(rows[treatment][target]["gdt_ts"]) - float(rows[CONTROL][target]["gdt_ts"])
        for target in sorted(rows[CONTROL])
    }
    target_lddt_deltas = {
        target: float(rows[treatment][target]["lddt"]) - float(rows[CONTROL][target]["lddt"])
        for target in sorted(rows[CONTROL])
    }
    worst_gdt = min(target_gdt_deltas.values())
    worst_lddt = min(target_lddt_deltas.values())
    ca_lt_2 = int(treatment_row["pooled_counts"]["nonlocal_ca_clashes_lt_2A"])
    segment_lt_2p5_delta = int(
        treatment_row["pooled_counts"]["nonlocal_ca_segment_clashes_lt_2p5A"]
    ) - int(aggregates[CONTROL]["pooled_counts"]["nonlocal_ca_segment_clashes_lt_2p5A"])

    checks = {
        "coverage_complete_3_of_3": _check(
            coverage_complete, "all 5 conditions complete on exactly 3 targets", coverage_complete
        ),
        "gt_segment_sanity": _check(
            gt_sane, "GT segment <2.5 A count is zero with complete GT", gt_sane
        ),
        "direct_vdw_loss_mean_reduction": _check(
            direct_reduction,
            ">= 0.20",
            direct_reduction is not None and direct_reduction + _EPS >= 0.20,
        ),
        "ost_clash_mean_reduction": _check(
            ost_reduction,
            ">= 0.10",
            ost_reduction is not None and ost_reduction + _EPS >= 0.10,
        ),
        "hard_clash_mean_delta": _check(
            deltas["hard_clashes_per_1k_atoms"],
            "<= 0",
            float(deltas["hard_clashes_per_1k_atoms"]) <= _EPS,
        ),
        "gdt_ts_mean_delta": _check(
            deltas["gdt_ts"], ">= -0.002", float(deltas["gdt_ts"]) + _EPS >= -0.002
        ),
        "lddt_mean_delta": _check(
            deltas["lddt"], ">= -0.002", float(deltas["lddt"]) + _EPS >= -0.002
        ),
        "gdt_ts_each_target_delta": {
            **_check(
                worst_gdt,
                ">= -0.005",
                worst_gdt + _EPS >= -0.005,
            ),
            "per_target": target_gdt_deltas,
        },
        "lddt_each_target_delta": {
            **_check(
                worst_lddt,
                ">= -0.005",
                worst_lddt + _EPS >= -0.005,
            ),
            "per_target": target_lddt_deltas,
        },
        "bond_p95_mean_delta_A": _check(
            deltas["bond_p95_A"],
            "<= 0.005 A",
            float(deltas["bond_p95_A"]) <= 0.005 + _EPS,
        ),
        "nonlocal_ca_lt_2A": _check(ca_lt_2, "== 0", ca_lt_2 == 0),
        "nonlocal_ca_segment_lt_2p5_delta": _check(
            segment_lt_2p5_delta,
            "<= 0",
            segment_lt_2p5_delta <= 0,
        ),
    }
    return {
        "all_pass": all(bool(check["pass"]) for check in checks.values()),
        "checks": checks,
    }


def summarize(
    root: Path,
    conditions: list[str] | tuple[str, ...] = DEFAULT_CONDITIONS,
) -> dict[str, Any]:
    """Load and strictly gate the high-clash three-target dose experiment."""
    condition_list = list(conditions)
    if condition_list != list(DEFAULT_CONDITIONS):
        raise ValueError("conditions must be exactly, in order: " + ", ".join(DEFAULT_CONDITIONS))

    loaded = {condition: _load(root, condition) for condition in condition_list}
    rows = {condition: loaded[condition][0] for condition in condition_list}
    manifests = {condition: loaded[condition][1] for condition in condition_list}
    coverage = {condition: loaded[condition][2] for condition in condition_list}

    targets = set(rows[CONTROL])
    if len(targets) != 3:
        raise ValueError(
            f"exploratory VDW dose summary requires exactly 3 targets, got {len(targets)}"
        )
    for condition in condition_list:
        if set(rows[condition]) != targets:
            raise ValueError(f"{condition}: target set differs from {CONTROL}")
        if len(rows[condition]) != 3:
            raise ValueError(f"{condition}: expected complete 3/3 coverage")
        for target in targets:
            if int(rows[condition][target]["length"]) != int(rows[CONTROL][target]["length"]):
                raise ValueError(f"{condition}/{target}: condition length mismatch")

    doses = {
        condition: _validate_dose(manifests[condition], condition) for condition in condition_list
    }
    control_config = _comparison_config(manifests[CONTROL], CONTROL)
    for condition in TREATMENTS:
        if _comparison_config(manifests[condition], condition) != control_config:
            raise ValueError(
                f"{condition}: manifest/guidance config mismatch beyond "
                f"{', '.join(VARIED_GUIDANCE_FIELDS)}"
            )

    coverage_complete = all(
        bool(coverage[condition]["complete"])
        and int(coverage[condition]["manifest"]) == 3
        and int(coverage[condition]["local_geometry"]) == 3
        and int(coverage[condition]["openstructure_summary"]) == 3
        and int(coverage[condition]["openstructure_raw"]) == 3
        for condition in condition_list
    )
    if not coverage_complete:
        raise ValueError("incomplete 3/3 paired artifact coverage")
    gt_sane = all(
        bool(coverage[condition]["gt_present"])
        and int(coverage[condition]["gt_segment_lt_2p5_total"]) == 0
        for condition in condition_list
    )
    if not gt_sane:
        raise ValueError("complete GT with zero segment <2.5 A events is required")

    control_means = {metric: _mean(rows[CONTROL], metric) for metric in MEAN_METRICS}
    aggregates: dict[str, dict[str, Any]] = {}
    for condition in condition_list:
        means = {metric: _mean(rows[condition], metric) for metric in MEAN_METRICS}
        aggregates[condition] = {
            "n": 3,
            "dose": doses[condition],
            "means": means,
            "paired_mean_delta_vs_control": {
                metric: means[metric] - control_means[metric] for metric in MEAN_METRICS
            },
            "relative_reduction_vs_control": {
                DIRECT_VDW_METRIC: _reduction_fraction(
                    control_means[DIRECT_VDW_METRIC], means[DIRECT_VDW_METRIC]
                ),
                "ost_model_clashes_per_1k_atoms": _reduction_fraction(
                    control_means["ost_model_clashes_per_1k_atoms"],
                    means["ost_model_clashes_per_1k_atoms"],
                ),
            },
            "pooled_counts": {
                "nonlocal_ca_clashes_lt_2A": sum(
                    int(row["nonlocal_ca_clashes_lt_2A"]) for row in rows[condition].values()
                ),
                "nonlocal_ca_segment_clashes_lt_2p5A": sum(
                    int(row["nonlocal_ca_segment_clashes_lt_2p5A"])
                    for row in rows[condition].values()
                ),
            },
        }

    gates = {
        condition: _treatment_gates(
            treatment=condition,
            rows=rows,
            aggregates=aggregates,
            coverage_complete=coverage_complete,
            gt_sane=gt_sane,
        )
        for condition in TREATMENTS
    }
    passing = [condition for condition in TREATMENTS if gates[condition]["all_pass"]]
    selected = (
        min(
            passing,
            key=lambda condition: (
                float(aggregates[condition]["means"]["runtime_s"]),
                float(doses[condition]["vdw_scale"]),
                condition,
            ),
        )
        if passing
        else None
    )

    per_target = []
    for target in sorted(targets):
        condition_values: dict[str, Any] = {}
        for condition in condition_list:
            values = {metric: rows[condition][target][metric] for metric in MEAN_METRICS}
            values.update(
                {
                    "nonlocal_ca_clashes_lt_2A": rows[condition][target][
                        "nonlocal_ca_clashes_lt_2A"
                    ],
                    "nonlocal_ca_segment_clashes_lt_2p5A": rows[condition][target][
                        "nonlocal_ca_segment_clashes_lt_2p5A"
                    ],
                }
            )
            condition_values[condition] = {
                "values": values,
                "delta_vs_control": {
                    metric: float(rows[condition][target][metric])
                    - float(rows[CONTROL][target][metric])
                    for metric in MEAN_METRICS
                }
                | {
                    "nonlocal_ca_clashes_lt_2A": int(
                        rows[condition][target]["nonlocal_ca_clashes_lt_2A"]
                    )
                    - int(rows[CONTROL][target]["nonlocal_ca_clashes_lt_2A"]),
                    "nonlocal_ca_segment_clashes_lt_2p5A": int(
                        rows[condition][target]["nonlocal_ca_segment_clashes_lt_2p5A"]
                    )
                    - int(rows[CONTROL][target]["nonlocal_ca_segment_clashes_lt_2p5A"]),
                },
            }
        per_target.append(
            {
                "target": target,
                "length": int(rows[CONTROL][target]["length"]),
                "conditions": condition_values,
            }
        )

    return {
        "schema_version": 1,
        "experiment": "exploratory_independent_vdw_channel_dose",
        "control_condition": CONTROL,
        "conditions": condition_list,
        "target_count": 3,
        "target_ids": sorted(targets),
        "study_scope": {
            "selection": (
                "exploratory selection: two high-clash targets plus one "
                "prior adverse-response control"
            ),
            "independent_confirmation_claim": False,
            "allowed_next_claim": "candidate for an independent confirmatory evaluation only",
        },
        "vdw_metric_roles": {
            DIAGNOSTIC_VDW_METRIC: "diagnostic_only_not_gated",
            DIRECT_VDW_METRIC: "OST_aligned_primary_gate",
        },
        "coverage": {
            "paired_complete_3_of_3": coverage_complete,
            "gt_segment_sanity": gt_sane,
            "by_condition": coverage,
        },
        "aggregates": aggregates,
        "treatment_gates": gates,
        "selection": {
            "rule": "among all-gate passes, lowest mean runtime_s then lower vdw_scale",
            "passing_conditions": passing,
            "selected_condition": selected,
            "status": "candidate_for_confirmation" if selected else "no_promotion",
            "independent_confirmation_claim": False,
        },
        "rows": per_target,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _mean_delta(row: dict[str, Any], metric: str) -> str:
    return (
        f"{_fmt(row['means'][metric])} ({float(row['paired_mean_delta_vs_control'][metric]):+.4f})"
    )


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the audit-friendly exploratory report."""
    lines = [
        "# Exploratory independent-VDW channel dose probe",
        "",
        "**Scope:** two deliberately selected high-clash targets plus one prior "
        "adverse-response control. This is a dose-selection probe only; it makes "
        "**no independent confirmation or efficacy claim**.",
        "",
        "Paired coverage: **3/3** targets in all five conditions. GT segment "
        f"sanity: **{'PASS' if summary['coverage']['gt_segment_sanity'] else 'FAIL'}**. "
        "Parentheses are paired mean deltas versus `steric_1`; lower is better "
        "for geometry, clash, runtime, and direct VDW loss. The 0.6 Å VDW loss "
        "is diagnostic only; gates use the OST-aligned 1.5 Å loss.",
        "",
        "## Condition means and paired deltas",
        "",
        "| Condition | scale/every | GDT-TS (Δ) | lDDT (Δ) | TM (Δ) | "
        "Bond p95 Å (Δ) | Hard clash/1k (Δ) | OST clash/1k (Δ) | "
        "Runtime s (Δ) | VDW loss tol 0.6 Å (Δ, diagnostic) | "
        "VDW loss tol 1.5 Å (Δ, gated) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in summary["conditions"]:
        row = summary["aggregates"][condition]
        dose = row["dose"]
        lines.append(
            f"| {condition} | {dose['vdw_scale']:.2f}/{dose['vdw_every_n_steps']} | "
            f"{_mean_delta(row, 'gdt_ts')} | {_mean_delta(row, 'lddt')} | "
            f"{_mean_delta(row, 'tm_score')} | {_mean_delta(row, 'bond_p95_A')} | "
            f"{_mean_delta(row, 'hard_clashes_per_1k_atoms')} | "
            f"{_mean_delta(row, 'ost_model_clashes_per_1k_atoms')} | "
            f"{_mean_delta(row, 'runtime_s')} | "
            f"{_mean_delta(row, DIAGNOSTIC_VDW_METRIC)} | "
            f"{_mean_delta(row, DIRECT_VDW_METRIC)} |"
        )

    lines.extend(
        [
            "",
            "## Preregistered exploratory gates",
            "",
            "| Condition | Overall | VDW 1.5 Å ≥20% | OST ≥10% | Hard nonincrease | "
            "Mean GDT/lDDT | Each-target GDT/lDDT | Bond Δ≤0.005 Å | "
            "Cα<2 Å=0 | Segment<2.5 no increase |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in TREATMENTS:
        gate = summary["treatment_gates"][condition]
        check = gate["checks"]
        mean_accuracy = check["gdt_ts_mean_delta"]["pass"] and check["lddt_mean_delta"]["pass"]
        per_target_accuracy = (
            check["gdt_ts_each_target_delta"]["pass"] and check["lddt_each_target_delta"]["pass"]
        )
        lines.append(
            f"| {condition} | {_fmt(gate['all_pass'])} | "
            f"{_fmt(check['direct_vdw_loss_mean_reduction']['pass'])} | "
            f"{_fmt(check['ost_clash_mean_reduction']['pass'])} | "
            f"{_fmt(check['hard_clash_mean_delta']['pass'])} | "
            f"{_fmt(mean_accuracy)} | {_fmt(per_target_accuracy)} | "
            f"{_fmt(check['bond_p95_mean_delta_A']['pass'])} | "
            f"{_fmt(check['nonlocal_ca_lt_2A']['pass'])} | "
            f"{_fmt(check['nonlocal_ca_segment_lt_2p5_delta']['pass'])} |"
        )

    selection = summary["selection"]
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"Status: **{selection['status']}**. Selected condition: "
            f"`{selection['selected_condition']}`. The rule is lowest mean runtime, "
            "then lower VDW scale, among treatments passing every gate. Any selected "
            "setting remains only a candidate for an independent confirmatory run.",
            "",
            "## Per-target paired deltas",
            "",
            "| Target | Condition | GDT Δ | lDDT Δ | TM Δ | Bond p95 Δ Å | "
            "Hard clash/1k Δ | OST clash/1k Δ | Runtime Δ s | "
            "VDW loss 0.6 Å Δ | VDW loss 1.5 Å Δ | Cα<2 Å | Segment<2.5 Δ |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["rows"]:
        for condition in TREATMENTS:
            condition_row = row["conditions"][condition]
            values = condition_row["values"]
            delta = condition_row["delta_vs_control"]
            lines.append(
                f"| {row['target']} | {condition} | {delta['gdt_ts']:+.4f} | "
                f"{delta['lddt']:+.4f} | {delta['tm_score']:+.4f} | "
                f"{delta['bond_p95_A']:+.4f} | "
                f"{delta['hard_clashes_per_1k_atoms']:+.4f} | "
                f"{delta['ost_model_clashes_per_1k_atoms']:+.4f} | "
                f"{delta['runtime_s']:+.4f} | "
                f"{delta[DIAGNOSTIC_VDW_METRIC]:+.4f} | "
                f"{delta[DIRECT_VDW_METRIC]:+.4f} | "
                f"{values['nonlocal_ca_clashes_lt_2A']} | "
                f"{delta['nonlocal_ca_segment_clashes_lt_2p5A']:+d} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(DEFAULT_CONDITIONS),
        help="Exact five-condition exploratory independent-VDW dose set",
    )
    args = parser.parse_args()
    summary = summarize(args.root, args.conditions)
    json_path = args.root / "vdw_dose_comparison.json"
    markdown_path = args.root / "vdw_dose_comparison.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    markdown_path.write_text(render_markdown(summary))
    print(json.dumps(summary["selection"], indent=2))


if __name__ == "__main__":
    main()
