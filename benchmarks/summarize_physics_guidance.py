#!/usr/bin/env python3
"""Summarize paired CASP14 inference-time physics-guidance conditions.

This module only combines already-produced inference, local-geometry, and
OpenStructure artifacts.  It deliberately fails closed when a condition is
not a complete paired run or when anything other than the two intended
guidance weights differs from the control configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from numbers import Real
from pathlib import Path
from typing import Any

CONTROL = "steric_1"
DEFAULT_CONDITIONS = (
    CONTROL,
    "steric_1_vdw",
    "steric_1_segment",
    "steric_1_vdw_segment",
)
INTENDED_GUIDANCE_FIELDS = (
    "all_atom_clash_weight",
    "steric_segment_weight",
)

MEAN_METRICS = (
    "gdt_ts",
    "lddt",
    "tm_score",
    "bond_p95_A",
    "hard_clashes_per_1k_atoms",
    "ost_model_clashes_per_1k_atoms",
    "nonlocal_ca_segment_penetration_rms_A",
)
POINT_COUNT_METRICS = (
    "nonlocal_ca_clashes_lt_2A",
    "nonlocal_ca_clashes_lt_3A",
    "nonlocal_ca_clashes_lt_3p6A",
)
SEGMENT_COUNT_METRICS = (
    "nonlocal_ca_segment_clashes_lt_2A",
    "nonlocal_ca_segment_clashes_lt_2p5A",
    "nonlocal_ca_segment_clashes_lt_3A",
)

# These keys record where/how shards were assembled, not scientific run
# configuration.  Everything else in the manifest is compared exactly after
# removing the two intentionally varied guidance weights.
MANIFEST_PROVENANCE_KEYS = {
    "condition",
    "rows",
    "generated_at",
    "created_at",
    "parallel_target_shard_merge",
    "source_shards",
    "source_ids_files",
    "source_target_shards",
}


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"missing required artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON artifact: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _finite(value: Any, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label}: expected a numeric value, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label}: expected a finite value, got {value!r}")
    if minimum is not None and number < minimum:
        raise ValueError(f"{label}: expected value >= {minimum}, got {number}")
    return number


def _count(value: Any, *, label: str) -> int:
    number = _finite(value, label=label, minimum=0.0)
    integer = int(number)
    if float(integer) != number:
        raise ValueError(f"{label}: expected an integer count, got {value!r}")
    return integer


def _index_rows(
    rows: Any,
    *,
    key: str,
    label: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{label}: expected a rows list")
    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{label}: row {index} is not an object")
        target = row.get(key)
        if not isinstance(target, str) or not target:
            raise ValueError(f"{label}: row {index} has invalid {key}={target!r}")
        if target in indexed:
            raise ValueError(f"{label}: duplicate target {target}")
        indexed[target] = row
    if not indexed:
        raise ValueError(f"{label}: no target rows")
    return indexed


def _manifest_config(manifest: dict[str, Any], *, condition: str) -> dict[str, Any]:
    manifest_condition = manifest.get("condition")
    if manifest_condition not in (None, condition):
        raise ValueError(f"{condition}: manifest condition is {manifest_condition!r}")
    guidance = manifest.get("geometry_guidance")
    if not isinstance(guidance, dict):
        raise ValueError(f"{condition}: missing geometry_guidance object")
    normalized_guidance = dict(guidance)
    for field in INTENDED_GUIDANCE_FIELDS:
        if field not in normalized_guidance:
            raise ValueError(f"{condition}: missing geometry_guidance.{field}")
        _finite(
            normalized_guidance[field],
            label=f"{condition} geometry_guidance.{field}",
            minimum=0.0,
        )
        normalized_guidance.pop(field)

    config = {
        key: value
        for key, value in manifest.items()
        if key not in MANIFEST_PROVENANCE_KEYS and key != "geometry_guidance"
    }
    config["geometry_guidance"] = normalized_guidance
    return config


def _guidance_weights(manifest: dict[str, Any], *, condition: str) -> dict[str, float]:
    guidance = manifest["geometry_guidance"]
    return {
        field: _finite(
            guidance[field],
            label=f"{condition} geometry_guidance.{field}",
            minimum=0.0,
        )
        for field in INTENDED_GUIDANCE_FIELDS
    }


def _load_condition(
    root: Path,
    condition: str,
) -> tuple[dict[str, dict[str, float | int]], dict[str, Any], dict[str, int | bool]]:
    manifest = _read(root / "inference" / condition / "manifest.json")
    manifest_rows = _index_rows(manifest.get("rows"), key="pdb_id", label=f"{condition} manifest")
    local = _read(root / "scores" / condition / "local_geometry.json")
    local_rows = _index_rows(local.get("rows"), key="pdb_id", label=f"{condition} local geometry")
    ost = _read(root / "scores" / condition / "openstructure" / "summary.json")
    ost_rows = _index_rows(
        ost.get("rows"), key="target", label=f"{condition} OpenStructure summary"
    )

    manifest_targets = set(manifest_rows)
    local_targets = set(local_rows)
    ost_targets = set(ost_rows)
    if manifest_targets != local_targets or manifest_targets != ost_targets:
        raise ValueError(
            f"{condition}: target mismatch: manifest={sorted(manifest_targets)}, "
            f"local={sorted(local_targets)}, openstructure={sorted(ost_targets)}"
        )

    rows: dict[str, dict[str, float | int]] = {}
    gt_present: bool | None = None
    gt_lt_2p5_total = 0
    for target in sorted(manifest_targets):
        timing = manifest_rows[target]
        local_row = local_rows[target]
        pred = local_row.get("pred")
        if not isinstance(pred, dict):
            raise ValueError(f"{condition}/{target}: missing local pred object")
        summary_row = ost_rows[target]
        raw = _read(root / "scores" / condition / "openstructure" / f"{target}.json")
        # `score_openstructure.py` stores the native OpenStructure JSON, which
        # has no status field on success.  Other scorers may add SUCCESS.
        if raw.get("status") not in (None, "SUCCESS"):
            raise ValueError(f"{condition}/{target}: OpenStructure status={raw.get('status')!r}")
        raw_clashes = raw.get("model_clashes")
        if not isinstance(raw_clashes, list):
            raise ValueError(f"{condition}/{target}: model_clashes must be a list")

        n_atoms = _finite(
            pred.get("n_atoms"), label=f"{condition}/{target} local n_atoms", minimum=1.0
        )
        row: dict[str, float | int] = {
            "length": _count(timing.get("L"), label=f"{condition}/{target} length"),
            "gdt_ts": _finite(summary_row.get("oligo_gdtts"), label=f"{condition}/{target} GDT-TS"),
            "lddt": _finite(summary_row.get("lddt"), label=f"{condition}/{target} lDDT"),
            "tm_score": _finite(
                summary_row.get("tm_score"), label=f"{condition}/{target} TM-score"
            ),
            "bond_p95_A": _finite(
                pred.get("bond_p95_A"),
                label=f"{condition}/{target} bond p95",
                minimum=0.0,
            ),
            "hard_clashes_per_1k_atoms": _finite(
                pred.get("clashes_per_1k_atoms"),
                label=f"{condition}/{target} hard clashes/1k",
                minimum=0.0,
            ),
            "ost_model_clashes_per_1k_atoms": len(raw_clashes) * 1000.0 / n_atoms,
            "nonlocal_ca_segment_penetration_rms_A": _finite(
                pred.get("nonlocal_ca_segment_penetration_rms_A"),
                label=f"{condition}/{target} segment penetration RMS",
                minimum=0.0,
            ),
            "runtime_s": _finite(
                timing.get("runtime_s"),
                label=f"{condition}/{target} runtime",
                minimum=0.0,
            ),
            "peak_vram_gib": _finite(
                timing.get("peak_vram_gib"),
                label=f"{condition}/{target} peak VRAM",
                minimum=0.0,
            ),
            "n_atoms": n_atoms,
            "ost_model_clashes": len(raw_clashes),
        }
        for metric in POINT_COUNT_METRICS + SEGMENT_COUNT_METRICS:
            row[metric] = _count(pred.get(metric), label=f"{condition}/{target} {metric}")

        has_gt = local_row.get("gt") is not None
        if gt_present is None:
            gt_present = has_gt
        elif gt_present != has_gt:
            raise ValueError(f"{condition}: GT coverage is incomplete")
        if has_gt:
            gt = local_row["gt"]
            if not isinstance(gt, dict):
                raise ValueError(f"{condition}/{target}: GT row must be an object")
            gt_lt_2p5_total += _count(
                gt.get("nonlocal_ca_segment_clashes_lt_2p5A"),
                label=f"{condition}/{target} GT segment <2.5 A",
            )
        rows[target] = row

    raw_count = sum(
        (root / "scores" / condition / "openstructure" / f"{target}.json").is_file()
        for target in manifest_targets
    )
    coverage: dict[str, int | bool] = {
        "manifest": len(manifest_rows),
        "local_geometry": len(local_rows),
        "openstructure_summary": len(ost_rows),
        "openstructure_raw": raw_count,
        "complete": raw_count == len(manifest_targets),
        "gt_present": bool(gt_present),
        "gt_segment_lt_2p5_total": gt_lt_2p5_total,
    }
    if gt_present and gt_lt_2p5_total != 0:
        raise ValueError(
            f"{condition}: GT nonlocal C-alpha segment <2.5 A total must be 0, "
            f"got {gt_lt_2p5_total}"
        )
    return rows, manifest, coverage


def _mean(rows: dict[str, dict[str, float | int]], metric: str) -> float:
    return statistics.fmean(float(row[metric]) for row in rows.values())


def _relative_change(control: float, treatment: float) -> float | None:
    if control == 0.0:
        return None
    return treatment / control - 1.0


def _worst_regressions(
    control: dict[str, dict[str, float | int]],
    treatment: dict[str, dict[str, float | int]],
    metric: str,
    *,
    limit: int = 5,
) -> list[dict[str, float | int | str]]:
    values = []
    for target in sorted(control):
        delta = float(treatment[target][metric]) - float(control[target][metric])
        values.append(
            {
                "target": target,
                "length": int(control[target]["length"]),
                "control": float(control[target][metric]),
                "treatment": float(treatment[target][metric]),
                "delta": delta,
            }
        )
    return sorted(values, key=lambda row: (float(row["delta"]), str(row["target"])))[:limit]


def summarize(
    root: Path,
    conditions: list[str] | tuple[str, ...] = DEFAULT_CONDITIONS,
) -> dict[str, Any]:
    """Load, strictly validate, and aggregate generic physics conditions."""
    condition_list = list(conditions)
    if not condition_list or CONTROL not in condition_list:
        raise ValueError(f"conditions must include control {CONTROL!r}")
    if len(set(condition_list)) != len(condition_list):
        raise ValueError("conditions contain duplicates")

    loaded = {condition: _load_condition(root, condition) for condition in condition_list}
    rows = {condition: loaded[condition][0] for condition in condition_list}
    manifests = {condition: loaded[condition][1] for condition in condition_list}
    coverage = {condition: loaded[condition][2] for condition in condition_list}

    control_targets = set(rows[CONTROL])
    for condition in condition_list:
        if set(rows[condition]) != control_targets:
            raise ValueError(f"{condition}: condition target set differs from {CONTROL}")
        for target in control_targets:
            if int(rows[condition][target]["length"]) != int(rows[CONTROL][target]["length"]):
                raise ValueError(f"{condition}/{target}: condition length mismatch")

    control_config = _manifest_config(manifests[CONTROL], condition=CONTROL)
    for condition in condition_list:
        config = _manifest_config(manifests[condition], condition=condition)
        if config != control_config:
            raise ValueError(
                f"{condition}: manifest/guidance config mismatch beyond "
                f"{', '.join(INTENDED_GUIDANCE_FIELDS)}"
            )

    guidance_weights = {
        condition: _guidance_weights(manifests[condition], condition=condition)
        for condition in condition_list
    }
    control_weights = guidance_weights[CONTROL]
    if any(value != 0.0 for value in control_weights.values()):
        raise ValueError(f"{CONTROL}: intended treatment weights must both be zero")
    for condition, weights in guidance_weights.items():
        if condition != CONTROL and not any(value > 0.0 for value in weights.values()):
            raise ValueError(
                f"{condition}: treatment must enable at least one intended guidance weight"
            )

    target_ids = sorted(control_targets)
    aggregates: dict[str, dict[str, Any]] = {}
    worst: dict[str, dict[str, list[dict[str, float | int | str]]]] = {}
    per_target: list[dict[str, Any]] = []

    control_means = {metric: _mean(rows[CONTROL], metric) for metric in MEAN_METRICS}
    control_runtime = sum(float(row["runtime_s"]) for row in rows[CONTROL].values())
    control_vram = max(float(row["peak_vram_gib"]) for row in rows[CONTROL].values())

    for condition in condition_list:
        condition_rows = rows[condition]
        means = {metric: _mean(condition_rows, metric) for metric in MEAN_METRICS}
        runtime_total = sum(float(row["runtime_s"]) for row in condition_rows.values())
        vram_max = max(float(row["peak_vram_gib"]) for row in condition_rows.values())
        aggregates[condition] = {
            "n": len(condition_rows),
            "coverage": coverage[condition],
            "guidance_weights": guidance_weights[condition],
            "means": means,
            "paired_mean_delta_vs_control": {
                metric: means[metric] - control_means[metric] for metric in MEAN_METRICS
            },
            "pooled_nonlocal_ca_point_counts": {
                metric: sum(int(row[metric]) for row in condition_rows.values())
                for metric in POINT_COUNT_METRICS
            },
            "pooled_nonlocal_ca_segment_counts": {
                metric: sum(int(row[metric]) for row in condition_rows.values())
                for metric in SEGMENT_COUNT_METRICS
            },
            "runtime": {
                "total_s": runtime_total,
                "delta_total_s_vs_control": runtime_total - control_runtime,
                "relative_total_change_vs_control": _relative_change(
                    control_runtime, runtime_total
                ),
            },
            "vram": {
                "max_peak_gib": vram_max,
                "delta_max_peak_gib_vs_control": vram_max - control_vram,
            },
        }
        if condition != CONTROL:
            worst[condition] = {
                metric: _worst_regressions(rows[CONTROL], condition_rows, metric)
                for metric in ("gdt_ts", "lddt")
            }

    for target in target_ids:
        condition_values: dict[str, Any] = {}
        for condition in condition_list:
            value = dict(rows[condition][target])
            value["delta_vs_control"] = {
                metric: float(rows[condition][target][metric])
                - float(rows[CONTROL][target][metric])
                for metric in MEAN_METRICS
            }
            condition_values[condition] = value
        per_target.append(
            {
                "target": target,
                "length": int(rows[CONTROL][target]["length"]),
                "conditions": condition_values,
            }
        )

    return {
        "schema_version": 1,
        "experiment": "physics_guidance_casp14",
        "control_condition": CONTROL,
        "conditions": condition_list,
        "target_count": len(target_ids),
        "target_ids": target_ids,
        "coverage": {
            "n": len(target_ids),
            "paired_complete": all(
                bool(coverage[condition]["complete"])
                and int(coverage[condition]["manifest"]) == len(target_ids)
                for condition in condition_list
            ),
            "by_condition": coverage,
        },
        "gt_segment_validation": {
            "required_when_present": True,
            "threshold_A": 2.5,
            "all_present_gt_totals_zero": all(
                not bool(coverage[condition]["gt_present"])
                or int(coverage[condition]["gt_segment_lt_2p5_total"]) == 0
                for condition in condition_list
            ),
        },
        "aggregates": aggregates,
        "worst_target_regressions": worst,
        "rows": per_target,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _value_and_delta(row: dict[str, Any], metric: str) -> str:
    value = _fmt(row["means"][metric])
    delta = float(row["paired_mean_delta_vs_control"][metric])
    return f"{value} ({delta:+.4f})"


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a compact audit table; parenthesized values are paired deltas."""
    lines = [
        "# CASP14 physics-guidance comparison",
        "",
        f"Control: `{summary['control_condition']}`. Paired coverage: "
        f"**{summary['target_count']}/{summary['target_count']}** targets across "
        f"{len(summary['conditions'])} conditions. Parentheses report treatment minus "
        "control; lower is better for geometry/clash metrics.",
        "",
        "## Accuracy, local validity, and resources",
        "",
        "| Condition | n | GDT-TS (Δ) | lDDT (Δ) | TM (Δ) | Bond p95 Å (Δ) | "
        "Hard clash/1k (Δ) | OST clash/1k (Δ) | Segment penetration RMS Å (Δ) | "
        "Time s (Δ, %) | Peak GiB (Δ) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in summary["conditions"]:
        row = summary["aggregates"][condition]
        runtime = row["runtime"]
        vram = row["vram"]
        relative = runtime["relative_total_change_vs_control"]
        relative_text = "NA" if relative is None else f"{100.0 * relative:+.1f}%"
        lines.append(
            f"| {condition} | {row['n']} | {_value_and_delta(row, 'gdt_ts')} | "
            f"{_value_and_delta(row, 'lddt')} | {_value_and_delta(row, 'tm_score')} | "
            f"{_value_and_delta(row, 'bond_p95_A')} | "
            f"{_value_and_delta(row, 'hard_clashes_per_1k_atoms')} | "
            f"{_value_and_delta(row, 'ost_model_clashes_per_1k_atoms')} | "
            f"{_value_and_delta(row, 'nonlocal_ca_segment_penetration_rms_A')} | "
            f"{_fmt(runtime['total_s'])} ({runtime['delta_total_s_vs_control']:+.2f}, "
            f"{relative_text}) | {_fmt(vram['max_peak_gib'])} "
            f"({vram['delta_max_peak_gib_vs_control']:+.4f}) |"
        )

    lines.extend(
        [
            "",
            "## Pooled nonlocal Cα close contacts",
            "",
            "| Condition | Point <2 Å | Point <3 Å | Point <3.6 Å | "
            "Segment <2 Å | Segment <2.5 Å | Segment <3 Å |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in summary["conditions"]:
        row = summary["aggregates"][condition]
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

    lines.extend(["", "## Worst per-target accuracy deltas", ""])
    for condition in summary["conditions"]:
        if condition == summary["control_condition"]:
            continue
        lines.extend(
            [
                f"### {condition}",
                "",
                "| Metric | Target | L | Control | Treatment | Δ |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for metric in ("gdt_ts", "lddt"):
            for row in summary["worst_target_regressions"][condition][metric]:
                lines.append(
                    f"| {metric} | {row['target']} | {row['length']} | "
                    f"{_fmt(row['control'])} | {_fmt(row['treatment'])} | "
                    f"{float(row['delta']):+.4f} |"
                )
        lines.append("")

    gt = summary["gt_segment_validation"]
    lines.extend(
        [
            "GT segment sanity check (<2.5 Å): "
            f"**{'PASS' if gt['all_present_gt_totals_zero'] else 'FAIL'}**. "
            "OpenStructure clash rates use each prediction's local-geometry atom count.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(DEFAULT_CONDITIONS),
        help="Control steric_1 followed by any physics treatments to compare",
    )
    args = parser.parse_args()
    summary = summarize(args.root, args.conditions)
    json_path = args.root / "physics_comparison.json"
    markdown_path = args.root / "physics_comparison.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    markdown_path.write_text(render_markdown(summary))
    print(json.dumps(summary["aggregates"], indent=2))


if __name__ == "__main__":
    main()
