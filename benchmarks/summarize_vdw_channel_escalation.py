#!/usr/bin/env python3
"""Strict held-out report for the independent-VDW channel escalation.

The three targets used to choose the dose are reported separately from the
five held-out targets.  Only the held-out split can affect the decision.
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
TREATMENT = "steric_1_vdw_sep_s0p10_e1"
DEFAULT_CONDITIONS = (CONTROL, TREATMENT)

SELECTION_TARGETS = ("t1036s1", "t1040", "t1096")
HELDOUT_TARGETS = ("t1052", "t1068", "t1039", "t1041", "t1061")
FULL_TARGETS = (*SELECTION_TARGETS, *HELDOUT_TARGETS)

DIAGNOSTIC_VDW_METRIC = "final_vdw_loss_tol_0p6"
DIRECT_VDW_METRIC = "final_vdw_loss_tol_1p5"
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

EXPECTED_CHECKPOINT_SHA256 = "8b3f8667cbcd59f12c62c8a08b54b263233d2b8cf618d9617482ade81d1b973f"
EXPECTED_GPU = "NVIDIA RTX 6000 Ada Generation"
EXPECTED_DTYPE = "bfloat16"
EXPECTED_DOSES = {CONTROL: (0.0, 8), TREATMENT: (0.10, 1)}
VARIED_GUIDANCE_FIELDS = ("vdw_scale", "vdw_every_n_steps")
_FIXED_GUIDANCE = {
    "all_atom_clash_weight": 0.0,
    "vdw_start": 0.65,
    "vdw_overlap_tolerance_A": 1.5,
    "vdw_max_step_A": 0.01,
}
_FIXED_MANIFEST = {
    "sampler": "sde",
    "n_steps": 500,
    "seed": 0,
    "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
    "cuda_device_name": EXPECTED_GPU,
    "autocast_dtype": EXPECTED_DTYPE,
    "sde_tau": 0.01,
    "sde_eps": 0.01,
    "sde_w_cutoff": 0.99,
    "sde_log_timesteps": True,
}
_EPS = 1e-12


def _guidance(manifest: dict[str, Any], condition: str) -> dict[str, Any]:
    value = manifest.get("geometry_guidance")
    if not isinstance(value, dict):
        raise ValueError(f"{condition}: missing geometry_guidance object")
    return value


def _validate_manifest(manifest: dict[str, Any], condition: str) -> dict[str, float | int]:
    """Require the frozen inference contract and the exact independent-VDW dose."""
    manifest_condition = manifest.get("condition")
    if manifest_condition not in (None, condition):
        raise ValueError(f"{condition}: manifest condition is {manifest_condition!r}")

    for field, expected in _FIXED_MANIFEST.items():
        value = manifest.get(field)
        if value != expected:
            raise ValueError(f"{condition}: {field} must be exactly {expected!r}, got {value!r}")

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
            f"{condition}: expected independent VDW dose vdw_scale={expected_scale}, "
            f"vdw_every_n_steps={expected_every}; got {scale}, {every}"
        )

    for field, expected in _FIXED_GUIDANCE.items():
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
) -> tuple[dict[str, dict[str, float | int]], dict[str, Any], dict[str, int | bool]]:
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


def _relative_change(control: float, treatment: float) -> float | None:
    if control <= 0.0:
        return None
    return treatment / control - 1.0


def _check(value: Any, threshold: str, passed: bool) -> dict[str, Any]:
    return {"value": value, "threshold": threshold, "pass": bool(passed)}


def _split_rows(
    rows: dict[str, dict[str, dict[str, float | int]]], target_ids: tuple[str, ...]
) -> dict[str, dict[str, dict[str, float | int]]]:
    return {
        condition: {target: rows[condition][target] for target in target_ids}
        for condition in DEFAULT_CONDITIONS
    }


def _aggregate_split(
    rows: dict[str, dict[str, dict[str, float | int]]], target_ids: tuple[str, ...]
) -> dict[str, Any]:
    split_rows = _split_rows(rows, target_ids)
    control_means = {metric: _mean(split_rows[CONTROL], metric) for metric in MEAN_METRICS}
    aggregates: dict[str, dict[str, Any]] = {}
    for condition in DEFAULT_CONDITIONS:
        condition_rows = split_rows[condition]
        means = {metric: _mean(condition_rows, metric) for metric in MEAN_METRICS}
        aggregates[condition] = {
            "n": len(target_ids),
            "means": means,
            "paired_mean_delta_vs_control": {
                metric: means[metric] - control_means[metric] for metric in MEAN_METRICS
            },
            "relative_change_vs_control": {
                "runtime_s": _relative_change(control_means["runtime_s"], means["runtime_s"]),
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
                    int(row["nonlocal_ca_clashes_lt_2A"]) for row in condition_rows.values()
                ),
                "nonlocal_ca_segment_clashes_lt_2p5A": sum(
                    int(row["nonlocal_ca_segment_clashes_lt_2p5A"])
                    for row in condition_rows.values()
                ),
            },
        }

    per_target = []
    for target in target_ids:
        conditions: dict[str, Any] = {}
        for condition in DEFAULT_CONDITIONS:
            values = {metric: split_rows[condition][target][metric] for metric in MEAN_METRICS}
            values.update(
                {
                    "nonlocal_ca_clashes_lt_2A": split_rows[condition][target][
                        "nonlocal_ca_clashes_lt_2A"
                    ],
                    "nonlocal_ca_segment_clashes_lt_2p5A": split_rows[condition][target][
                        "nonlocal_ca_segment_clashes_lt_2p5A"
                    ],
                }
            )
            conditions[condition] = {
                "values": values,
                "delta_vs_control": {
                    metric: float(split_rows[condition][target][metric])
                    - float(split_rows[CONTROL][target][metric])
                    for metric in MEAN_METRICS
                }
                | {
                    "nonlocal_ca_clashes_lt_2A": int(
                        split_rows[condition][target]["nonlocal_ca_clashes_lt_2A"]
                    )
                    - int(split_rows[CONTROL][target]["nonlocal_ca_clashes_lt_2A"]),
                    "nonlocal_ca_segment_clashes_lt_2p5A": int(
                        split_rows[condition][target]["nonlocal_ca_segment_clashes_lt_2p5A"]
                    )
                    - int(split_rows[CONTROL][target]["nonlocal_ca_segment_clashes_lt_2p5A"]),
                },
            }
        per_target.append(
            {
                "target": target,
                "length": int(split_rows[CONTROL][target]["length"]),
                "conditions": conditions,
            }
        )
    return {"aggregates": aggregates, "rows": per_target}


def _heldout_gates(
    split: dict[str, Any], *, coverage_complete: bool, gt_sane: bool
) -> dict[str, Any]:
    aggregates = split["aggregates"]
    control = aggregates[CONTROL]
    treatment = aggregates[TREATMENT]
    deltas = treatment["paired_mean_delta_vs_control"]
    direct_reduction = treatment["relative_reduction_vs_control"][DIRECT_VDW_METRIC]
    ost_reduction = treatment["relative_reduction_vs_control"]["ost_model_clashes_per_1k_atoms"]
    runtime_relative = treatment["relative_change_vs_control"]["runtime_s"]

    target_gdt_deltas = {
        row["target"]: row["conditions"][TREATMENT]["delta_vs_control"]["gdt_ts"]
        for row in split["rows"]
    }
    target_lddt_deltas = {
        row["target"]: row["conditions"][TREATMENT]["delta_vs_control"]["lddt"]
        for row in split["rows"]
    }
    worst_gdt = min(float(value) for value in target_gdt_deltas.values())
    worst_lddt = min(float(value) for value in target_lddt_deltas.values())
    ca_lt_2 = int(treatment["pooled_counts"]["nonlocal_ca_clashes_lt_2A"])
    segment_delta = int(treatment["pooled_counts"]["nonlocal_ca_segment_clashes_lt_2p5A"]) - int(
        control["pooled_counts"]["nonlocal_ca_segment_clashes_lt_2p5A"]
    )

    checks = {
        "coverage_complete_5_of_5": _check(
            coverage_complete,
            "both conditions complete on exactly 5 held-out targets",
            coverage_complete,
        ),
        "gt_segment_sanity": _check(
            gt_sane, "complete held-out GT and GT segment <2.5 A count == 0", gt_sane
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
            **_check(worst_gdt, ">= -0.005", worst_gdt + _EPS >= -0.005),
            "per_target": target_gdt_deltas,
        },
        "lddt_each_target_delta": {
            **_check(worst_lddt, ">= -0.005", worst_lddt + _EPS >= -0.005),
            "per_target": target_lddt_deltas,
        },
        "bond_p95_mean_delta_A": _check(
            deltas["bond_p95_A"],
            "<= 0.001 A",
            float(deltas["bond_p95_A"]) <= 0.001 + _EPS,
        ),
        "nonlocal_ca_lt_2A": _check(ca_lt_2, "== 0", ca_lt_2 == 0),
        "nonlocal_ca_segment_lt_2p5_delta": _check(segment_delta, "<= 0", segment_delta <= 0),
        "runtime_relative_change": _check(
            runtime_relative,
            "<= 0.15",
            runtime_relative is not None and runtime_relative <= 0.15 + _EPS,
        ),
    }
    return {"all_pass": all(bool(check["pass"]) for check in checks.values()), "checks": checks}


def summarize(
    root: Path, conditions: list[str] | tuple[str, ...] = DEFAULT_CONDITIONS
) -> dict[str, Any]:
    """Load and gate the exact eight-target escalation, using held-out targets only."""
    condition_list = list(conditions)
    if condition_list != list(DEFAULT_CONDITIONS):
        raise ValueError("conditions must be exactly, in order: " + ", ".join(DEFAULT_CONDITIONS))

    loaded = {condition: _load(root, condition) for condition in condition_list}
    rows = {condition: loaded[condition][0] for condition in condition_list}
    manifests = {condition: loaded[condition][1] for condition in condition_list}
    coverage = {condition: loaded[condition][2] for condition in condition_list}

    expected_targets = set(FULL_TARGETS)
    for condition in condition_list:
        actual_targets = set(rows[condition])
        if actual_targets != expected_targets:
            missing = sorted(expected_targets - actual_targets)
            extra = sorted(actual_targets - expected_targets)
            raise ValueError(
                f"{condition}: target split must be exact selection3+heldout5; "
                f"missing={missing}, extra={extra}"
            )
        for target in FULL_TARGETS:
            if int(rows[condition][target]["length"]) != int(rows[CONTROL][target]["length"]):
                raise ValueError(f"{condition}/{target}: condition length mismatch")

    doses = {
        condition: _validate_manifest(manifests[condition], condition)
        for condition in condition_list
    }
    control_config = _comparison_config(manifests[CONTROL], CONTROL)
    if _comparison_config(manifests[TREATMENT], TREATMENT) != control_config:
        raise ValueError(
            f"{TREATMENT}: manifest/guidance config mismatch beyond "
            f"{', '.join(VARIED_GUIDANCE_FIELDS)}"
        )

    coverage_complete = all(
        bool(coverage[condition]["complete"])
        and int(coverage[condition]["manifest"]) == len(FULL_TARGETS)
        and int(coverage[condition]["local_geometry"]) == len(FULL_TARGETS)
        and int(coverage[condition]["openstructure_summary"]) == len(FULL_TARGETS)
        and int(coverage[condition]["openstructure_raw"]) == len(FULL_TARGETS)
        for condition in condition_list
    )
    if not coverage_complete:
        raise ValueError("incomplete paired artifact coverage; exact 8/8 is required")
    gt_sane = all(
        bool(coverage[condition]["gt_present"])
        and int(coverage[condition]["gt_segment_lt_2p5_total"]) == 0
        for condition in condition_list
    )
    if not gt_sane:
        raise ValueError("complete GT with zero segment <2.5 A events is required")

    selection = _aggregate_split(rows, SELECTION_TARGETS)
    heldout = _aggregate_split(rows, HELDOUT_TARGETS)
    full = _aggregate_split(rows, FULL_TARGETS)
    gates = _heldout_gates(heldout, coverage_complete=coverage_complete, gt_sane=gt_sane)
    status = (
        "candidate_for_multiseed_confirmation"
        if gates["all_pass"]
        else "abandon_channel_only_escalation"
    )

    return {
        "schema_version": 1,
        "experiment": "independent_vdw_channel_escalation_heldout",
        "control_condition": CONTROL,
        "treatment_condition": TREATMENT,
        "conditions": condition_list,
        "frozen_contract": {
            **_FIXED_MANIFEST,
            "doses": doses,
            "legacy_all_atom_clash_weight": 0.0,
            "vdw_start": 0.65,
            "vdw_overlap_tolerance_A": 1.5,
            "vdw_max_step_A": 0.01,
        },
        "vdw_metric_roles": {
            DIAGNOSTIC_VDW_METRIC: "reported_diagnostic",
            DIRECT_VDW_METRIC: "heldout_primary_gate",
        },
        "coverage": {
            "paired_complete_8_of_8": coverage_complete,
            "gt_segment_sanity": gt_sane,
            "by_condition": coverage,
        },
        "splits": {
            "selection": {
                "role": "reported_only_never_gated",
                "target_count": len(SELECTION_TARGETS),
                "target_ids": list(SELECTION_TARGETS),
                **selection,
            },
            "heldout": {
                "role": "sole_preregistered_gate_split",
                "target_count": len(HELDOUT_TARGETS),
                "target_ids": list(HELDOUT_TARGETS),
                **heldout,
            },
            "full": {
                "role": "reported_only_never_gated",
                "target_count": len(FULL_TARGETS),
                "target_ids": list(FULL_TARGETS),
                **full,
            },
        },
        "heldout_gates": gates,
        "decision": {
            "gate_scope": "heldout_only",
            "selection_metrics_used_for_gate": False,
            "full_metrics_used_for_gate": False,
            "status": status,
            "claim_limit": (
                "candidate requires multiseed confirmation; this single-seed held-out "
                "experiment is not a promotion claim"
            ),
        },
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
    value = _fmt(row["means"][metric])
    delta = float(row["paired_mean_delta_vs_control"][metric])
    return f"{value} ({delta:+.4f})"


def render_markdown(summary: dict[str, Any]) -> str:
    """Render selection, held-out, and full results without mixing gate scope."""
    lines = [
        "# Independent-VDW channel escalation",
        "",
        "The **five-target held-out split is the only gate**. The three dose-selection "
        "targets and the combined eight-target view are descriptive only. This is a "
        "single-seed SDE-500 result; even a pass is only a multiseed-confirmation candidate.",
        "",
    ]
    for split_name in ("selection", "heldout", "full"):
        split = summary["splits"][split_name]
        lines.extend(
            [
                f"## {split_name.capitalize()} ({split['target_count']} targets)",
                "",
                f"Role: `{split['role']}`. Targets: `{', '.join(split['target_ids'])}`.",
                "",
                "| Condition | GDT-TS (Δ) | lDDT (Δ) | TM (Δ) | Bond p95 Å (Δ) | "
                "Hard clash/1k (Δ) | OST clash/1k (Δ) | Runtime s (Δ) | "
                "VDW 0.6 Å (Δ) | VDW 1.5 Å (Δ) | Cα<2 Å | Segment<2.5 Å |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for condition in summary["conditions"]:
            row = split["aggregates"][condition]
            counts = row["pooled_counts"]
            lines.append(
                f"| {condition} | {_mean_delta(row, 'gdt_ts')} | "
                f"{_mean_delta(row, 'lddt')} | {_mean_delta(row, 'tm_score')} | "
                f"{_mean_delta(row, 'bond_p95_A')} | "
                f"{_mean_delta(row, 'hard_clashes_per_1k_atoms')} | "
                f"{_mean_delta(row, 'ost_model_clashes_per_1k_atoms')} | "
                f"{_mean_delta(row, 'runtime_s')} | "
                f"{_mean_delta(row, DIAGNOSTIC_VDW_METRIC)} | "
                f"{_mean_delta(row, DIRECT_VDW_METRIC)} | "
                f"{counts['nonlocal_ca_clashes_lt_2A']} | "
                f"{counts['nonlocal_ca_segment_clashes_lt_2p5A']} |"
            )
        lines.append("")

    checks = summary["heldout_gates"]["checks"]
    lines.extend(
        [
            "## Held-out-only gates",
            "",
            "| Gate | Value | Threshold | Result |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, check in checks.items():
        lines.append(
            f"| {name} | {_fmt(check['value'])} | {check['threshold']} | "
            f"{'PASS' if check['pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"Decision: **{summary['decision']['status']}**. Selection and full-set "
            "metrics were not used by any gate.",
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
        help="Exact control and independent-VDW escalation treatment",
    )
    args = parser.parse_args()
    summary = summarize(args.root, args.conditions)
    json_path = args.root / "vdw_escalation_comparison.json"
    markdown_path = args.root / "vdw_escalation_comparison.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    markdown_path.write_text(render_markdown(summary))
    print(json.dumps(summary["decision"], indent=2))


if __name__ == "__main__":
    main()
