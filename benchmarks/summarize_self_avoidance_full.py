#!/usr/bin/env python3
"""Summarize the frozen full-CASP14 self-avoidance confirmation experiment.

The steric scale was selected on two known-failure targets.  This report keeps
those targets separate from the remaining confirmatory targets while also
providing a full-set deployment summary.  It performs no structure scoring of
its own: all inputs come from the existing local-geometry and OpenStructure
artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    from benchmarks.summarize_stereochemical_examples import _condition_rows
except ModuleNotFoundError:  # direct ``python benchmarks/...py`` execution
    from summarize_stereochemical_examples import _condition_rows


CONTROL = "split_local_control"
GUIDED = "steric_1"
CONDITIONS = (CONTROL, GUIDED)
EXPECTED_FULL_COUNT = 70
EXPECTED_TUNING_COUNT = 2
EXPECTED_HELDOUT_COUNT = 68

MEAN_METRICS = (
    "gdt_ts",
    "lddt",
    "bond_p95_A",
    "hard_clashes_per_1k_atoms",
    "nonlocal_ca_penetration_rms_A",
    "ost_clashes_per_1k_atoms",
    "ost_bad_bonds_per_1k_atoms",
    "ost_bad_angles_per_1k_atoms",
    "ca_chirality_wrong_frac",
)
COUNT_METRICS = (
    "nonlocal_ca_clashes_lt_2A",
    "nonlocal_ca_clashes_lt_3A",
    "nonlocal_ca_clashes_lt_3p6A",
)
LOWER_IS_BETTER = set(MEAN_METRICS) - {"gdt_ts", "lddt"}
LOWER_IS_BETTER.update(COUNT_METRICS)

CONTRACT = {
    "confirmatory_subset": "full CASP14 single-chain set minus the two tuning targets",
    "expected_counts": {
        "full": EXPECTED_FULL_COUNT,
        "tuning": EXPECTED_TUNING_COUNT,
        "heldout": EXPECTED_HELDOUT_COUNT,
    },
    "primary_efficacy": {
        "pooled_nonlocal_ca_lt_3p6A_reduction_fraction_min": 0.75,
        "paired_target_bootstrap_95pct_ci_lower_min": 0.50,
        "guided_nonlocal_ca_lt_2A_total_max": 0,
        "equal_target_penetration_rms_delta_max_A": 0.0,
    },
    "accuracy_guardrails": {
        "mean_gdt_ts_delta_min": -0.005,
        "mean_lddt_delta_min": -0.005,
        "paired_bootstrap_95pct_ci_lower_min": -0.01,
        "single_target_gdt_ts_delta_min": -0.10,
        "single_target_lddt_delta_min": -0.10,
    },
    "validity_guardrails": {
        "mean_bond_p95_delta_max_A": 0.01,
        "relative_rate_increase_max": 0.10,
        "relative_rate_metrics": [
            "hard_clashes_per_1k_atoms",
            "ost_clashes_per_1k_atoms",
            "ost_bad_bonds_per_1k_atoms",
            "ost_bad_angles_per_1k_atoms",
        ],
        "wrong_ca_chirality_increase_max": 0.0,
        "zero_control_rule": "a zero control rate permits no new guided event",
    },
    "operational": {
        "complete_paired_coverage_required": True,
        "runtime_overhead_25pct_is_informational_unless_order_counterbalanced": True,
    },
}


def _read(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"missing required artifact: {path}") from exc


def read_tuning_ids(path: Path) -> list[str]:
    """Read exactly two unique tuning IDs from a whitespace-delimited file."""
    try:
        values = [value.strip().lower() for value in path.read_text().split() if value.strip()]
    except FileNotFoundError as exc:
        raise ValueError(f"missing tuning ID file: {path}") from exc
    if len(values) != EXPECTED_TUNING_COUNT or len(set(values)) != len(values):
        raise ValueError(f"expected two unique tuning IDs, got {values}")
    return values


def _unique_rows(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        target = str(row[key]).lower()
        if target in indexed:
            raise ValueError(f"{label}: duplicate target {target}")
        indexed[target] = row
    return indexed


def _positive_float(value: Any, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label}: expected a positive finite value, got {value!r}")
    return number


def _nonnegative_float(value: Any, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label}: expected a non-negative finite value, got {value!r}")
    return number


def _load_condition(
    root: Path, condition: str
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    base_rows = _condition_rows(root, condition)
    manifest = _read(root / "inference" / condition / "manifest.json")
    if manifest.get("condition") not in (None, condition):
        raise ValueError(f"{condition}: manifest condition is {manifest.get('condition')!r}")
    timing = _unique_rows(manifest.get("rows", []), "pdb_id", f"{condition} manifest")

    local = _read(root / "scores" / condition / "local_geometry.json")
    local_rows = _unique_rows(local.get("rows", []), "pdb_id", f"{condition} local score")
    targets = set(base_rows)
    if targets != set(timing) or targets != set(local_rows):
        raise ValueError(
            f"{condition}: target mismatch among combined scores, manifest, and local scores"
        )

    rows: dict[str, dict[str, float]] = {}
    for target in sorted(targets):
        row = {key: float(value) for key, value in base_rows[target].items()}
        raw = _read(root / "scores" / condition / "openstructure" / f"{target}.json")
        if raw.get("status") not in (None, "SUCCESS"):
            raise ValueError(f"{condition}/{target}: OpenStructure status={raw.get('status')!r}")
        local_pred = local_rows[target].get("pred")
        if not isinstance(local_pred, dict):
            raise ValueError(f"{condition}/{target}: missing local pred row")

        atom_source = "openstructure"
        atom_count = raw.get("n_atoms", raw.get("model_n_atoms"))
        if atom_count is None:
            atom_source = "local_geometry"
            atom_count = local_pred.get("n_atoms")
        n_atoms = _positive_float(atom_count, label=f"{condition}/{target} n_atoms")
        if "n_ca_chiral_centres" not in local_pred:
            raise ValueError(f"{condition}/{target}: missing chiral-centre count")
        n_chiral = int(local_pred["n_ca_chiral_centres"])
        if n_chiral < 0:
            raise ValueError(f"{condition}/{target}: negative chiral-centre count")
        wrong_chiral = int(round(float(row["ca_chirality_wrong_frac"]) * n_chiral))

        row.update(
            {
                "n_atoms": n_atoms,
                "ost_clashes_per_1k_atoms": float(row["ost_clashes"]) * 1000.0 / n_atoms,
                "ost_bad_bonds_per_1k_atoms": float(row["ost_bad_bonds"]) * 1000.0 / n_atoms,
                "ost_bad_angles_per_1k_atoms": float(row["ost_bad_angles"]) * 1000.0 / n_atoms,
                "runtime_s": _positive_float(
                    timing[target].get("runtime_s"), label=f"{condition}/{target} runtime"
                ),
                "peak_vram_gib": _nonnegative_float(
                    timing[target].get("peak_vram_gib"),
                    label=f"{condition}/{target} peak VRAM",
                ),
                "ca_chirality_wrong_count": wrong_chiral,
                "atom_count_source": atom_source,
            }
        )
        for metric in MEAN_METRICS + COUNT_METRICS:
            value = float(row[metric])
            if not math.isfinite(value):
                raise ValueError(f"{condition}/{target}: non-finite {metric}={value}")
        rows[target] = row
    return rows, manifest


def _validate_pair(
    rows: dict[str, dict[str, dict[str, float]]], manifests: dict[str, dict[str, Any]]
) -> list[str]:
    control_targets = set(rows[CONTROL])
    guided_targets = set(rows[GUIDED])
    if control_targets != guided_targets:
        raise ValueError("condition target mismatch")
    if len(control_targets) != EXPECTED_FULL_COUNT:
        raise ValueError(
            f"expected {EXPECTED_FULL_COUNT} full-set targets, got {len(control_targets)}"
        )
    for target in control_targets:
        if int(rows[CONTROL][target]["length"]) != int(rows[GUIDED][target]["length"]):
            raise ValueError(f"{target}: condition length mismatch")

    control_cfg = dict(manifests[CONTROL].get("geometry_guidance") or {})
    guided_cfg = dict(manifests[GUIDED].get("geometry_guidance") or {})
    if float(control_cfg.get("steric_scale", float("nan"))) != 0.0:
        raise ValueError(f"{CONTROL}: expected steric_scale=0")
    if float(guided_cfg.get("steric_scale", float("nan"))) != 1.0:
        raise ValueError(f"{GUIDED}: expected steric_scale=1")
    control_cfg.pop("steric_scale")
    guided_cfg.pop("steric_scale")
    if control_cfg != guided_cfg:
        raise ValueError("guidance configs differ beyond steric_scale")

    paired_manifest_keys = (
        "checkpoint",
        "checkpoint_sha256",
        "sampler",
        "n_steps",
        "seed",
        "sde_tau",
        "sde_eps",
        "sde_w_cutoff",
        "sde_log_timesteps",
    )
    for key in paired_manifest_keys:
        if manifests[CONTROL].get(key) != manifests[GUIDED].get(key):
            raise ValueError(f"condition manifest mismatch for {key}")
    return sorted(control_targets)


def _mean(rows: dict[str, dict[str, float]], targets: list[str], metric: str) -> float:
    return statistics.fmean(float(rows[target][metric]) for target in targets)


def _condition_aggregate(rows: dict[str, dict[str, float]], targets: list[str]) -> dict[str, Any]:
    return {
        "n": len(targets),
        "means": {metric: _mean(rows, targets, metric) for metric in MEAN_METRICS},
        "pooled_nonlocal_ca_counts": {
            metric: sum(int(rows[target][metric]) for target in targets) for metric in COUNT_METRICS
        },
        "pooled_ca_chirality_wrong_count": sum(
            int(rows[target]["ca_chirality_wrong_count"]) for target in targets
        ),
        "runtime": {
            "total_s": sum(float(rows[target]["runtime_s"]) for target in targets),
            "mean_s": _mean(rows, targets, "runtime_s"),
        },
        "vram": {
            "mean_peak_gib": _mean(rows, targets, "peak_vram_gib"),
            "max_peak_gib": max(float(rows[target]["peak_vram_gib"]) for target in targets),
        },
        "atom_count_sources": dict(
            Counter(str(rows[target]["atom_count_source"]) for target in targets)
        ),
    }


def _percentile_ci(values: np.ndarray) -> list[float] | None:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None
    low, high = np.percentile(finite, [2.5, 97.5])
    return [float(low), float(high)]


def _relative_change(control: float, guided: float) -> float | None:
    if abs(control) <= 1e-12:
        return None
    return guided / control - 1.0


def _paired_subset(
    rows: dict[str, dict[str, dict[str, float]]],
    targets: list[str],
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(targets), size=(bootstrap, len(targets)))
    deltas: dict[str, dict[str, Any]] = {}
    target_deltas: dict[str, np.ndarray] = {}

    for metric in MEAN_METRICS:
        paired = np.asarray(
            [rows[GUIDED][target][metric] - rows[CONTROL][target][metric] for target in targets],
            dtype=np.float64,
        )
        target_deltas[metric] = paired
        entry: dict[str, Any] = {"guided_minus_control": float(paired.mean())}
        if metric in {"gdt_ts", "lddt"}:
            entry["paired_target_bootstrap_95pct_ci"] = _percentile_ci(paired[indices].mean(axis=1))
        deltas[metric] = entry

    for metric in COUNT_METRICS:
        paired = np.asarray(
            [rows[GUIDED][target][metric] - rows[CONTROL][target][metric] for target in targets],
            dtype=np.float64,
        )
        target_deltas[metric] = paired
        deltas[metric] = {"guided_minus_control_mean": float(paired.mean())}

    control_counts = np.asarray(
        [rows[CONTROL][target]["nonlocal_ca_clashes_lt_3p6A"] for target in targets],
        dtype=np.float64,
    )
    guided_counts = np.asarray(
        [rows[GUIDED][target]["nonlocal_ca_clashes_lt_3p6A"] for target in targets],
        dtype=np.float64,
    )
    control_total = float(control_counts.sum())
    reduction = None if control_total == 0.0 else 1.0 - float(guided_counts.sum()) / control_total
    sampled_control = control_counts[indices].sum(axis=1)
    sampled_guided = guided_counts[indices].sum(axis=1)
    sampled_reduction = np.full(bootstrap, np.nan, dtype=np.float64)
    nonzero = sampled_control > 0.0
    sampled_reduction[nonzero] = 1.0 - sampled_guided[nonzero] / sampled_control[nonzero]

    runtime_control = np.asarray(
        [rows[CONTROL][target]["runtime_s"] for target in targets], dtype=np.float64
    )
    runtime_guided = np.asarray(
        [rows[GUIDED][target]["runtime_s"] for target in targets], dtype=np.float64
    )
    vram_control = np.asarray(
        [rows[CONTROL][target]["peak_vram_gib"] for target in targets], dtype=np.float64
    )
    vram_guided = np.asarray(
        [rows[GUIDED][target]["peak_vram_gib"] for target in targets], dtype=np.float64
    )

    worst: dict[str, list[dict[str, float | str]]] = {}
    for metric, paired in target_deltas.items():
        order = np.argsort(paired) if metric not in LOWER_IS_BETTER else np.argsort(-paired)
        worst[metric] = [
            {
                "target": targets[int(index)],
                "control": float(rows[CONTROL][targets[int(index)]][metric]),
                "guided": float(rows[GUIDED][targets[int(index)]][metric]),
                "delta": float(paired[int(index)]),
            }
            for index in order[:5]
        ]

    return {
        "deltas": deltas,
        "nonlocal_ca_clashes_lt_3p6A_reduction": {
            "fraction": reduction,
            "paired_target_bootstrap_95pct_ci": _percentile_ci(sampled_reduction),
        },
        "runtime": {
            "guided_minus_control_mean_s": float((runtime_guided - runtime_control).mean()),
            "relative_total_change": _relative_change(
                float(runtime_control.sum()), float(runtime_guided.sum())
            ),
        },
        "vram": {
            "guided_minus_control_mean_peak_gib": float((vram_guided - vram_control).mean()),
            "guided_minus_control_max_peak_gib": float(vram_guided.max() - vram_control.max()),
        },
        "worst_target_regressions": worst,
    }


def _subset_summary(
    name: str,
    role: str,
    rows: dict[str, dict[str, dict[str, float]]],
    targets: list[str],
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    coverage = {
        "expected": len(targets),
        CONTROL: sum(target in rows[CONTROL] for target in targets),
        GUIDED: sum(target in rows[GUIDED] for target in targets),
        "paired": sum(target in rows[CONTROL] and target in rows[GUIDED] for target in targets),
    }
    coverage["complete"] = all(coverage[key] == len(targets) for key in CONDITIONS)
    return {
        "name": name,
        "role": role,
        "n": len(targets),
        "target_ids": targets,
        "coverage": coverage,
        "conditions": {
            condition: _condition_aggregate(rows[condition], targets) for condition in CONDITIONS
        },
        "paired": _paired_subset(rows, targets, bootstrap=bootstrap, seed=seed),
    }


def _criterion(
    observed: Any,
    threshold: Any,
    operator: str,
    passed: bool,
) -> dict[str, Any]:
    return {
        "observed": observed,
        "threshold": threshold,
        "operator": operator,
        "passed": bool(passed),
        "required": True,
    }


def _rate_guardrail(
    control: float, guided: float, *, maximum_relative_increase: float
) -> tuple[float | None, bool]:
    relative = _relative_change(control, guided)
    if relative is None:
        return None, guided <= 1e-12
    return relative, relative <= maximum_relative_increase


def _heldout_decision(subset: dict[str, Any]) -> dict[str, Any]:
    control = subset["conditions"][CONTROL]
    guided = subset["conditions"][GUIDED]
    paired = subset["paired"]
    reduction = paired["nonlocal_ca_clashes_lt_3p6A_reduction"]
    reduction_ci = reduction["paired_target_bootstrap_95pct_ci"]
    gdt = paired["deltas"]["gdt_ts"]
    lddt = paired["deltas"]["lddt"]
    gdt_worst = min(row["delta"] for row in paired["worst_target_regressions"]["gdt_ts"])
    lddt_worst = min(row["delta"] for row in paired["worst_target_regressions"]["lddt"])

    criteria: dict[str, dict[str, Any]] = {
        "expected_heldout_count": _criterion(
            subset["n"], EXPECTED_HELDOUT_COUNT, "==", subset["n"] == EXPECTED_HELDOUT_COUNT
        ),
        "complete_paired_coverage": _criterion(
            subset["coverage"]["complete"], True, "==", subset["coverage"]["complete"]
        ),
        "ca_lt3p6_reduction": _criterion(
            reduction["fraction"],
            0.75,
            ">=",
            reduction["fraction"] is not None and reduction["fraction"] >= 0.75,
        ),
        "ca_lt3p6_reduction_ci_lower": _criterion(
            None if reduction_ci is None else reduction_ci[0],
            0.50,
            ">=",
            reduction_ci is not None and reduction_ci[0] >= 0.50,
        ),
        "guided_ca_lt2_total": _criterion(
            guided["pooled_nonlocal_ca_counts"]["nonlocal_ca_clashes_lt_2A"],
            0,
            "<=",
            guided["pooled_nonlocal_ca_counts"]["nonlocal_ca_clashes_lt_2A"] == 0,
        ),
        "penetration_rms_delta": _criterion(
            paired["deltas"]["nonlocal_ca_penetration_rms_A"]["guided_minus_control"],
            0.0,
            "<=",
            paired["deltas"]["nonlocal_ca_penetration_rms_A"]["guided_minus_control"] <= 0.0,
        ),
        "gdt_ts_mean_delta": _criterion(
            gdt["guided_minus_control"], -0.005, ">=", gdt["guided_minus_control"] >= -0.005
        ),
        "gdt_ts_ci_lower": _criterion(
            gdt["paired_target_bootstrap_95pct_ci"][0],
            -0.01,
            ">=",
            gdt["paired_target_bootstrap_95pct_ci"][0] >= -0.01,
        ),
        "lddt_mean_delta": _criterion(
            lddt["guided_minus_control"], -0.005, ">=", lddt["guided_minus_control"] >= -0.005
        ),
        "lddt_ci_lower": _criterion(
            lddt["paired_target_bootstrap_95pct_ci"][0],
            -0.01,
            ">=",
            lddt["paired_target_bootstrap_95pct_ci"][0] >= -0.01,
        ),
        "worst_target_gdt_ts_delta": _criterion(gdt_worst, -0.10, ">=", gdt_worst >= -0.10),
        "worst_target_lddt_delta": _criterion(lddt_worst, -0.10, ">=", lddt_worst >= -0.10),
        "bond_p95_mean_delta": _criterion(
            paired["deltas"]["bond_p95_A"]["guided_minus_control"],
            0.01,
            "<=",
            paired["deltas"]["bond_p95_A"]["guided_minus_control"] <= 0.01,
        ),
        "wrong_ca_chirality_mean_delta": _criterion(
            paired["deltas"]["ca_chirality_wrong_frac"]["guided_minus_control"],
            0.0,
            "<=",
            paired["deltas"]["ca_chirality_wrong_frac"]["guided_minus_control"] <= 0.0,
        ),
        "wrong_ca_chirality_pooled_count": _criterion(
            guided["pooled_ca_chirality_wrong_count"],
            control["pooled_ca_chirality_wrong_count"],
            "<=",
            guided["pooled_ca_chirality_wrong_count"] <= control["pooled_ca_chirality_wrong_count"],
        ),
    }

    for metric in CONTRACT["validity_guardrails"]["relative_rate_metrics"]:
        relative, passed = _rate_guardrail(
            float(control["means"][metric]),
            float(guided["means"][metric]),
            maximum_relative_increase=0.10,
        )
        criteria[f"{metric}_relative_increase"] = _criterion(
            relative if relative is not None else guided["means"][metric],
            "<=10% (or zero new events when control is zero)",
            "<=",
            passed,
        )

    return {
        "criteria": criteria,
        "passed": all(row["passed"] for row in criteria.values() if row["required"]),
    }


def summarize(
    root: Path,
    tuning_ids: list[str],
    *,
    bootstrap: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Load, validate, and aggregate the paired full-set experiment."""
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    normalized_tuning = [target.lower() for target in tuning_ids]
    if len(normalized_tuning) != EXPECTED_TUNING_COUNT or len(set(normalized_tuning)) != len(
        normalized_tuning
    ):
        raise ValueError(f"expected two unique tuning IDs, got {tuning_ids}")

    loaded = {condition: _load_condition(root, condition) for condition in CONDITIONS}
    rows = {condition: loaded[condition][0] for condition in CONDITIONS}
    manifests = {condition: loaded[condition][1] for condition in CONDITIONS}
    full_ids = _validate_pair(rows, manifests)
    missing_tuning = sorted(set(normalized_tuning) - set(full_ids))
    if missing_tuning:
        raise ValueError(f"tuning IDs absent from full target set: {missing_tuning}")
    tuning = sorted(normalized_tuning)
    heldout = sorted(set(full_ids) - set(tuning))
    if len(heldout) != EXPECTED_HELDOUT_COUNT:
        raise ValueError(f"expected {EXPECTED_HELDOUT_COUNT} held-out targets, got {len(heldout)}")

    subsets = {
        "heldout": _subset_summary(
            "heldout",
            "confirmatory scale-generalization set",
            rows,
            heldout,
            bootstrap=bootstrap,
            seed=seed,
        ),
        "tuning": _subset_summary(
            "tuning",
            "known-failure scale-selection set; exploratory only",
            rows,
            tuning,
            bootstrap=bootstrap,
            seed=seed,
        ),
        "full": _subset_summary(
            "full",
            "deployment summary containing confirmatory and tuning targets",
            rows,
            full_ids,
            bootstrap=bootstrap,
            seed=seed,
        ),
    }
    decision = _heldout_decision(subsets["heldout"])
    return {
        "schema_version": 1,
        "experiment": "self_overlap_guidance_full_casp14_v1",
        "conditions": {"control": CONTROL, "guided": GUIDED},
        "selection_disclosure": {
            "tuning_ids": tuning,
            "tested_steric_scales": [0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.2, 1.6, 2.4],
            "selected_scale": 1.0,
            "scale_selection_checkpoint": (
                "outputs/train/direct_puremamba_attn6_geo_adaln_sf360_esmc6b_ada_"
                "dstate64_gpu8_v5/ckpt_0132000.pt"
            ),
            "scale_selection_checkpoint_sha256": (
                "6fbd9be35bfcec3aef093ebbde5ba539c8523dc6dd9f323ab23ad6bdea7f96be"
            ),
            "confirmation_checkpoint": manifests[CONTROL].get("checkpoint"),
            "confirmation_checkpoint_sha256": manifests[CONTROL].get("checkpoint_sha256"),
            "selection_rule": "lowest scale passing the exploratory two-target gate",
            "scope_note": (
                "Scale selection used step 132000; the full paired confirmation uses the "
                "nearest retained snapshot, step 132500. Held-out is confirmatory only "
                "for steric-scale transfer, and CASP14 is not an untouched model-development "
                "benchmark."
            ),
        },
        "bootstrap": {
            "resamples": bootstrap,
            "seed": seed,
            "unit": "paired target",
            "interval": "percentile 95%",
        },
        "preregistered_contract": CONTRACT,
        "subsets": subsets,
        "heldout_decision": decision,
        "passed": decision["passed"],
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _fmt_ci(value: list[float] | None) -> str:
    if value is None:
        return "NA"
    return f"[{value[0]:+.4f}, {value[1]:+.4f}]"


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the confirmation, tuning, and full summaries without pooling roles."""
    lines = [
        "# Full CASP14 self-avoidance confirmation",
        "",
        "The steric scale was selected on two known-failure targets. The held-out 68-target "
        "subset is the primary decision set; tuning-2 and full-70 results are secondary.",
        "",
        f"Held-out decision: **{'PASS' if summary['passed'] else 'FAIL'}**.",
        "",
        "## Condition aggregates",
        "",
        "| Subset | n | Condition | GDT-TS | lDDT | Bond p95 Å | Hard clash/1k | "
        "OST clash/1k | OST bad bond/1k | OST bad angle/1k | Cα<2 | Cα<3 | Cα<3.6 | "
        "Penetration RMS Å | Time s | Peak GiB |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for subset_name in ("heldout", "tuning", "full"):
        subset = summary["subsets"][subset_name]
        for condition in CONDITIONS:
            row = subset["conditions"][condition]
            means = row["means"]
            pooled = row["pooled_nonlocal_ca_counts"]
            lines.append(
                f"| {subset_name} | {subset['n']} | {condition} | "
                f"{_fmt(means['gdt_ts'])} | {_fmt(means['lddt'])} | "
                f"{_fmt(means['bond_p95_A'])} | "
                f"{_fmt(means['hard_clashes_per_1k_atoms'])} | "
                f"{_fmt(means['ost_clashes_per_1k_atoms'])} | "
                f"{_fmt(means['ost_bad_bonds_per_1k_atoms'])} | "
                f"{_fmt(means['ost_bad_angles_per_1k_atoms'])} | "
                f"{pooled['nonlocal_ca_clashes_lt_2A']} | "
                f"{pooled['nonlocal_ca_clashes_lt_3A']} | "
                f"{pooled['nonlocal_ca_clashes_lt_3p6A']} | "
                f"{_fmt(means['nonlocal_ca_penetration_rms_A'])} | "
                f"{_fmt(row['runtime']['total_s'], 2)} | "
                f"{_fmt(row['vram']['max_peak_gib'], 3)} |"
            )

    lines.extend(
        [
            "",
            "## Paired effects",
            "",
            "| Subset | ΔGDT-TS [95% CI] | ΔlDDT [95% CI] | Cα<3.6 reduction "
            "[95% CI] | Δbond p95 Å | Δhard clash/1k | Runtime change |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for subset_name in ("heldout", "tuning", "full"):
        paired = summary["subsets"][subset_name]["paired"]
        gdt = paired["deltas"]["gdt_ts"]
        lddt = paired["deltas"]["lddt"]
        reduction = paired["nonlocal_ca_clashes_lt_3p6A_reduction"]
        reduction_ci = reduction["paired_target_bootstrap_95pct_ci"]
        if reduction["fraction"] is None or reduction_ci is None:
            reduction_text = "NA"
        else:
            reduction_text = (
                f"{100.0 * reduction['fraction']:.1f}% "
                f"[{100.0 * reduction_ci[0]:.1f}%, {100.0 * reduction_ci[1]:.1f}%]"
            )
        lines.append(
            f"| {subset_name} | {gdt['guided_minus_control']:+.4f} "
            f"{_fmt_ci(gdt['paired_target_bootstrap_95pct_ci'])} | "
            f"{lddt['guided_minus_control']:+.4f} "
            f"{_fmt_ci(lddt['paired_target_bootstrap_95pct_ci'])} | "
            f"{reduction_text} | "
            f"{paired['deltas']['bond_p95_A']['guided_minus_control']:+.4f} | "
            f"{paired['deltas']['hard_clashes_per_1k_atoms']['guided_minus_control']:+.4f} | "
            f"{100.0 * paired['runtime']['relative_total_change']:+.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Pre-registered held-out decision",
            "",
            "| Criterion | Observed | Rule | Threshold | Result |",
            "|---|---:|:---:|---:|:---:|",
        ]
    )
    for name, criterion in summary["heldout_decision"]["criteria"].items():
        lines.append(
            f"| {name} | {_fmt(criterion['observed'])} | {criterion['operator']} | "
            f"{_fmt(criterion['threshold'])} | "
            f"{'PASS' if criterion['passed'] else 'FAIL'} |"
        )

    lines.extend(
        [
            "",
            "## Worst held-out accuracy regressions",
            "",
            "| Metric | Target | Control | Guided | Δ |",
            "|---|---|---:|---:|---:|",
        ]
    )
    worst = summary["subsets"]["heldout"]["paired"]["worst_target_regressions"]
    for metric in ("gdt_ts", "lddt"):
        for row in worst[metric]:
            lines.append(
                f"| {metric} | {row['target']} | {_fmt(row['control'])} | "
                f"{_fmt(row['guided'])} | {row['delta']:+.4f} |"
            )

    lines.extend(
        [
            "",
            "## Selection and interpretation",
            "",
            "- Tuning targets are excluded from the held-out decision and shown separately.",
            "- Steric scale 1.0 was selected at step 132000; this full paired run uses the "
            "nearest retained checkpoint at step 132500.",
            "- Full-70 is a deployment summary, not an independent confirmatory estimate.",
            "- Runtime is descriptive unless condition order was counterbalanced.",
            "- OpenStructure rates use its issue lists divided by predicted atom count; the "
            "local scorer's atom count is used when OpenStructure does not expose one.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tuning-ids", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tuning_ids = read_tuning_ids(args.tuning_ids)
    result = summarize(
        args.root.resolve(),
        tuning_ids,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    (args.root / "full_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.root / "full_comparison.md").write_text(render_markdown(result) + "\n")
    print(json.dumps(result["heldout_decision"], indent=2))


if __name__ == "__main__":
    main()
