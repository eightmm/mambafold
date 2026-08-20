#!/usr/bin/env python3
"""Summarize paired stereochemical-guidance examples as JSON and Markdown."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

METRICS = {
    "gdt_ts": ("GDT-TS", "higher"),
    "lddt": ("lDDT", "higher"),
    "bond_mae_A": ("Bond MAE (Å)", "lower"),
    "bond_p95_A": ("Bond p95 (Å)", "lower"),
    "hard_clashes_per_1k_atoms": ("Hard clashes / 1k atoms", "lower"),
    "ca_chirality_wrong_frac": ("Wrong Cα chirality fraction", "lower"),
    "nonlocal_ca_min_A": ("Minimum nonlocal Cα distance (Å)", "higher"),
    "nonlocal_ca_clashes_lt_2A": ("Nonlocal Cα pairs <2 Å", "lower"),
    "nonlocal_ca_clashes_lt_3A": ("Nonlocal Cα pairs <3 Å", "lower"),
    "nonlocal_ca_clashes_lt_3p6A": ("Nonlocal Cα pairs <3.6 Å", "lower"),
    "nonlocal_ca_penetration_rms_A": ("Nonlocal Cα penetration RMS (Å)", "lower"),
    "ost_bad_bonds": ("OpenStructure bad bonds", "lower"),
    "ost_bad_angles": ("OpenStructure bad angles", "lower"),
    "ost_clashes": ("OpenStructure clashes", "lower"),
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _condition_rows(root: Path, condition: str) -> dict[str, dict[str, float | int]]:
    inference = root / "inference" / condition
    score = root / "scores" / condition
    manifest = _read(inference / "manifest.json")
    lengths = {row["pdb_id"]: int(row["L"]) for row in manifest["rows"]}
    local = _read(score / "local_geometry.json")
    local_rows = {row["pdb_id"]: row["pred"] for row in local["rows"]}
    ost = _read(score / "openstructure" / "summary.json")
    ost_rows = {row["target"]: row for row in ost["rows"]}

    targets = set(lengths)
    if targets != set(local_rows) or targets != set(ost_rows):
        raise ValueError(
            f"{condition}: target mismatch: manifest={sorted(targets)}, "
            f"local={sorted(local_rows)}, ost={sorted(ost_rows)}"
        )

    rows: dict[str, dict[str, float | int]] = {}
    for target in sorted(targets):
        local_row = local_rows[target]
        ost_row = ost_rows[target]
        ost_raw = _read(score / "openstructure" / f"{target}.json")
        rows[target] = {
            "length": lengths[target],
            "gdt_ts": float(ost_row["oligo_gdtts"]),
            "lddt": float(ost_row["lddt"]),
            "bond_mae_A": float(local_row["bond_mae_A"]),
            "bond_p95_A": float(local_row["bond_p95_A"]),
            "hard_clashes_per_1k_atoms": float(local_row["clashes_per_1k_atoms"]),
            "ca_chirality_wrong_frac": float(local_row["ca_chirality_wrong_frac"]),
            "nonlocal_ca_min_A": float(local_row["nonlocal_ca_min_A"]),
            "nonlocal_ca_clashes_lt_2A": int(local_row["nonlocal_ca_clashes_lt_2A"]),
            "nonlocal_ca_clashes_lt_3A": int(local_row["nonlocal_ca_clashes_lt_3A"]),
            "nonlocal_ca_clashes_lt_3p6A": int(local_row["nonlocal_ca_clashes_lt_3p6A"]),
            "nonlocal_ca_penetration_rms_A": float(local_row["nonlocal_ca_penetration_rms_A"]),
            "ost_bad_bonds": len(ost_raw.get("model_bad_bonds") or []),
            "ost_bad_angles": len(ost_raw.get("model_bad_angles") or []),
            "ost_clashes": len(ost_raw.get("model_clashes") or []),
        }
    return rows


def _mean(rows: dict[str, dict[str, float | int]], metric: str) -> float:
    return statistics.fmean(float(row[metric]) for row in rows.values())


def summarize(root: Path) -> dict[str, Any]:
    baseline = _condition_rows(root, "baseline")
    guided = _condition_rows(root, "guided")
    if set(baseline) != set(guided):
        raise ValueError("Baseline and guided target sets differ")

    rows = []
    for target in sorted(baseline):
        base = baseline[target]
        guide = guided[target]
        rows.append(
            {
                "target": target,
                "length": int(base["length"]),
                "baseline": base,
                "guided": guide,
                "delta_guided_minus_baseline": {
                    metric: float(guide[metric]) - float(base[metric]) for metric in METRICS
                },
            }
        )
    aggregate = {}
    for metric, (label, direction) in METRICS.items():
        base_mean = _mean(baseline, metric)
        guided_mean = _mean(guided, metric)
        aggregate[metric] = {
            "label": label,
            "better": direction,
            "baseline_mean": base_mean,
            "guided_mean": guided_mean,
            "delta_guided_minus_baseline": guided_mean - base_mean,
        }
    return {
        "schema_version": 1,
        "experiment": "stereochemical_guidance_examples_v1",
        "checkpoint_step": 132000,
        "selection": (
            "Exploratory examples selected for poor local geometry in the prior "
            "ESMC-6B step-119500 CASP14 baseline; not an unbiased benchmark."
        ),
        "target_count": len(rows),
        "aggregate": aggregate,
        "rows": rows,
    }


def _fmt(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Stereochemical guidance examples",
        "",
        "These are exploratory CASP14 examples selected because the earlier ESMC-6B "
        "step-119500 baseline had poor local geometry. The paired run below uses the "
        "same step-132000 EMA checkpoint, SDE-500 sampler, seed 0, and differs only "
        "by inference-time stereochemical guidance (scale 0 versus 0.03).",
        "",
        "## Aggregate",
        "",
        "| Metric | Better | Baseline | Guided | Δ (guided − baseline) |",
        "|---|:---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        row = summary["aggregate"][metric]
        lines.append(
            f"| {row['label']} | {row['better']} | "
            f"{_fmt(row['baseline_mean'])} | {_fmt(row['guided_mean'])} | "
            f"{_fmt(row['delta_guided_minus_baseline'])} |"
        )

    lines.extend(
        [
            "",
            "## Per-target local validity",
            "",
            "| Target | L | Condition | GDT-TS | lDDT | Bond p95 Å | "
            "Hard clashes/1k | OST bad bonds | OST bad angles | OST clashes |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["rows"]:
        for condition in ("baseline", "guided"):
            values = row[condition]
            lines.append(
                f"| {row['target']} | {row['length']} | {condition} | "
                f"{_fmt(values['gdt_ts'])} | {_fmt(values['lddt'])} | "
                f"{_fmt(values['bond_p95_A'])} | "
                f"{_fmt(values['hard_clashes_per_1k_atoms'])} | "
                f"{values['ost_bad_bonds']} | {values['ost_bad_angles']} | "
                f"{values['ost_clashes']} |"
            )

    lines.extend(
        [
            "",
            "## Structures",
            "",
            "| Target | Baseline PDB | Baseline CIF | Guided PDB | Guided CIF |",
            "|---|---|---|---|---|",
        ]
    )
    for row in summary["rows"]:
        target = row["target"]
        lines.append(
            f"| {target} | [PDB](inference/baseline/{target}_pred.pdb) | "
            f"[CIF](inference/baseline/{target}_pred.cif) | "
            f"[PDB](inference/guided/{target}_pred.pdb) | "
            f"[CIF](inference/guided/{target}_pred.cif) |"
        )
    lines.extend(
        [
            "",
            "OpenStructure counts use its topology-aware stereochemical checks. "
            "The simpler hard-clash metric excludes same and adjacent residues and "
            "counts atom pairs below 1.5 Å. Lower local-validity counts are better; "
            "higher GDT-TS/lDDT are better.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.root)
    (args.root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.root / "comparison.md").write_text(render_markdown(summary))
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
