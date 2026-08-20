#!/usr/bin/env python3
"""Build the active four-model accuracy tables from external scores."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

METRICS = ("oligo_gdtts", "oligo_gdtha", "tm_score", "lddt", "bb_lddt", "rmsd")
MODELS = (
    ("mambafold_esmc6b_step170000", "MambaFold-ESMC-6B, step 170,000 preview"),
    ("simplefold_360m", "SimpleFold-360M"),
    ("esmfold_v1", "ESMFold v1"),
    ("dplm2_bit_650m", "DPLM-2 Bit 650M"),
)
DATASETS = (
    ("casp16", "CASP16 strict single-chain"),
    ("casp15", "CASP15 strict single-chain"),
)
INCOMPARABLE_METRICS = {"dplm2_bit_650m": {"lddt"}}


def aggregate(rows: dict[str, dict[str, Any]], target_ids: set[str]) -> dict[str, Any]:
    selected = [rows[target_id] for target_id in sorted(target_ids)]
    return {
        "n": len(selected),
        **{metric: statistics.fmean(float(row[metric]) for row in selected) for metric in METRICS},
    }


def render_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        (
            "| Model | N | GDT-TS ↑ | GDT-HA ↑ | TM-score ↑ | "
            "all-atom lDDT ↑ | backbone lDDT ↑ | RMSD (Å) ↓ |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        values = {
            metric: "NA" if row[metric] is None else f"{float(row[metric]):.3f}"
            for metric in METRICS
        }
        lines.append(
            f"| {row['label']} | {row['n']} | {values['oligo_gdtts']} | "
            f"{values['oligo_gdtha']} | {values['tm_score']} | {values['lddt']} | "
            f"{values['bb_lddt']} | {values['rmsd']} |"
        )
    return lines


def build_report(score_root: Path) -> tuple[dict[str, Any], str]:
    output: dict[str, Any] = {
        "schema_version": 2,
        "generated_at": datetime.now().astimezone().isoformat(),
        "score_root": str(score_root.resolve()),
        "datasets": {},
    }
    md = [
        "# External structure-accuracy tables",
        "",
        "All values are seed-0 OpenStructure 2.9.1 target means.",
        "",
    ]

    for dataset, dataset_label in DATASETS:
        summaries: dict[str, dict[str, Any]] = {}
        rows_by_model: dict[str, dict[str, dict[str, Any]]] = {}
        for model, _ in MODELS:
            summary_path = score_root / dataset / model / "summary.json"
            if not summary_path.is_file():
                raise SystemExit(f"Missing score summary: {summary_path}")
            summary = json.loads(summary_path.read_text())
            complete = summary.get(
                "evaluation_complete",
                summary.get("evaluation_complete_for_available_predictions", False),
            )
            if not complete:
                raise SystemExit(f"Incomplete score summary: {summary_path}")
            summaries[model] = summary
            rows_by_model[model] = {row["target_id"]: row for row in summary["target_rows"]}

        active_models = [model for model, _ in MODELS]
        expected_counts = {
            int(summaries[model]["expected_target_count"]) for model in active_models
        }
        if len(expected_counts) != 1:
            raise SystemExit(
                f"{dataset}: inconsistent expected target counts: {sorted(expected_counts)}"
            )
        expected_count = expected_counts.pop()
        full_targets = set(rows_by_model[active_models[0]])
        for model in active_models:
            model_targets = set(rows_by_model[model])
            if len(model_targets) != expected_count:
                raise SystemExit(
                    f"{dataset}/{model}: scored {len(model_targets)} targets, "
                    f"expected {expected_count}"
                )
            if model_targets != full_targets:
                missing = sorted(full_targets - model_targets)
                extra = sorted(model_targets - full_targets)
                raise SystemExit(
                    f"{dataset}/{model}: target set mismatch: missing={missing}, extra={extra}"
                )

        full_rows = []
        for model, label in MODELS:
            aggregates = aggregate(rows_by_model[model], full_targets)
            for metric in INCOMPARABLE_METRICS.get(model, set()):
                aggregates[metric] = None
            full_rows.append(
                {
                    "model": model,
                    "label": label,
                    **aggregates,
                }
            )

        output["datasets"][dataset] = {
            "label": dataset_label,
            "aggregation": summaries[active_models[0]]["aggregation"],
            "full_comparison": {"target_ids": sorted(full_targets), "rows": full_rows},
        }

        md.extend((f"## {dataset_label}", ""))
        if dataset == "casp15":
            md.extend(
                (
                    "Official domain/EU scores are mapped-residue-weighted within each target; "
                    "the table is then an unweighted target mean.",
                    "",
                )
            )
        elif dataset == "casp16":
            md.extend(("Official whole-chain references are used.", ""))
        else:
            md.extend(("Frozen state-1 references are used.", ""))
        md.extend((f"### Full four-model comparison (N={len(full_targets)})", ""))
        md.extend(render_table(full_rows))
        md.append("")

    md.extend(
        (
            "DPLM-2 writes only N, CA, C, O, and CB atoms; its all-atom lDDT is "
            "not directly comparable with full-side-chain outputs. Backbone lDDT "
            "is the appropriate local metric.",
            "",
        )
    )
    return output, "\n".join(md)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-root",
        type=Path,
        default=repo_root / "outputs/eval/external_compare_esmc6b/scores/external_accuracy_v2",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    output, markdown = build_report(args.score_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    args.output_md.write_text(markdown)
    print(args.output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()
