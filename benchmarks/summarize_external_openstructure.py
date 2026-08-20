#!/usr/bin/env python3
"""Build full-five and common-six accuracy tables from external scores."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

METRICS = ("oligo_gdtts", "oligo_gdtha", "tm_score", "lddt", "bb_lddt", "rmsd")
MODELS = (
    ("mambafold_esm3_step120000", "MambaFold-ESM3, step 120,000"),
    ("mambafold_esmc6b_step119500", "MambaFold-ESMC-6B, step 119,500"),
    ("simplefold_360m", "SimpleFold-360M"),
    ("esmfold_v1_6000ada", "ESMFold v1"),
    ("dplm2_bit_650m_6000ada", "DPLM-2 Bit 650M"),
    ("omegafold_model2_cycle1", "OmegaFold model 2"),
)
DATASETS = (
    ("casp15", "CASP15 strict single-chain"),
    ("casp16", "CASP16 strict single-chain"),
    ("cameo22", "CAMEO22"),
)


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
        lines.append(
            "| {label} | {n} | {oligo_gdtts:.3f} | {oligo_gdtha:.3f} | "
            "{tm_score:.3f} | {lddt:.3f} | {bb_lddt:.3f} | {rmsd:.3f} |".format(**row)
        )
    return lines


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-root",
        type=Path,
        default=repo_root / "outputs/eval/external_compare_v1_20260812/scores/external_accuracy_v2",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    output: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "score_root": str(args.score_root.resolve()),
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
            summary_path = args.score_root / dataset / model / "summary.json"
            if not summary_path.is_file():
                raise SystemExit(f"Missing score summary: {summary_path}")
            summary = json.loads(summary_path.read_text())
            if not summary["evaluation_complete_for_available_predictions"]:
                raise SystemExit(f"Incomplete score summary: {summary_path}")
            summaries[model] = summary
            rows_by_model[model] = {row["target_id"]: row for row in summary["target_rows"]}

        full_models = [model for model, _ in MODELS[:-1]]
        full_targets = set.intersection(*(set(rows_by_model[model]) for model in full_models))
        expected_count = summaries[full_models[0]]["expected_target_count"]
        if len(full_targets) != expected_count:
            raise SystemExit(
                f"{dataset}: full-five intersection is {len(full_targets)}, "
                f"expected {expected_count}"
            )
        common_targets = set.intersection(*(set(rows_by_model[model]) for model, _ in MODELS))

        full_rows = []
        common_rows = []
        for model, label in MODELS:
            if model in full_models:
                full_rows.append(
                    {
                        "model": model,
                        "label": label,
                        **aggregate(rows_by_model[model], full_targets),
                    }
                )
            common_rows.append(
                {
                    "model": model,
                    "label": label,
                    **aggregate(rows_by_model[model], common_targets),
                }
            )

        output["datasets"][dataset] = {
            "label": dataset_label,
            "aggregation": summaries[full_models[0]]["aggregation"],
            "full_five": {"target_ids": sorted(full_targets), "rows": full_rows},
            "common_six": {"target_ids": sorted(common_targets), "rows": common_rows},
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
        md.extend((f"### Full five-model set (N={len(full_targets)})", ""))
        md.extend(render_table(full_rows))
        md.extend(("", f"### All-six common set (N={len(common_targets)})", ""))
        md.extend(render_table(common_rows))
        md.append("")

    md.extend(
        (
            "DPLM-2 writes only N, CA, C, O, and CB atoms; its all-atom lDDT is "
            "not directly comparable with full-side-chain outputs. Backbone lDDT "
            "is the appropriate local metric.",
            "",
        )
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    args.output_md.write_text("\n".join(md))
    print(args.output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()
