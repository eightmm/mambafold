#!/usr/bin/env python3
"""Aggregate CASP15 domain scores into residue-weighted target scores."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = ("oligo_gdtts", "oligo_gdtha", "tm_score", "lddt", "bb_lddt", "rmsd")


def aggregate(manifest: dict, pair_summary: dict) -> dict:
    if manifest["dataset"] != "casp15" or not pair_summary["evaluation_complete"]:
        raise ValueError("complete CASP15 domain scores required")
    pair_scores = {row["target_id"]: row for row in pair_summary["rows"]}
    if len(pair_scores) != len(manifest["rows"]) or set(pair_scores) != {row["pair_id"] for row in manifest["rows"]}:
        raise ValueError("scored pairs differ from staged references")
    grouped = defaultdict(list)
    for row in manifest["rows"]:
        grouped[row["target_id"]].append(row)
    if len(grouped) != manifest["target_count"]:
        raise ValueError("target count differs from manifest")
    target_rows = []
    for target, refs in sorted(grouped.items()):
        weight = sum(row["weight"] for row in refs)
        if weight <= 0:
            raise ValueError(f"invalid weight for {target}")
        target_rows.append({
            "target_id": target,
            "reference_count": len(refs),
            "total_weight": weight,
            **{
                metric: sum(pair_scores[row["pair_id"]][metric] * row["weight"] for row in refs) / weight
                for metric in METRICS
            },
        })
    return {
        "schema_version": 1,
        "dataset": "casp15",
        "aggregation": "mapped-residue-weighted domain mean within target, then unweighted target mean",
        "target_count": len(target_rows),
        "reference_pair_count": len(manifest["rows"]),
        "target_rows": target_rows,
        "metrics": {metric: {"n": len(target_rows), "mean": statistics.fmean(row[metric] for row in target_rows)} for metric in METRICS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(json.loads(args.manifest.read_text()), json.loads(args.pair_summary.read_text()))
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
