#!/usr/bin/env python3
"""Recompute a baseline mean on a frozen, model-independent target ID list."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

METRICS = ("oligo_gdtts", "oligo_gdtha", "tm_score", "lddt", "bb_lddt", "rmsd")


def summarize(summary: dict, admitted_ids: list[str]) -> dict:
    rows = summary.get("target_rows", summary.get("rows"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("source summary has no target rows")
    id_field = "target_id" if "target_id" in rows[0] else "target"
    indexed = {row[id_field].lower(): row for row in rows}
    selected = [identifier.lower() for identifier in admitted_ids]
    if len(indexed) != len(rows) or len(selected) != len(set(selected)):
        raise ValueError("duplicate source or admitted target ID")
    if set(selected) - indexed.keys():
        raise ValueError(f"missing baseline targets: {sorted(set(selected) - indexed.keys())}")
    if summary.get("success_count", summary.get("successful_target_count")) != len(rows):
        raise ValueError("source summary is incomplete")
    chosen = [indexed[identifier] for identifier in selected]
    return {
        "schema_version": 1,
        "source_target_count": len(rows),
        "admitted_target_count": len(chosen),
        "target_ids": selected,
        "metrics": {
            metric: {
                "n": len(chosen),
                "mean": statistics.fmean(float(row[metric]) for row in chosen),
                "median": statistics.median(float(row[metric]) for row in chosen),
            }
            for metric in METRICS
        },
        "rows": [{"target_id": identifier, **{metric: float(indexed[identifier][metric]) for metric in METRICS}} for identifier in selected],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--target-ids", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.summary.read_text())
    identifiers = [line.strip() for line in args.target_ids.read_text().splitlines() if line.strip()]
    result = summarize(source, identifiers)
    result["source_summary_sha256"] = hashlib.sha256(args.summary.read_bytes()).hexdigest()
    result["target_ids_sha256"] = hashlib.sha256(args.target_ids.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({metric: value["mean"] for metric, value in result["metrics"].items()}, indent=2))


if __name__ == "__main__":
    main()
