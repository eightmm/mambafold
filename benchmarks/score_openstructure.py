#!/usr/bin/env python3
"""Score paired ``*_pred.pdb``/``*_gt.pdb`` files with OpenStructure."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

METRICS = ("oligo_gdtts", "oligo_gdtha", "tm_score", "lddt", "bb_lddt", "rmsd")


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--ost", required=True, type=Path)
    parser.add_argument("--expected", type=int, default=None)
    args = parser.parse_args()

    prediction_paths = sorted(args.in_dir.glob("*_pred.pdb"))
    if args.expected is not None and len(prediction_paths) != args.expected:
        raise SystemExit(
            f"Expected {args.expected} predictions in {args.in_dir}, found {len(prediction_paths)}"
        )
    if not prediction_paths:
        raise SystemExit(f"No canonical *_pred.pdb files found in {args.in_dir}")
    if not args.ost.is_file():
        raise SystemExit(f"OpenStructure executable not found: {args.ost}")

    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, float | str]] = []
    for prediction in prediction_paths:
        target = prediction.name.removesuffix("_pred.pdb")
        reference = args.in_dir / f"{target}_gt.pdb"
        if not reference.is_file():
            raise SystemExit(f"Missing reference for {target}: {reference}")
        output = args.out_dir / f"{target}.json"
        command = [
            str(args.ost),
            "compare-structures",
            "-m",
            str(prediction),
            "-r",
            str(reference),
            "-o",
            str(output),
            "--fault-tolerant",
            "--min-pep-length",
            "4",
            "--lddt",
            "--bb-lddt",
            "--rigid-scores",
            "--tm-score",
        ]
        print(f"SCORE {target}", flush=True)
        subprocess.run(command, check=True)
        result = json.loads(output.read_text())
        row: dict[str, float | str] = {"target": target}
        for metric in METRICS:
            value = result.get(metric)
            if not isinstance(value, (int, float)):
                raise SystemExit(f"{target}: missing numeric metric {metric}")
            row[metric] = float(value)
        rows.append(row)

    version = subprocess.run(
        [str(args.ost), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "evaluation_dir": str(args.in_dir.resolve()),
        "target_count": len(rows),
        "success_count": len(rows),
        "openstructure": {
            "version": version,
            "executable": str(args.ost),
            "command": (
                "ost compare-structures -m MODEL_FILE -r REFERENCE_FILE "
                "-o OUTPUT_FILE --fault-tolerant --min-pep-length 4 "
                "--lddt --bb-lddt --rigid-scores --tm-score"
            ),
        },
        "metrics": {metric: summarize([float(row[metric]) for row in rows]) for metric in METRICS},
        "rows": rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
