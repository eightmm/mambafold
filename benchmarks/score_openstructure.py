#!/usr/bin/env python3
"""Score staged ``*_pred.pdb``/``*_gt.pdb`` pairs with OpenStructure."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

METRICS = ("oligo_gdtts", "oligo_gdtha", "tm_score", "lddt", "bb_lddt", "rmsd")
COMPARE_ARGS = (
    "--fault-tolerant",
    "--min-pep-length",
    "4",
    "--lddt",
    "--bb-lddt",
    "--rigid-scores",
    "--tm-score",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_result(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if result.get("status") != "SUCCESS":
        return None
    if any(not isinstance(result.get(metric), (int, float)) for metric in METRICS):
        return None
    return result


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ost", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()

    predictions = sorted(args.in_dir.glob("*_pred.pdb"))
    if len(predictions) != args.expected:
        raise SystemExit(
            f"expected {args.expected} predictions in {args.in_dir}, found {len(predictions)}"
        )
    if not args.ost.is_file():
        raise SystemExit(f"OpenStructure executable not found: {args.ost}")
    version = subprocess.run(
        [str(args.ost), "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()

    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    rows = []
    for index, prediction in enumerate(predictions, start=1):
        target = prediction.name.removesuffix("_pred.pdb")
        reference = args.in_dir / f"{target}_gt.pdb"
        if not reference.is_file():
            raise SystemExit(f"missing reference for {target}: {reference}")
        output = raw_dir / f"{target}.json"
        identity_path = raw_dir / f"{target}.inputs.json"
        identity = {
            "schema_version": 1,
            "prediction_sha256": sha256(prediction),
            "reference_sha256": sha256(reference),
            "openstructure_version": version,
            "compare_args": list(COMPARE_ARGS),
        }
        result = valid_result(output)
        if result is not None and identity_path.is_file():
            try:
                cached_identity = json.loads(identity_path.read_text())
            except json.JSONDecodeError:
                cached_identity = None
            if cached_identity != identity:
                result = None
        else:
            result = None

        if result is None:
            command = [
                str(args.ost),
                "compare-structures",
                "-m",
                str(prediction),
                "-r",
                str(reference),
                "-o",
                str(output),
                *COMPARE_ARGS,
            ]
            print(f"[{index}/{len(predictions)}] SCORE {target}", flush=True)
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            result = valid_result(output)
            if completed.returncode != 0 or result is None:
                failures.append(
                    {
                        "target_id": target,
                        "returncode": completed.returncode,
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                    }
                )
                continue
            identity_path.write_text(json.dumps(identity, indent=2) + "\n")
        else:
            print(f"[{index}/{len(predictions)}] RESUME {target}", flush=True)
        rows.append(
            {"target_id": target, **{metric: float(result[metric]) for metric in METRICS}}
        )

    summary = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "input_dir": str(args.in_dir.resolve()),
        "input_manifest_sha256": sha256(args.in_dir / "manifest.json"),
        "expected_targets": args.expected,
        "successful_targets": len(rows),
        "evaluation_complete": not failures and len(rows) == args.expected,
        "openstructure": {
            "version": version,
            "executable": str(args.ost),
            "compare_args": list(COMPARE_ARGS),
        },
        "metrics": {
            metric: summarize([float(row[metric]) for row in rows]) for metric in METRICS
        },
        "rows": rows,
        "failures": failures,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out_dir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    print(json.dumps(summary["metrics"], indent=2), flush=True)
    raise SystemExit(0 if summary["evaluation_complete"] else 1)


if __name__ == "__main__":
    main()
