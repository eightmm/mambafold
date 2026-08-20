#!/usr/bin/env python3
"""Stage paired baseline/guided CASP14 inputs for validity scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASELINE_DIRS = {
    "esm3": Path(
        "outputs/eval/external_compare_v1_20260812/scores/casp14_full70/"
        "pairs/mambafold_esm3_step120000"
    ),
    "esmc6b": Path(
        "outputs/eval/external_compare_v1_20260812/scores/external_accuracy_v2/"
        "casp14/mambafold_esmc6b_step119500/inputs"
    ),
}


def link(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise SystemExit(f"Missing source PDB: {source}")
    destination.symlink_to(source.resolve())


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(BASELINE_DIRS), required=True)
    parser.add_argument("--condition", choices=("baseline", "guided"), required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=None,
        help="Explicit prediction directory; overrides the legacy condition lookup.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="Explicit GT directory; defaults to the legacy baseline directory.",
    )
    args = parser.parse_args()

    ids = [line.strip() for line in args.ids.read_text().splitlines() if line.strip()]
    if len(ids) != 69 or len(set(ids)) != len(ids) or "t1061" in ids:
        raise SystemExit("Expected 69 unique CASP14 IDs excluding guidance-selection target t1061")

    baseline_dir = repo_root / BASELINE_DIRS[args.model]
    guided_dir = args.experiment_root / "inference" / args.model / "guided"
    prediction_dir = args.prediction_dir or (
        baseline_dir if args.condition == "baseline" else guided_dir
    )
    reference_dir = args.reference_dir or baseline_dir
    output_dir = args.experiment_root / "scoring_inputs" / args.model / args.condition
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite scoring input directory: {output_dir}")
    output_dir.mkdir(parents=True)

    rows = []
    for target_id in ids:
        prediction = prediction_dir / f"{target_id}_pred.pdb"
        reference = reference_dir / f"{target_id}_gt.pdb"
        link(prediction, output_dir / f"{target_id}_pred.pdb")
        link(reference, output_dir / f"{target_id}_gt.pdb")
        rows.append(
            {
                "target_id": target_id,
                "prediction": str(prediction.resolve()),
                "reference": str(reference.resolve()),
            }
        )

    manifest = {
        "schema_version": 1,
        "model": args.model,
        "condition": args.condition,
        "selection": "CASP14 full70 minus prior guidance-selection target T1061",
        "prediction_dir": str(prediction_dir.resolve()),
        "reference_dir": str(reference_dir.resolve()),
        "target_count": len(rows),
        "rows": rows,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"staged {len(rows)} pairs -> {output_dir}")


if __name__ == "__main__":
    main()
