#!/usr/bin/env python3
"""Pair new predictions with the exact reference files used by a frozen run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage_casp_references import sha256, stage_pdb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--template-pairs", type=Path, required=True)
    parser.add_argument("--target-ids", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    target_ids = [
        line.strip().lower() for line in args.target_ids.read_text().splitlines() if line.strip()
    ]
    if not target_ids or len(target_ids) != len(set(target_ids)):
        raise ValueError("target IDs must be nonempty and unique")
    selected = set(target_ids)
    template_path = args.template_pairs / "manifest.json"
    template = json.loads(template_path.read_text())
    rows = [row for row in template["rows"] if row["target_id"].lower() in selected]
    if {row["target_id"].lower() for row in rows} != selected:
        raise ValueError("selected targets are absent from the frozen pair manifest")

    rollout_path = args.rollout_dir / "rollout.json"
    rollout = json.loads(rollout_path.read_text())
    if set(rollout["per_target"]) != selected:
        raise ValueError("rollout targets differ from selected target IDs")

    args.out_dir.mkdir(parents=True, exist_ok=False)
    staged = []
    for row in rows:
        target = row["target_id"].lower()
        pair_id = row.get("pair_id", target)
        if pair_id in {"", ".", ".."} or "/" in pair_id:
            raise ValueError(f"invalid pair ID: {pair_id}")
        reference = args.template_pairs / f"{pair_id}_gt.pdb"
        reference_hash = sha256(reference)
        if reference_hash != row["reference_sha256"]:
            raise ValueError(f"frozen reference changed: {pair_id}")
        prediction = args.rollout_dir / "structures" / f"{target}.pdb"
        if not prediction.is_file():
            raise FileNotFoundError(prediction)
        staged_pred = args.out_dir / f"{pair_id}_pred.pdb"
        stage_pdb(prediction, staged_pred)
        (args.out_dir / f"{pair_id}_gt.pdb").symlink_to(reference.resolve())
        staged.append(
            {
                "target_id": target,
                "pair_id": pair_id,
                "weight": row.get("weight", 1),
                "prediction_sha256": sha256(staged_pred),
                "reference_sha256": reference_hash,
            }
        )
    manifest = {
        "schema_version": 1,
        "dataset": template.get("dataset", "casp14"),
        "rollout_sha256": sha256(rollout_path),
        "template_manifest_sha256": sha256(template_path),
        "target_ids_sha256": sha256(args.target_ids),
        "target_count": len(selected),
        "reference_pair_count": len(staged),
        "rows": staged,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"staged {len(staged)} frozen-reference pairs for {len(selected)} targets")


if __name__ == "__main__":
    main()
