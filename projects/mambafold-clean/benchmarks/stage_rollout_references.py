#!/usr/bin/env python3
"""Stage rollout predictions and frozen references as ``*_pred/gt.pdb`` pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ids(path: Path) -> list[str]:
    identifiers = [
        line.split("#", 1)[0].strip()
        for line in path.read_text().splitlines()
        if line.split("#", 1)[0].strip()
    ]
    if not identifiers:
        raise ValueError(f"target id file is empty: {path}")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"target id file contains duplicates: {path}")
    return identifiers


def manifest_references(path: Path, state: str | None) -> dict[str, Path]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if state is not None:
        rows = [row for row in rows if row.get("state") == state]
    references: dict[str, Path] = {}
    for row in rows:
        target = row["target_id"].lower()
        if target in references:
            raise ValueError(f"duplicate reference target in {path}: {target}")
        references[target] = path.parent / row["reference_path"]
    return references


def directory_references(path: Path, target_ids: list[str]) -> dict[str, Path]:
    references = {}
    for target in target_ids:
        candidates = (path / f"{target}_gt.pdb", path / f"{target}.pdb")
        matches = [candidate for candidate in candidates if candidate.is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one reference for {target} under {path}, found {matches}"
            )
        references[target.lower()] = matches[0]
    return references


def link(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite staged input: {destination}")
    destination.symlink_to(source.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--reference-dir", type=Path)
    source.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--reference-state", help="optional manifest state filter")
    parser.add_argument("--target-ids", type=Path)
    args = parser.parse_args()

    rollout_path = args.rollout_dir / "rollout.json"
    rollout = json.loads(rollout_path.read_text())
    available = {str(target).lower(): str(target) for target in rollout["per_target"]}
    target_ids = read_ids(args.target_ids) if args.target_ids else sorted(available.values())
    normalized_ids = [target.lower() for target in target_ids]
    unknown = sorted(set(normalized_ids) - set(available))
    if unknown:
        raise ValueError(f"targets absent from rollout: {unknown}")

    if args.reference_manifest:
        references = manifest_references(args.reference_manifest, args.reference_state)
    else:
        references = directory_references(args.reference_dir, target_ids)
    missing_references = sorted(set(normalized_ids) - set(references))
    if missing_references:
        raise FileNotFoundError(f"targets without references: {missing_references}")

    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for requested, target in zip(target_ids, normalized_ids):
        rollout_target = available[target]
        prediction = args.rollout_dir / "structures" / f"{rollout_target}.pdb"
        reference = references[target]
        if not prediction.is_file():
            raise FileNotFoundError(f"missing rollout prediction: {prediction}")
        if not reference.is_file():
            raise FileNotFoundError(f"missing reference: {reference}")
        prediction_out = args.out_dir / f"{requested}_pred.pdb"
        reference_out = args.out_dir / f"{requested}_gt.pdb"
        link(prediction, prediction_out)
        link(reference, reference_out)
        rows.append(
            {
                "target_id": requested,
                "prediction": str(prediction.resolve()),
                "prediction_sha256": sha256(prediction),
                "reference": str(reference.resolve()),
                "reference_sha256": sha256(reference),
            }
        )

    manifest = {
        "schema_version": 1,
        "rollout": str(rollout_path.resolve()),
        "rollout_sha256": sha256(rollout_path),
        "target_ids_file": str(args.target_ids.resolve()) if args.target_ids else None,
        "target_ids_sha256": sha256(args.target_ids) if args.target_ids else None,
        "reference_manifest": (
            str(args.reference_manifest.resolve()) if args.reference_manifest else None
        ),
        "reference_manifest_sha256": (
            sha256(args.reference_manifest) if args.reference_manifest else None
        ),
        "reference_state": args.reference_state,
        "targets": len(rows),
        "rows": rows,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"staged {len(rows)} prediction/reference pairs -> {args.out_dir}")


if __name__ == "__main__":
    main()
