#!/usr/bin/env python3
"""Stage CASP15 domain or CASP16 whole-chain references for a rollout."""

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


def stage_pdb(source: Path, destination: Path) -> None:
    lines = source.read_text().splitlines(keepends=True)
    atoms = [line for line in lines if line.startswith("ATOM  ")]
    if not atoms:
        raise ValueError(f"PDB has no ATOM records: {source}")
    named = {line[21:22] for line in atoms if len(line) > 21 and line[21:22].strip()}
    if len(named) > 1:
        raise ValueError(f"multiple chain IDs in {source}: {named}")
    assigned = next(iter(named), "A")
    normalized = []
    changed = False
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM", "TER   ")) and (len(line) <= 21 or not line[21:22].strip()):
            newline = "\n" if line.endswith("\n") else ""
            body = (line[:-1] if newline else line).ljust(22)
            line = body[:21] + assigned + body[22:] + newline
            changed = True
        normalized.append(line)
    if changed:
        destination.write_text("".join(normalized))
    else:
        destination.symlink_to(source.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("casp15", "casp16"), required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--target-ids", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    target_ids = [line.strip().lower() for line in args.target_ids.read_text().splitlines() if line.strip()]
    if not target_ids or len(target_ids) != len(set(target_ids)):
        raise ValueError("target IDs must be nonempty and unique")
    selected = set(target_ids)
    rollout_path = args.rollout_dir / "rollout.json"
    rollout = json.loads(rollout_path.read_text())
    available = {key.lower(): key for key in rollout["per_target"]}
    if selected - available.keys():
        raise ValueError(f"missing rollout targets: {sorted(selected - available.keys())}")

    manifest_path = args.reference_root / "primary_reference_manifest.tsv"
    with manifest_path.open(newline="") as handle:
        references = list(csv.DictReader(handle, delimiter="\t"))
    kind = "domain_EU" if args.dataset == "casp15" else "whole"
    references = [row for row in references if row["reference_kind"] == kind and row["prediction_id"].lower() in selected]
    if {row["prediction_id"].lower() for row in references} != selected:
        raise ValueError("selected targets do not match reference manifest")

    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    pair_ids = set()
    for row in references:
        target = row["prediction_id"].lower()
        pair_id = row["reference_id"]
        if pair_id in pair_ids or "/" in pair_id or pair_id in {".", ".."}:
            raise ValueError(f"invalid or duplicate reference ID: {pair_id}")
        pair_ids.add(pair_id)
        prediction = args.rollout_dir / "structures" / f"{available[target]}.pdb"
        reference = args.reference_root / row["reference_path"]
        if not prediction.is_file() or not reference.is_file():
            raise FileNotFoundError(f"missing pair for {pair_id}: {prediction}, {reference}")
        prediction_out = args.out_dir / f"{pair_id}_pred.pdb"
        reference_out = args.out_dir / f"{pair_id}_gt.pdb"
        stage_pdb(prediction, prediction_out)
        stage_pdb(reference, reference_out)
        rows.append({
            "target_id": target,
            "pair_id": pair_id,
            "weight": int(row["mapped_residues"]),
            "prediction_sha256": sha256(prediction_out),
            "reference_sha256": sha256(reference_out),
        })
    manifest = {
        "schema_version": 1,
        "dataset": args.dataset,
        "rollout_sha256": sha256(rollout_path),
        "reference_manifest_sha256": sha256(manifest_path),
        "target_ids_sha256": sha256(args.target_ids),
        "target_count": len(selected),
        "reference_pair_count": len(rows),
        "rows": rows,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"staged {len(rows)} references for {len(selected)} targets")


if __name__ == "__main__":
    main()
