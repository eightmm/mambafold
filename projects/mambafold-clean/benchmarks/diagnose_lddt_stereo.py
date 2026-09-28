#!/usr/bin/env python3
"""Separate raw local-distance accuracy from OpenStructure stereochemistry penalties."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score(ost: Path, pred: Path, ref: Path, out: Path) -> dict:
    command = [
        str(ost), "compare-structures", "-m", str(pred), "-r", str(ref), "-o", str(out),
        "--fault-tolerant", "--min-pep-length", "4", "--lddt", "--bb-lddt",
        "--rigid-scores", "--tm-score", "--lddt-no-stereochecks",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = json.loads(out.read_text())
    if result.get("status") != "SUCCESS" or result.get("lddt_no_stereochecks") is not True:
        raise ValueError(f"invalid no-stereocheck score: {out}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("casp15", "casp16"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mamba-pairs", type=Path, required=True)
    parser.add_argument("--simplefold-pairs", type=Path, required=True)
    parser.add_argument("--mamba-raw", type=Path, required=True)
    parser.add_argument("--simplefold-raw", type=Path, required=True)
    parser.add_argument("--ost", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    if manifest["dataset"] != args.dataset:
        raise ValueError("manifest dataset mismatch")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "raw").mkdir()
    paired = defaultdict(list)
    for index, row in enumerate(manifest["rows"], start=1):
        pair_id = row["pair_id"]
        target_id = row["target_id"].lower()
        weight = row["weight"]
        scored = {}
        ref_hashes = set()
        for name, pairs, raw in (
            ("mambafold", args.mamba_pairs, args.mamba_raw),
            ("simplefold_360m", args.simplefold_pairs, args.simplefold_raw),
        ):
            pred = pairs / f"{pair_id}_pred.pdb"
            ref = pairs / f"{pair_id}_gt.pdb"
            ref_hashes.add(sha256(ref))
            default = json.loads((raw / f"{pair_id}.json").read_text())
            if default.get("status") != "SUCCESS" or default.get("lddt_no_stereochecks") is not False:
                raise ValueError(f"invalid default score: {raw / f'{pair_id}.json'}")
            unrestricted = score(args.ost, pred, ref, args.out_dir / "raw" / f"{pair_id}_{name}.json")
            scored[name] = {
                "default_lddt": float(default["lddt"]),
                "no_stereocheck_lddt": float(unrestricted["lddt"]),
                "penalty": float(unrestricted["lddt"] - default["lddt"]),
                "bad_bonds": len(default["model_bad_bonds"]),
                "bad_angles": len(default["model_bad_angles"]),
                "clashes": len(default["model_clashes"]),
            }
        if len(ref_hashes) != 1:
            raise ValueError(f"reference differs between models: {pair_id}")
        paired[target_id].append({"pair_id": pair_id, "weight": weight, **scored})
        print(f"[{index}/{len(manifest['rows'])}] {pair_id}", flush=True)

    if len(paired) != manifest["target_count"]:
        raise ValueError("target count mismatch")
    targets = []
    for target_id, pairs in sorted(paired.items()):
        weight_sum = sum(pair["weight"] for pair in pairs)
        target = {"target_id": target_id, "reference_pairs": len(pairs)}
        for name in ("mambafold", "simplefold_360m"):
            target[name] = {
                metric: sum(pair[name][metric] * pair["weight"] for pair in pairs) / weight_sum
                for metric in ("default_lddt", "no_stereocheck_lddt", "penalty", "bad_bonds", "bad_angles", "clashes")
            }
        targets.append(target)
    summary = {
        "schema_version": 1,
        "dataset": args.dataset,
        "manifest_sha256": sha256(args.manifest),
        "target_count": len(targets),
        "reference_pair_count": len(manifest["rows"]),
        "aggregation": "mapped-residue-weighted within target, unweighted across targets",
        "means": {
            name: {
                metric: statistics.fmean(target[name][metric] for target in targets)
                for metric in ("default_lddt", "no_stereocheck_lddt", "penalty", "bad_bonds", "bad_angles", "clashes")
            }
            for name in ("mambafold", "simplefold_360m")
        },
        "targets": targets,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["means"], indent=2))


if __name__ == "__main__":
    main()
