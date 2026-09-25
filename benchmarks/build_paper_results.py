#!/usr/bin/env python3
"""Build a compact, checked manuscript table from frozen benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCORE_ROOT = ROOT / "outputs/benchmarks/run-a-final"
DATA_ROOT = ROOT / "docs/paper/data"
AUDIT = ROOT / "data/audit"
METRICS = ("oligo_gdtts", "oligo_gdtha", "tm_score", "lddt", "bb_lddt", "rmsd")
DATASETS = {
    "casp14": (70, 62, "casp14_70"),
    "casp15": (22, 19, "casp15_single_chain_22"),
    "casp16": (21, 18, "casp16_single_chain_21"),
    "cameo22": (183, 68, "cameo22_183"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_rows(rows: list[dict], ids: list[str], label: str) -> list[dict]:
    indexed = {row["target_id"].lower(): row for row in rows}
    if len(indexed) != len(rows) or len(ids) != len(set(ids)) or set(indexed) != set(ids):
        raise ValueError(f"target IDs differ for {label}")
    return [{"target_id": target, **{metric: float(indexed[target][metric]) for metric in METRICS}} for target in ids]


def main() -> None:
    checkpoint = ROOT / "out/run-a-mamba3-1024-atom14/ckpt_0300000.pt"
    result = {
        "schema_version": 1,
        "model": "MambaFold Run A final EMA",
        "checkpoint_sha256": sha256(checkpoint),
        "sampler": {"method": "sde", "steps": 500, "tau": 0.01, "seed": 0},
        "scorer": "OpenStructure 2.9.1",
        "datasets": {},
    }
    for name, (full_n, admitted_n, stem) in DATASETS.items():
        root = SCORE_ROOT / name
        ids_path = AUDIT / f"{stem}-admitted.ids"
        ids = [line.strip().lower() for line in ids_path.read_text().splitlines() if line.strip()]
        if len(ids) != admitted_n:
            raise ValueError(f"wrong admitted count for {name}")
        baseline_path = DATA_ROOT / f"simplefold_360m_{name}_admitted.json"
        baseline = json.loads(baseline_path.read_text())
        baseline_rows = checked_rows(baseline["rows"], ids, f"{name} baseline")
        if baseline["target_ids_sha256"] != sha256(ids_path):
            raise ValueError(f"baseline target IDs changed for {name}")

        summary_name = "target-summary.json" if name == "casp15" else "summary.json"
        model_path = root / "scores-admitted" / summary_name
        model = json.loads(model_path.read_text())
        if name == "casp15":
            if model["target_count"] != admitted_n:
                raise ValueError("incomplete CASP15 target aggregation")
            model_rows = model["target_rows"]
        else:
            if model["expected_targets"] != admitted_n or not model["evaluation_complete"]:
                raise ValueError(f"incomplete admitted score for {name}")
            model_rows = model["rows"]
        model_rows = checked_rows(model_rows, ids, f"{name} model")

        full_path = root / "scores-full" / summary_name
        full = json.loads(full_path.read_text())
        if (full.get("target_count", full.get("expected_targets")) != full_n or
                full.get("successful_targets", full.get("target_count")) != full_n):
            raise ValueError(f"incomplete full score for {name}")
        rollout = json.loads((root / "rollout/rollout.json").read_text())
        if rollout["n_steps"] != 500 or rollout["seed"] != 0 or rollout["sde_tau"] != 0.01:
            raise ValueError(f"sampler mismatch for {name}")

        item = {
            "full_target_count": full_n,
            "admitted_target_count": admitted_n,
            "admitted_ids_sha256": sha256(ids_path),
            "model_summary_sha256": sha256(model_path),
            "baseline_summary_sha256": sha256(baseline_path),
            "model_full_means": {metric: float(full["metrics"][metric]["mean"]) for metric in METRICS},
            "admitted_means": {
                "mambafold": {metric: statistics.fmean(row[metric] for row in model_rows) for metric in METRICS},
                "simplefold_360m": {metric: statistics.fmean(row[metric] for row in baseline_rows) for metric in METRICS},
            },
            "target_rows": [
                {"target_id": target, "mambafold": model_rows[i], "simplefold_360m": baseline_rows[i]}
                for i, target in enumerate(ids)
            ],
        }
        result["datasets"][name] = item

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    out = DATA_ROOT / "matched_results.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    for name, item in result["datasets"].items():
        print(name, item["admitted_target_count"],
              {model: {metric: round(values[metric], 4) for metric in ("tm_score", "lddt")}
               for model, values in item["admitted_means"].items()})


if __name__ == "__main__":
    main()
