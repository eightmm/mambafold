#!/usr/bin/env python
"""Measure how redundant homomeric chain copies actually are.

`extract_monomer_chains` makes a chain the training unit, so a 300-chain
homomer becomes 300 examples. Across the admitted RCSB corpus, 269,053 of
503,669 chains (53.4%) carry a sequence that another chain in the *same entry*
already carries.

Whether that is 269,053 wasted examples or 269,053 useful ones turns on a
measurable question, not a plausible one. The FM target is rigid-aligned
(`engine.py`, `weighted_rigid_align`, on by default), so two copies related by
a pure rigid transform are the *same* target — the alignment removes the
difference before the loss sees it. What survives alignment is the
conformational difference between copies: different crystal contacts, different
loop placements, different disorder.

So this measures, over copies of one sequence inside one entry:

* Kabsch Cα RMSD between copies, on the residues both resolve
* how much the copies disagree about which residues are resolved at all

A median RMSD well under 1 Å means the copies are near-duplicates after the
loss's own alignment and the multiplicity is close to pure waste. A broad
distribution means the copies carry real conformational signal and should be
downweighted rather than dropped.

The second measurement decides *which* copy to keep if dropping: copies differ
in coverage, and keeping the best-resolved one is strictly more supervision
than keeping an arbitrary one.

Usage:
    python benchmarks/probe_homomer_redundancy.py --sample 3000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mambafold.data.constants import AA_3TO1, AA_TO_ID, RESIDUE_ATOMS  # noqa: E402

MOL_TYPE_PROTEIN = 0


def kabsch_rmsd(mobile: np.ndarray, ref: np.ndarray) -> float:
    """RMSD after optimal rigid superposition. Both [N, 3]."""
    mobile = mobile - mobile.mean(axis=0)
    ref = ref - ref.mean(axis=0)
    u, _, vt = np.linalg.svd(mobile.T @ ref)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    diff = (rotation @ mobile.T).T - ref
    return float(np.sqrt((diff**2).sum() / len(ref)))


def chain_ca(residues: np.ndarray, atoms: np.ndarray, start: int, end: int):
    """Canonical sequence plus per-residue Cα coordinate and resolved flag.

    Mirrors the loader's residue filter so the copies compared here are the
    copies the loader would have turned into examples.
    """
    sequence: list[str] = []
    coords: list[np.ndarray] = []
    resolved: list[bool] = []
    for i in range(start, end):
        residue = residues[i]
        name = str(residue["name"])
        if not bool(residue["is_standard"]) or name not in AA_TO_ID or name == "UNK":
            continue
        sequence.append(AA_3TO1[name])
        canonical = RESIDUE_ATOMS.get(name, [])
        atom_start = int(residue["atom_idx"])
        usable = min(int(residue["atom_num"]), len(canonical))
        ca_local = canonical.index("CA") if "CA" in canonical else -1
        if 0 <= ca_local < usable:
            atom = atoms[atom_start + ca_local]
            coords.append(np.asarray(atom["coords"], dtype=np.float64))
            resolved.append(bool(atom["is_present"]))
        else:
            coords.append(np.zeros(3))
            resolved.append(False)
    if not sequence:
        return None
    return "".join(sequence), np.asarray(coords), np.asarray(resolved, dtype=bool)


def _one(path_str: str) -> list[dict]:
    """Pairwise comparisons among same-sequence chains of one entry."""
    path = Path(path_str)
    try:
        data = np.load(path, allow_pickle=False)
        chains, residues, atoms = data["chains"], data["residues"], data["atoms"]
    except Exception:  # noqa: BLE001
        return []

    by_sequence: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for chain in chains:
        if int(chain["mol_type"]) != MOL_TYPE_PROTEIN:
            continue
        start = int(chain["res_idx"])
        parsed = chain_ca(residues, atoms, start, start + int(chain["res_num"]))
        if parsed is None:
            continue
        sequence, coords, resolved = parsed
        if len(sequence) < 20:
            continue
        by_sequence[sequence].append((coords, resolved))

    out = []
    for sequence, copies in by_sequence.items():
        if len(copies) < 2:
            continue
        coverage = [int(resolved.sum()) for _, resolved in copies]
        # Compare against the best-resolved copy: that is the one a
        # keep-the-best dedup would retain, so these RMSDs are exactly what
        # dropping the others would discard.
        best = int(np.argmax(coverage))
        ref_coords, ref_resolved = copies[best]
        for index, (coords, resolved) in enumerate(copies):
            if index == best:
                continue
            both = ref_resolved & resolved
            if int(both.sum()) < 20:
                continue
            out.append(
                {
                    "entry": path.stem,
                    "length": len(sequence),
                    "copies": len(copies),
                    "compared_residues": int(both.sum()),
                    "rmsd": kabsch_rmsd(coords[both], ref_coords[both]),
                    "coverage_best": coverage[best],
                    "coverage_other": coverage[index],
                }
            )
    return out


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = np.sort(np.asarray(values))
    return {
        "n": int(ordered.size),
        "mean": round(float(ordered.mean()), 3),
        "median": round(float(np.median(ordered)), 3),
        "p90": round(float(np.quantile(ordered, 0.9)), 3),
        "p99": round(float(np.quantile(ordered, 0.99)), 3),
        "max": round(float(ordered[-1]), 3),
        "frac_under_0.5A": round(float((ordered < 0.5).mean()), 4),
        "frac_under_1A": round(float((ordered < 1.0).mean()), 4),
        "frac_under_2A": round(float((ordered < 2.0).mean()), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--sample", type=int, default=3000,
                    help="entries to examine (0 = all admitted entries)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/audit/homomer_redundancy.json")
    ap.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK") or max(1, (os.cpu_count() or 2) - 1)),
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    farm = root / "data" / "rcsb_train"
    entries = [
        line.strip()
        for line in (root / "data" / "splits" / "admitted_all.txt").read_text().splitlines()
        if line.strip()
    ]
    if args.sample and args.sample < len(entries):
        rng = np.random.default_rng(args.seed)
        picked = rng.choice(len(entries), size=args.sample, replace=False)
        entries = [entries[i] for i in sorted(picked)]
    paths = [str(farm / e) for e in entries]

    print(f"== {len(paths)} entries, {args.workers} workers", flush=True)
    records: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, result in enumerate(pool.map(_one, paths, chunksize=32), 1):
            records.extend(result)
            if done % 20000 == 0:
                print(f"   {done}/{len(paths)}  {len(records)} comparisons", flush=True)

    rmsds = [r["rmsd"] for r in records]
    coverage_gain = [r["coverage_best"] - r["coverage_other"] for r in records]
    by_length = {
        "<=128": [r["rmsd"] for r in records if r["length"] <= 128],
        "129-256": [r["rmsd"] for r in records if 128 < r["length"] <= 256],
        "257-512": [r["rmsd"] for r in records if 256 < r["length"] <= 512],
        ">512": [r["rmsd"] for r in records if r["length"] > 512],
    }
    report = {
        "entries_examined": len(paths),
        "sample": args.sample,
        "seed": args.seed,
        "comparisons": len(records),
        "note": (
            "RMSD is Kabsch-aligned Cα, each redundant copy against the "
            "best-resolved copy of the same sequence in the same entry. The FM "
            "target is rigid-aligned, so this is the difference the loss would "
            "still see."
        ),
        "rmsd": summarize(rmsds),
        "rmsd_by_length": {k: summarize(v) for k, v in by_length.items()},
        "residues_gained_by_keeping_best": summarize([float(v) for v in coverage_gain]),
    }
    print(json.dumps(report, indent=2), flush=True)

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
