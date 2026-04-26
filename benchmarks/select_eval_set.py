"""Select tiered evaluation sets from the date-cutoff holdout.

Reads `data/splits/holdout_ids.txt` (PDB IDs deposited after the train cutoff)
and partitions them into reproducible monomer/multimer evaluation tiers based
on residue length L (after RCSBDataset standardisation).

Tiers:
  T0 smoke (L ≤  512): 10 monomer +  10 multimer
  T1 quick (L ≤ 1024): 50 monomer +  50 multimer
  T2 full  (L ≤ 2048):150 monomer + 150 multimer

Outputs (deterministic with seed=0):
  benchmarks/sets/t0_smoke.txt
  benchmarks/sets/t1_quick.txt
  benchmarks/sets/t2_full.txt
  benchmarks/sets/manifest.tsv   (id, n_chains, n_residues, tier)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
HOLDOUT = REPO / "data/splits/holdout_ids.txt"
NPZ_DIR = REPO / "data/rcsb"
OUT_DIR = REPO / "benchmarks/sets"

TIERS = {
    "t0_smoke": {"max_L": 512,  "n_mono": 10,  "n_multi": 10},
    "t1_quick": {"max_L": 1024, "n_mono": 50,  "n_multi": 50},
    "t2_full":  {"max_L": 2048, "n_mono": 150, "n_multi": 150},
}


def npz_meta(pid: str) -> tuple[int, int] | None:
    """Return (n_chains, n_residues) or None if unreadable."""
    p = NPZ_DIR / pid[1:3] / f"{pid}.npz"
    if not p.exists():
        return None
    try:
        d = np.load(p, allow_pickle=True)
        return int(d["chains"].shape[0]), int(d["residues"].shape[0])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ids = HOLDOUT.read_text().split()
    print(f"[load] holdout ids = {len(ids)}")

    # Read meta for every id once (could parallelise; ~15k npz reads on local SSD ~ a couple min)
    print("[scan] reading npz metadata for every holdout id...")
    meta = {}
    bad = 0
    for i, pid in enumerate(ids):
        m = npz_meta(pid)
        if m is None:
            bad += 1
            continue
        meta[pid] = m
        if (i + 1) % 2000 == 0:
            print(f"  scanned {i+1}/{len(ids)}  bad={bad}")
    print(f"[scan] usable={len(meta)} bad={bad}")

    # Partition by chain count
    monos = sorted(p for p, (c, _) in meta.items() if c == 1)
    multis = sorted(p for p, (c, _) in meta.items() if c > 1)
    print(f"[split] monomers={len(monos)} multimers={len(multis)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    manifest_rows: list[str] = ["id\tn_chains\tn_residues\ttier"]

    for tier, spec in TIERS.items():
        max_L = spec["max_L"]
        n_mono, n_multi = spec["n_mono"], spec["n_multi"]

        cand_mono  = [p for p in monos  if meta[p][1] <= max_L]
        cand_multi = [p for p in multis if meta[p][1] <= max_L]

        # Sample without replacement; if pool too small fall back to whole pool.
        pick_mono  = rng.sample(cand_mono,  min(n_mono,  len(cand_mono)))
        pick_multi = rng.sample(cand_multi, min(n_multi, len(cand_multi)))
        picks = sorted(pick_mono + pick_multi)

        out = OUT_DIR / f"{tier}.txt"
        out.write_text("\n".join(picks) + "\n")
        print(f"[{tier}] L≤{max_L}: mono {len(pick_mono)}/{n_mono}  multi {len(pick_multi)}/{n_multi}  → {out.name}")

        for pid in picks:
            c, L = meta[pid]
            manifest_rows.append(f"{pid}\t{c}\t{L}\t{tier}")

    (OUT_DIR / "manifest.tsv").write_text("\n".join(manifest_rows) + "\n")
    print(f"[done] manifest at {OUT_DIR / 'manifest.tsv'}")


if __name__ == "__main__":
    main()
