"""GT-as-pred sanity check.

Picks a small set of ids from t0_smoke, loads each via RCSBDataset, and writes
GT coordinates as BOTH `<pid>_gt.pdb` and `<pid>_pred.pdb`. The scorer should
return lDDT≈1.0, TM≈1.0, RMSD≈0.0, DockQ≈1.0 on every target.

CPU-only (no model load).

Usage:
    .venv/bin/python benchmarks/_sanity_gt_as_pred.py
    tools/scoring_venv/bin/python benchmarks/score.py \
        --in_dir benchmarks/results/_sanity_gt --out benchmarks/results/_sanity_gt/scores.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks"))

from mambafold.data.constants import COORD_SCALE
from mambafold.data.dataset import RCSBDataset
from mambafold.data.transforms import center_and_scale
from run_inference import save_pdb_multichain


def main():
    ids = (REPO / "benchmarks/sets/t0_smoke.txt").read_text().split()
    # Pick first 6 with a mix of chain counts
    picks = ids[:6]
    out_dir = REPO / "benchmarks/results/_sanity_gt"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = RCSBDataset("data/rcsb", max_length=2048, min_length=10, min_obs_ratio=0.0)

    rows = []
    for pid in picks:
        npz = REPO / f"data/rcsb/{pid[1:3]}/{pid}.npz"
        if not npz.exists():
            print(f"[skip] {pid}: no npz")
            continue
        ds.files = [npz]
        try:
            ex = ds[0]
        except Exception as e:
            print(f"[skip] {pid}: {e}")
            continue
        if ex is None:
            continue

        ex_c = center_and_scale(ex)
        true_aa = ex_c.coords.numpy() * COORD_SCALE
        res_type = ex_c.res_type.numpy()
        atom_mask = ex_c.atom_mask.numpy().astype(bool)
        chain_id = ex_c.chain_id.numpy() if ex.chain_id is not None else np.zeros(ex.seq_len, dtype=np.int64)
        b_zero = np.zeros_like(atom_mask, dtype=np.float32)
        n_chains = int(chain_id.max()) + 1

        save_pdb_multichain(true_aa, res_type, atom_mask, b_zero, chain_id,
                            out_dir / f"{pid}_gt.pdb")
        save_pdb_multichain(true_aa, res_type, atom_mask, b_zero, chain_id,
                            out_dir / f"{pid}_pred.pdb")
        rows.append((pid, ex.seq_len, n_chains))
        print(f"[ok] {pid}: L={ex.seq_len} chains={n_chains}")

    print(f"\n[done] {len(rows)} pairs at {out_dir}")
    print("Now run:")
    print(f"  tools/scoring_venv/bin/python benchmarks/score.py "
          f"--in_dir {out_dir.relative_to(REPO)} --out {out_dir.relative_to(REPO)}/scores.json")


if __name__ == "__main__":
    main()
