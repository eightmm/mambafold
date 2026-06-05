"""Score single-chain predicted PDBs against ground-truth PDBs.

Inputs are `<pdb_id>_pred.pdb` and `<pdb_id>_gt.pdb` pairs produced by
benchmarks/run_inference.py. Metrics: CA-lDDT, TM-score, all-atom RMSD.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def ca_lddt(pred_ca: np.ndarray, true_ca: np.ndarray, cutoff: float = 15.0) -> float:
    if len(pred_ca) < 2 or len(pred_ca) != len(true_ca):
        return float("nan")
    dp = np.linalg.norm(pred_ca[:, None] - pred_ca[None], axis=-1)
    dt = np.linalg.norm(true_ca[:, None] - true_ca[None], axis=-1)
    np.fill_diagonal(dt, np.inf)
    pair = dt < cutoff
    if not pair.any():
        return float("nan")
    diff = np.abs(dp - dt)
    return float(np.mean([((diff < thr) & pair).sum() / pair.sum() for thr in (0.5, 1.0, 2.0, 4.0)]))


def aa_rmsd(pred: np.ndarray, true: np.ndarray) -> float:
    if len(pred) != len(true) or len(pred) == 0:
        return float("nan")
    d = pred - true
    return float(np.sqrt((d * d).sum(axis=-1).mean()))


def parse_pdb(path: Path) -> dict:
    ca = []
    all_atoms = []
    seen_res = set()
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        chain = line[21]
        res_seq = int(line[22:26])
        x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        all_atoms.append((x, y, z))
        key = (chain, res_seq)
        if atom_name == "CA" and key not in seen_res:
            ca.append((x, y, z))
            seen_res.add(key)
    return {
        "ca": np.asarray(ca, dtype=np.float32) if ca else np.empty((0, 3), np.float32),
        "all": np.asarray(all_atoms, dtype=np.float32) if all_atoms else np.empty((0, 3), np.float32),
    }


def tm_score(pred_ca: np.ndarray, true_ca: np.ndarray) -> float:
    try:
        from tmtools import tm_align
    except ImportError:
        return float("nan")
    if len(pred_ca) < 5 or len(true_ca) < 5:
        return float("nan")
    L = min(len(pred_ca), len(true_ca))
    seq = "A" * L
    res = tm_align(pred_ca[:L], true_ca[:L], seq, seq)
    return float(res.tm_norm_chain1)


def score_pair(pid: str, pred_pdb: Path, gt_pdb: Path) -> dict:
    pred = parse_pdb(pred_pdb)
    gt = parse_pdb(gt_pdb)
    L = min(len(pred["ca"]), len(gt["ca"]))
    n_all = min(len(pred["all"]), len(gt["all"]))
    return {
        "pdb_id": pid,
        "n_residues": int(L),
        "ca_lddt": ca_lddt(pred["ca"][:L], gt["ca"][:L]),
        "tm_score": tm_score(pred["ca"][:L], gt["ca"][:L]),
        "aa_rmsd": aa_rmsd(pred["all"][:n_all], gt["all"][:n_all]),
    }


def mean(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if not np.isnan(r[key])]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    rows = []
    t0 = time.time()
    for gt in sorted(in_dir.glob("*_gt.pdb")):
        pid = gt.name.removesuffix("_gt.pdb")
        pred = in_dir / f"{pid}_pred.pdb"
        if not pred.exists():
            print(f"[skip] {pid}: missing pred")
            continue
        try:
            row = score_pair(pid, pred, gt)
        except Exception as e:
            print(f"[err] {pid}: {type(e).__name__}: {e}")
            continue
        rows.append(row)
        print(
            f"[{len(rows):>4}] {pid:<6} L={row['n_residues']:>5} "
            f"lDDT={row['ca_lddt']:.3f} TM={row['tm_score']:.3f} "
            f"aaRMSD={row['aa_rmsd']:.2f} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )

    summary = {
        "n": len(rows),
        "ca_lddt": mean(rows, "ca_lddt"),
        "tm_score": mean(rows, "tm_score"),
        "aa_rmsd": mean(rows, "aa_rmsd"),
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(
        f"  single N={len(rows):>3} | lDDT={summary['ca_lddt']:.3f} "
        f"TM={summary['tm_score']:.3f} aaRMSD={summary['aa_rmsd']:.2f}"
    )
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
