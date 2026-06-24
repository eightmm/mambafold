"""Score prediction PDBs with SimpleFold-style aggregate metrics.

Inputs are `<target>_pred.pdb` and `<target>_gt.pdb` pairs produced by
`benchmarks/run_inference.py`.

Reported metrics:
- TM-score: tmtools TM-align on C-alpha traces.
- GDT-TS: mean fraction of C-alpha atoms within 1, 2, 4, 8 Angstrom after
  Kabsch alignment.
- LDDT: hard all-atom lDDT over common atom identities.
- LDDT-Ca: hard C-alpha lDDT.
- RMSD: Kabsch-aligned C-alpha RMSD.
- aaRMSD: Kabsch-aligned all-atom RMSD over common atom identities.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


THRESHOLDS_LDDT = (0.5, 1.0, 2.0, 4.0)
THRESHOLDS_GDT = (1.0, 2.0, 4.0, 8.0)


def kabsch_align(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    if len(pred) < 3 or len(pred) != len(true):
        return pred
    pc = pred.mean(axis=0, keepdims=True)
    tc = true.mean(axis=0, keepdims=True)
    p0 = pred - pc
    t0 = true - tc
    u, _, vt = np.linalg.svd(p0.T @ t0)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return p0 @ r + tc


def rmsd(pred: np.ndarray, true: np.ndarray) -> float:
    if len(pred) == 0 or len(pred) != len(true):
        return float("nan")
    d = pred - true
    return float(np.sqrt((d * d).sum(axis=-1).mean()))


def parse_pdb(path: Path) -> dict[str, Any]:
    atoms: dict[tuple[str, int, str, str], np.ndarray] = {}
    ca: dict[tuple[str, int, str], np.ndarray] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        atom = line[12:16].strip()
        chain = line[21]
        resseq = int(line[22:26])
        icode = line[26].strip()
        key_res = (chain, resseq, icode)
        key_atom = (chain, resseq, icode, atom)
        xyz = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=np.float32,
        )
        atoms[key_atom] = xyz
        if atom == "CA" and key_res not in ca:
            ca[key_res] = xyz
    return {"atoms": atoms, "ca": ca}


def common_coords(
    pred: dict[Any, np.ndarray],
    true: dict[Any, np.ndarray],
) -> tuple[list[Any], np.ndarray, np.ndarray]:
    keys = sorted(set(pred) & set(true))
    if not keys:
        empty = np.empty((0, 3), dtype=np.float32)
        return [], empty, empty
    p = np.stack([pred[k] for k in keys]).astype(np.float32)
    t = np.stack([true[k] for k in keys]).astype(np.float32)
    return keys, p, t


def hard_lddt(
    pred: np.ndarray,
    true: np.ndarray,
    residue_ids: list[Any] | None = None,
    cutoff: float = 15.0,
    block: int = 512,
) -> float:
    n = len(pred)
    if n < 2 or n != len(true):
        return float("nan")
    totals = np.zeros(len(THRESHOLDS_LDDT), dtype=np.float64)
    denom = 0
    res_ids = None
    if residue_ids is not None:
        # Keep residue identities as a true 1D array. `np.asarray(list_of_tuples,
        # dtype=object)` may still produce shape [N, 3], which breaks masking.
        res_ids = np.array(["|".join(map(str, r)) for r in residue_ids], dtype=object)
    for start in range(0, n, block):
        end = min(start + block, n)
        dp = np.linalg.norm(pred[start:end, None, :] - pred[None, :, :], axis=-1)
        dt = np.linalg.norm(true[start:end, None, :] - true[None, :, :], axis=-1)
        pair = dt < cutoff
        rows = np.arange(start, end)[:, None]
        cols = np.arange(n)[None, :]
        pair &= rows != cols
        if res_ids is not None:
            pair &= res_ids[start:end, None] != res_ids[None, :]
        n_pair = int(pair.sum())
        if n_pair == 0:
            continue
        diff = np.abs(dp - dt)
        for i, thr in enumerate(THRESHOLDS_LDDT):
            totals[i] += float(((diff < thr) & pair).sum())
        denom += n_pair
    if denom == 0:
        return float("nan")
    return float(np.mean(totals / denom))


def gdt_ts(pred_ca_aligned: np.ndarray, true_ca: np.ndarray) -> float:
    if len(pred_ca_aligned) == 0 or len(pred_ca_aligned) != len(true_ca):
        return float("nan")
    d = np.linalg.norm(pred_ca_aligned - true_ca, axis=-1)
    return float(np.mean([(d < thr).mean() for thr in THRESHOLDS_GDT]))


def tm_score(pred_ca: np.ndarray, true_ca: np.ndarray) -> float:
    try:
        from tmtools import tm_align
    except ImportError:
        return float("nan")
    if len(pred_ca) < 5 or len(pred_ca) != len(true_ca):
        return float("nan")
    seq = "A" * len(pred_ca)
    res = tm_align(pred_ca, true_ca, seq, seq)
    return float(res.tm_norm_chain1)


def finite(vals: list[float]) -> np.ndarray:
    arr = np.asarray(vals, dtype=np.float64)
    return arr[np.isfinite(arr)]


def stat(vals: list[float]) -> dict[str, float]:
    arr = finite(vals)
    if len(arr) == 0:
        return {"mean": float("nan"), "median": float("nan")}
    return {"mean": float(np.mean(arr)), "median": float(np.median(arr))}


def fmt_pair(s: dict[str, float]) -> str:
    a = s["mean"]
    b = s["median"]
    if math.isnan(a) or math.isnan(b):
        return "nan/nan"
    return f"{a:.3f}/{b:.3f}"


def score_pair(pid: str, pred_path: Path, gt_path: Path) -> dict[str, Any]:
    pred = parse_pdb(pred_path)
    true = parse_pdb(gt_path)

    ca_keys, pred_ca, true_ca = common_coords(pred["ca"], true["ca"])
    atom_keys, pred_atoms, true_atoms = common_coords(pred["atoms"], true["atoms"])

    pred_ca_aligned = kabsch_align(pred_ca, true_ca)
    pred_atoms_aligned = kabsch_align(pred_atoms, true_atoms)
    residue_ids = [(k[0], k[1], k[2]) for k in atom_keys]

    return {
        "pdb_id": pid,
        "n_residues": int(len(ca_keys)),
        "n_atoms": int(len(atom_keys)),
        "tm_score": tm_score(pred_ca, true_ca),
        "gdt_ts": gdt_ts(pred_ca_aligned, true_ca),
        "lddt": hard_lddt(pred_atoms, true_atoms, residue_ids=residue_ids),
        "lddt_ca": hard_lddt(pred_ca, true_ca),
        "rmsd": rmsd(pred_ca_aligned, true_ca),
        "aa_rmsd": rmsd(pred_atoms_aligned, true_atoms),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    rows = []
    t0 = time.time()
    for gt_path in sorted(in_dir.glob("*_gt.pdb")):
        pid = gt_path.name.removesuffix("_gt.pdb")
        pred_path = in_dir / f"{pid}_pred.pdb"
        if not pred_path.exists():
            print(f"[skip] {pid}: missing pred")
            continue
        try:
            row = score_pair(pid, pred_path, gt_path)
        except Exception as e:
            print(f"[err] {pid}: {type(e).__name__}: {e}")
            continue
        rows.append(row)
        print(
            f"[{len(rows):>4}] {pid:<10} L={row['n_residues']:>5} "
            f"TM={row['tm_score']:.3f} GDT={row['gdt_ts']:.3f} "
            f"lDDT={row['lddt']:.3f} lDDT-Ca={row['lddt_ca']:.3f} "
            f"RMSD={row['rmsd']:.2f} aaRMSD={row['aa_rmsd']:.2f} "
            f"elapsed={time.time() - t0:.1f}s",
            flush=True,
        )

    summary = {
        "n": len(rows),
        "tm_score": stat([r["tm_score"] for r in rows]),
        "gdt_ts": stat([r["gdt_ts"] for r in rows]),
        "lddt": stat([r["lddt"] for r in rows]),
        "lddt_ca": stat([r["lddt_ca"] for r in rows]),
        "rmsd": stat([r["rmsd"] for r in rows]),
        "aa_rmsd": stat([r["aa_rmsd"] for r in rows]),
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))

    print("\n=== SIMPLEFOLD-STYLE SUMMARY mean/median ===")
    print(f"N={len(rows)}")
    print(f"TM-score {fmt_pair(summary['tm_score'])}")
    print(f"GDT-TS   {fmt_pair(summary['gdt_ts'])}")
    print(f"LDDT     {fmt_pair(summary['lddt'])}")
    print(f"LDDT-Ca  {fmt_pair(summary['lddt_ca'])}")
    print(f"RMSD     {fmt_pair(summary['rmsd'])}")
    print(f"aaRMSD   {fmt_pair(summary['aa_rmsd'])}")
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
