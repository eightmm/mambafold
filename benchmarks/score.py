"""Score predicted PDBs against ground-truth PDBs.

Designed to run inside `tools/scoring_venv/` (numpy<2 + DockQ + tmtools + biotite).

Per-target metrics:
  monomer  → Cα-lDDT, GDT-TS-like proxy via TM-score, all-atom RMSD
  multimer → DockQ (best of all chain-pair mappings), interface lDDT, TM-score (per-chain)

Inputs:
  --in_dir   directory with `<pdb_id>_pred.pdb` and `<pdb_id>_gt.pdb` pairs
             (produced by benchmarks/run_inference.py)
  --out      output JSON (one row per id)

Usage:
  tools/scoring_venv/bin/python benchmarks/score.py \
      --in_dir benchmarks/results/<run> \
      --out    benchmarks/results/<run>/scores.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np


# ─── lDDT (pure numpy on Cα) ──────────────────────────────────────────────

def ca_lddt(pred_ca: np.ndarray, true_ca: np.ndarray, cutoff: float = 15.0) -> float:
    """Standard lDDT on Cα — cutoff 15 Å, thresholds 0.5/1/2/4 Å."""
    if len(pred_ca) < 2 or len(pred_ca) != len(true_ca):
        return float("nan")
    dp = np.linalg.norm(pred_ca[:, None] - pred_ca[None], axis=-1)
    dt = np.linalg.norm(true_ca[:, None] - true_ca[None], axis=-1)
    np.fill_diagonal(dt, np.inf)  # exclude self-pair
    pair = dt < cutoff
    diff = np.abs(dp - dt)
    if not pair.any():
        return float("nan")
    scores = []
    for thr in (0.5, 1.0, 2.0, 4.0):
        scores.append(((diff < thr) & pair).sum() / pair.sum())
    return float(np.mean(scores))


def aa_rmsd(pred: np.ndarray, true: np.ndarray) -> float:
    """Plain (no-Kabsch) RMSD; assumes pred already aligned to true."""
    if len(pred) != len(true) or len(pred) == 0:
        return float("nan")
    d = pred - true
    return float(np.sqrt((d * d).sum(axis=-1).mean()))


# ─── Parse PDB into (chain → Cα coords, all-atom coords) ──────────────────

def parse_pdb(path: Path):
    """Return dict {chain: {'ca': [N,3], 'all': [M,3], 'res_seq': [N,]}}."""
    chains: dict[str, dict] = {}
    seen_res: dict[str, set] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        chain = line[21]
        res_seq = int(line[22:26])
        x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        d = chains.setdefault(chain, {"ca": [], "all": [], "res_seq": []})
        sk = seen_res.setdefault(chain, set())
        d["all"].append((x, y, z))
        if atom_name == "CA" and res_seq not in sk:
            d["ca"].append((x, y, z))
            d["res_seq"].append(res_seq)
            sk.add(res_seq)
    out = {}
    for c, d in chains.items():
        out[c] = {
            "ca":      np.asarray(d["ca"], dtype=np.float32) if d["ca"] else np.empty((0, 3), np.float32),
            "all":     np.asarray(d["all"], dtype=np.float32) if d["all"] else np.empty((0, 3), np.float32),
            "res_seq": np.asarray(d["res_seq"], dtype=np.int32),
        }
    return out


def concat_ca(parsed) -> tuple[np.ndarray, list[str]]:
    """Concatenate Cα over chains (sorted by chain letter), return (coords, chain_label_per_residue)."""
    cs = sorted(parsed.keys())
    ca = np.concatenate([parsed[c]["ca"] for c in cs], axis=0) if cs else np.empty((0, 3), np.float32)
    labels = []
    for c in cs:
        labels.extend([c] * parsed[c]["ca"].shape[0])
    return ca, labels


def interface_lddt(pred_ca, true_ca, chain_labels, cutoff: float = 15.0) -> float:
    """lDDT restricted to inter-chain residue pairs."""
    if len(pred_ca) != len(true_ca) or len(pred_ca) < 2:
        return float("nan")
    dp = np.linalg.norm(pred_ca[:, None] - pred_ca[None], axis=-1)
    dt = np.linalg.norm(true_ca[:, None] - true_ca[None], axis=-1)
    same_chain = np.array([[a == b for b in chain_labels] for a in chain_labels])
    cross = ~same_chain
    pair = (dt < cutoff) & cross
    if not pair.any():
        return float("nan")
    diff = np.abs(dp - dt)
    scores = []
    for thr in (0.5, 1.0, 2.0, 4.0):
        scores.append(((diff < thr) & pair).sum() / pair.sum())
    return float(np.mean(scores))


# ─── TM-score (tmtools, pure python wrapper around TMalign C++) ──────────

def tm_score_single(pred_ca: np.ndarray, true_ca: np.ndarray) -> float:
    """TM-score over one Cα trace using tmtools (TM-align C++ binding)."""
    try:
        from tmtools import tm_align
    except ImportError:
        return float("nan")
    if len(pred_ca) < 5 or len(true_ca) < 5:
        return float("nan")
    L = min(len(pred_ca), len(true_ca))
    s = "A" * L  # equal-length placeholder; tmtools needs sequences of matched length
    res = tm_align(pred_ca[:L], true_ca[:L], s, s)
    return float(res.tm_norm_chain1)


def tm_score_per_chain(parsed_pred: dict, parsed_gt: dict) -> dict:
    """Per-chain TM-score (intra-chain fold quality), reported as mean over chains.

    For multimers TM-score on bulk-concatenated coords entangles fold quality
    with relative pose; this function isolates fold quality per chain. Pose is
    captured separately by DockQ / interface-lDDT.
    """
    common = sorted(set(parsed_pred) & set(parsed_gt))
    scores = []
    per_chain = {}
    for c in common:
        s = tm_score_single(parsed_pred[c]["ca"], parsed_gt[c]["ca"])
        per_chain[c] = s
        if not np.isnan(s):
            scores.append(s)
    return {
        "per_chain": per_chain,
        "mean": float(np.mean(scores)) if scores else float("nan"),
        "min":  float(np.min(scores))  if scores else float("nan"),
    }


# ─── DockQ (multimer) ─────────────────────────────────────────────────────

def dockq_score(pred_pdb: Path, gt_pdb: Path) -> dict:
    """Run DockQ, parse the simple 'best' summary out of its dict result.

    DockQ 2.x exposes `run_on_all_native_interfaces(model, native, ...)` returning
    {(c1,c2): {DockQ, fnat, iRMS, LRMS, ...}}. We aggregate to a single
    per-target score = mean(DockQ over interfaces) and also keep best.
    """
    try:
        from DockQ.DockQ import load_PDB, run_on_all_native_interfaces
    except ImportError:
        return {"error": "DockQ not installed"}
    try:
        model = load_PDB(str(pred_pdb))
        native = load_PDB(str(gt_pdb))
        res, total = run_on_all_native_interfaces(model, native)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    if not res:
        return {"n_interfaces": 0, "dockq_mean": float("nan"), "dockq_best": float("nan")}
    qs = [v["DockQ"] for v in res.values()]
    return {
        "n_interfaces": len(qs),
        "dockq_mean": float(np.mean(qs)),
        "dockq_best": float(np.max(qs)),
        "interfaces": {str(k): {kk: float(v[kk]) for kk in ("DockQ", "fnat", "iRMS", "LRMS") if kk in v}
                       for k, v in res.items()},
    }


# ─── Main loop ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    pred_files = sorted(in_dir.glob("*_pred.pdb"))
    print(f"[load] pred files = {len(pred_files)}")

    rows = []
    t0 = time.time()
    for i, pred in enumerate(pred_files):
        pid = pred.stem.replace("_pred", "")
        gt = in_dir / f"{pid}_gt.pdb"
        if not gt.exists():
            print(f"[skip] {pid}: no GT")
            continue

        try:
            P = parse_pdb(pred); G = parse_pdb(gt)
        except Exception as e:
            print(f"[skip] {pid}: parse fail {e}")
            continue

        pred_ca, lbl_p = concat_ca(P)
        true_ca, lbl_g = concat_ca(G)

        if pred_ca.shape != true_ca.shape:
            print(f"[warn] {pid}: shape mismatch pred={pred_ca.shape} gt={true_ca.shape} — best-effort")
            n = min(len(pred_ca), len(true_ca))
            pred_ca, true_ca = pred_ca[:n], true_ca[:n]
            lbl_p = lbl_p[:n]

        n_chains = len(set(lbl_p))
        is_multi = n_chains > 1

        row = {
            "pdb_id": pid,
            "n_chains": n_chains,
            "n_residues": int(len(pred_ca)),
            "ca_lddt": ca_lddt(pred_ca, true_ca),
            "ca_rmsd": aa_rmsd(pred_ca, true_ca),
        }
        if is_multi:
            tm = tm_score_per_chain(P, G)
            row["tm_score"] = tm["mean"]      # mean over chains (intra-chain fold quality)
            row["tm_score_min_chain"] = tm["min"]
            row["tm_score_per_chain"] = tm["per_chain"]
            row["interface_lddt"] = interface_lddt(pred_ca, true_ca, lbl_p)
            row["dockq"] = dockq_score(pred, gt)
        else:
            row["tm_score"] = tm_score_single(pred_ca, true_ca)

        rows.append(row)
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(pred_files) - i - 1)
        msg = f"[{i+1:>4}/{len(pred_files)}] {pid:<6} chains={n_chains} L={row['n_residues']:>5} " \
              f"lDDT={row['ca_lddt']:.3f} TM={row['tm_score']:.3f}"
        if is_multi:
            d = row["dockq"]
            if "dockq_mean" in d:
                msg += f" if_l={row['interface_lddt']:.3f} DockQ={d['dockq_mean']:.3f}({d['n_interfaces']}I)"
            else:
                msg += f" if_l={row['interface_lddt']:.3f} DockQ=ERR({d.get('error','?')})"
        msg += f"  eta={eta:6.1f}s"
        print(msg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))

    # Summary
    if rows:
        mono = [r for r in rows if r["n_chains"] == 1]
        multi = [r for r in rows if r["n_chains"] > 1]
        def mean(xs): return float(np.nanmean(xs)) if xs else float("nan")
        print("\n=== SUMMARY ===")
        if mono:
            print(f"  monomer  N={len(mono):>3} | "
                  f"lDDT={mean([r['ca_lddt'] for r in mono]):.3f} "
                  f"TM={mean([r['tm_score'] for r in mono]):.3f} "
                  f"RMSD={mean([r['ca_rmsd'] for r in mono]):.2f}")
        if multi:
            dq = [r['dockq'].get('dockq_mean', float('nan')) for r in multi if isinstance(r.get('dockq'), dict)]
            print(f"  multimer N={len(multi):>3} | "
                  f"lDDT={mean([r['ca_lddt'] for r in multi]):.3f} "
                  f"if_lDDT={mean([r['interface_lddt'] for r in multi]):.3f} "
                  f"TM={mean([r['tm_score'] for r in multi]):.3f} "
                  f"DockQ={mean(dq):.3f}")
    print(f"\n[done] scores → {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
