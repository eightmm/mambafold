"""Evaluate Stage 1 CA predictions on single-chain targets.

This is intentionally CA-only: it samples the Stage 1 Euler trajectory,
globally aligns predicted CA to ground truth CA, and reports hard CA-lDDT/RMSD.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mambafold.data.constants import AA_TO_ID, CA_ATOM_ID, COORD_SCALE
from mambafold.data.dataset import RCSBDataset
from mambafold.data.transforms import center_and_scale
from mambafold.data.types import ProteinBatch
from mambafold.sampling.samplers import _stage1_run
from mambafold.train.distributed import enable_cuda_perf_flags
from mambafold.train.trainer import load_from_checkpoint


def make_batch(x, ex, t_cur, device):
    L = ex.seq_len
    return ProteinBatch(
        res_type=ex.res_type.unsqueeze(0).to(device),
        res_seq_nums=ex.res_seq_nums.unsqueeze(0).to(device),
        atom_type=ex.atom_type.unsqueeze(0).to(device),
        pair_type=ex.pair_type.unsqueeze(0).to(device),
        res_mask=torch.ones(1, L, dtype=torch.bool, device=device),
        atom_mask=ex.atom_mask.unsqueeze(0).to(device),
        valid_mask=(ex.atom_mask & ex.observed_mask).unsqueeze(0).to(device),
        ca_mask=(ex.atom_mask[:, CA_ATOM_ID] & ex.observed_mask[:, CA_ATOM_ID]).unsqueeze(0).to(device),
        chain_id=ex.chain_id.unsqueeze(0).to(device),
        entity_id=ex.entity_id.unsqueeze(0).to(device),
        sym_id=ex.sym_id.unsqueeze(0).to(device),
        is_nterm=ex.is_nterm.unsqueeze(0).to(device),
        is_cterm=ex.is_cterm.unsqueeze(0).to(device),
        x_clean=ex.coords.unsqueeze(0).to(device),
        x_t=x.unsqueeze(0),
        eps=torch.zeros_like(x).unsqueeze(0),
        t=torch.tensor([[[[float(t_cur)]]]], device=device),
        esm=ex.esm.unsqueeze(0).to(device) if ex.esm is not None else None,
    )


def kabsch_align_np(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    if len(pred) < 3:
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


def kabsch_rmsd_allow_reflection(pred: np.ndarray, true: np.ndarray) -> float:
    """Best-superposition RMSD allowing an improper rotation (reflection).
    For a mirror-image prediction this collapses to ~dRMSD, isolating the
    chirality error from genuine structural error."""
    if len(pred) < 3:
        return float("nan")
    p0 = pred - pred.mean(axis=0, keepdims=True)
    t0 = true - true.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(p0.T @ t0)
    r = u @ vt                                  # no det correction → reflection allowed
    d = (p0 @ r) - t0
    return float(np.sqrt((d * d).sum(axis=-1).mean()))


def ca_rmsd(pred: np.ndarray, true: np.ndarray) -> float:
    d = pred - true
    return float(np.sqrt((d * d).sum(axis=-1).mean()))


def ca_lddt(pred: np.ndarray, true: np.ndarray, cutoff: float = 15.0) -> float:
    if len(pred) < 2:
        return float("nan")
    dp = np.linalg.norm(pred[:, None] - pred[None], axis=-1)
    dt = np.linalg.norm(true[:, None] - true[None], axis=-1)
    np.fill_diagonal(dt, np.inf)
    pair = dt < cutoff
    if not pair.any():
        return float("nan")
    diff = np.abs(dp - dt)
    return float(np.mean([((diff < thr) & pair).sum() / pair.sum() for thr in (0.5, 1.0, 2.0, 4.0)]))


def ca_drmsd(pred: np.ndarray, true: np.ndarray) -> float:
    """Distance-matrix RMSD (Å) over all non-self Cα pairs — alignment-free topology."""
    if len(pred) < 2:
        return float("nan")
    dp = np.linalg.norm(pred[:, None] - pred[None], axis=-1)
    dt = np.linalg.norm(true[:, None] - true[None], axis=-1)
    iu = np.triu_indices(len(pred), k=1)
    return float(np.sqrt(np.mean((dp[iu] - dt[iu]) ** 2)))


def lr_contact_precision(pred: np.ndarray, true: np.ndarray,
                         sep: int = 24, thr: float = 8.0) -> float:
    """Long-range Cα contact precision: among |i-j|>sep pairs, fraction of
    predicted contacts (pred_d<thr Å) that are true contacts (true_d<thr Å)."""
    n = len(pred)
    if n < sep + 2:
        return float("nan")
    dp = np.linalg.norm(pred[:, None] - pred[None], axis=-1)
    dt = np.linalg.norm(true[:, None] - true[None], axis=-1)
    i, j = np.triu_indices(n, k=sep + 1)            # |i-j| > sep
    pred_c = dp[i, j] < thr
    true_c = dt[i, j] < thr
    n_pred = int(pred_c.sum())
    if n_pred == 0:
        return float("nan")
    return float((pred_c & true_c).sum() / n_pred)


def mean(xs: list[float]) -> float:
    arr = np.asarray(xs, dtype=np.float64)
    return float(np.nanmean(arr)) if len(arr) else float("nan")


def unique_protein_chain_origins(data, min_length, dedup=True):
    """Protein-chain origins (0-based among protein chains) with >= min_length
    standard residues. With `dedup`, collapse homomer copies — one origin per
    unique residue sequence. Mirrors RCSBDataset chain-keeping (mol_type==0)."""
    chains = data["chains"]
    residues = data["residues"]
    origins, seen, origin = [], set(), -1
    for ch in chains:
        if int(ch["mol_type"]) != 0:            # 0 = protein
            continue
        origin += 1
        r0 = int(ch["res_idx"])
        r1 = r0 + int(ch["res_num"])
        seq = [str(residues[i]["name"]) for i in range(r0, r1)
               if residues[i]["is_standard"] and str(residues[i]["name"]) in AA_TO_ID
               and str(residues[i]["name"]) != "UNK"]
        if len(seq) < min_length:
            continue
        key = tuple(seq)
        if dedup and key in seen:
            continue
        seen.add(key)
        origins.append(origin)
    return origins


def main():
    enable_cuda_perf_flags()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_dir", default="data/rcsb")
    ap.add_argument("--esm_dir", default=None)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--n_steps", type=int, default=50)
    ap.add_argument("--limit", type=int, default=10, help="max single-chain targets to score; <=0 means all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_ema", dest="use_ema", action="store_false")
    ap.set_defaults(use_ema=True)
    # Monomer-extraction eval: score every protein chain of each entry (not just
    # whole-single-chain entries), matching extract_monomer_chains training and
    # giving far more targets (esp. short chains). Homomer copies are deduped by
    # sequence so an N-mer counts its unique chains once.
    ap.add_argument("--extract_chains", action="store_true", default=False)
    ap.add_argument("--dedup_homomers", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.esm_dir is None:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        ck_args = ck.get("args", {})
        if not isinstance(ck_args, dict):
            ck_args = vars(ck_args)
        if ck_args.get("use_plm", False):
            args.esm_dir = ck_args.get("esm_dir") or "data/rcsb_esm"
            print(f"[esm] auto-detected from ckpt: esm_dir={args.esm_dir}")
        del ck

    print(f"[load] ckpt={args.ckpt} ema={args.use_ema} device={device}")
    model = load_from_checkpoint(args.ckpt, device, use_ema=args.use_ema)
    if type(model).__name__ != "MambaFoldStage1":
        raise RuntimeError(f"expected MambaFoldStage1 checkpoint, got {type(model).__name__}")
    model.eval()

    ds = RCSBDataset(
        args.data_dir,
        max_length=args.max_length,
        min_length=10,
        min_obs_ratio=0.0,
        esm_dir=args.esm_dir,
        single_chain_only=True,
    )
    ids = [s.strip() for s in Path(args.ids).read_text().split() if s.strip()]

    rows = []
    t0 = time.time()

    def score(ex, pid, origin=None):
        """Sample + score one example; append a row. Returns True if scored."""
        if ex is None:
            return False
        ex_c = center_and_scale(ex)
        ca_mask = (ex_c.atom_mask[:, CA_ATOM_ID] & ex_c.observed_mask[:, CA_ATOM_ID]).numpy().astype(bool)
        true_ca = ex_c.coords[:, CA_ATOM_ID, :].numpy()[ca_mask] * COORD_SCALE
        try:
            if device == "cuda":
                torch.cuda.synchronize()
            t_fold = time.time()
            with torch.no_grad():
                pred_ca_norm = _stage1_run(
                    model, ex,
                    lambda x, ti: make_batch(x, ex_c, ti, device),
                    n_steps=args.n_steps, n_recycle=0, recycle_t_start=0.5,
                    seed=args.seed, device=device,
                )[0]
            if device == "cuda":
                torch.cuda.synchronize()
            fold_s = time.time() - t_fold
        except torch.cuda.OutOfMemoryError:
            print(f"[oom] {pid}: L={ex.seq_len}")
            torch.cuda.empty_cache()
            return False
        except Exception as e:
            print(f"[err] {pid}: {type(e).__name__}: {e}")
            return False

        pred_raw = pred_ca_norm.float().cpu().numpy()[ca_mask] * COORD_SCALE
        pred_ca = kabsch_align_np(pred_raw, true_ca)
        rmsd_proper = ca_rmsd(pred_ca, true_ca)
        rmsd_reflect = kabsch_rmsd_allow_reflection(pred_raw, true_ca)
        row = {
            "pdb_id": pid,
            "origin": origin,                # protein-chain index (extract mode) or None
            "n_residues": int(len(true_ca)),
            "ca_lddt": ca_lddt(pred_ca, true_ca),
            "ca_rmsd": rmsd_proper,
            "ca_rmsd_mirror": min(rmsd_proper, rmsd_reflect),   # RMSD if chirality forgiven
            "is_mirror": bool(rmsd_reflect < rmsd_proper - 2.0),  # reflection helps ≥2Å
            "ca_drmsd": ca_drmsd(pred_ca, true_ca),
            "lr_contact_prec": lr_contact_precision(pred_ca, true_ca),
            "fold_s": round(fold_s, 3),     # wall-clock for the 50-step Euler sampling
        }
        rows.append(row)
        tag = f"{pid}" + (f":ch{origin}" if origin is not None else "")
        print(
            f"[{len(rows):>4}] {tag:<10} L={len(true_ca):>5} "
            f"lDDT={row['ca_lddt']:.3f} RMSD={row['ca_rmsd']:.2f} "
            f"dRMSD={row['ca_drmsd']:.2f} LRcontact={row['lr_contact_prec']:.3f} "
            f"fold={row['fold_s']:.2f}s elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
        return True

    for pid in ids:
        npz_path = REPO / args.data_dir / pid[1:3] / f"{pid}.npz"
        if not npz_path.exists():
            continue
        if args.extract_chains:
            # Score each unique-sequence protein chain of the entry.
            try:
                data = np.load(npz_path)
            except Exception:
                continue
            for origin in unique_protein_chain_origins(data, ds.min_length, args.dedup_homomers):
                try:
                    ex = ds._canonicalize(data, npz_path, only_chain_origin=origin)
                except Exception as e:
                    print(f"[skip] {pid}:ch{origin}: invalid ({type(e).__name__})")
                    continue
                score(ex, pid, origin)
                if args.limit > 0 and len(rows) >= args.limit:
                    break
        else:
            ds.files = [npz_path]
            try:
                ex = ds[0]
            except Exception as e:
                print(f"[skip] {pid}: not single-chain or invalid ({type(e).__name__})")
                continue
            score(ex, pid)
        if args.limit > 0 and len(rows) >= args.limit:
            break

    summary = {
        "ckpt": args.ckpt,
        "ids": args.ids,
        "n_steps": args.n_steps,
        "single_chain_only": True,
        "n": len(rows),
        "ca_lddt": mean([r["ca_lddt"] for r in rows]),
        "ca_rmsd": mean([r["ca_rmsd"] for r in rows]),
        "ca_drmsd": mean([r["ca_drmsd"] for r in rows]),
        "lr_contact_prec": mean([r["lr_contact_prec"] for r in rows]),
        "rows": rows,
    }
    (out_dir / "stage1_ca_scores.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(f"  single N={len(rows):>3} | lDDT={summary['ca_lddt']:.3f} "
          f"RMSD={summary['ca_rmsd']:.2f} dRMSD={summary['ca_drmsd']:.2f} "
          f"LRcontact={summary['lr_contact_prec']:.3f}")
    print(f"[done] {out_dir / 'stage1_ca_scores.json'}")


if __name__ == "__main__":
    main()
