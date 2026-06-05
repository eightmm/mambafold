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

from mambafold.data.constants import CA_ATOM_ID, COORD_SCALE
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


def mean(xs: list[float]) -> float:
    arr = np.asarray(xs, dtype=np.float64)
    return float(np.nanmean(arr)) if len(arr) else float("nan")


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
    for pid in ids:
        npz_path = REPO / args.data_dir / pid[1:3] / f"{pid}.npz"
        if not npz_path.exists():
            continue
        ds.files = [npz_path]
        try:
            ex = ds[0]
        except Exception as e:
            print(f"[skip] {pid}: not single-chain or invalid ({type(e).__name__})")
            continue
        if ex is None:
            continue

        ex_c = center_and_scale(ex)
        ca_mask = (ex_c.atom_mask[:, CA_ATOM_ID] & ex_c.observed_mask[:, CA_ATOM_ID]).numpy().astype(bool)
        true_ca = ex_c.coords[:, CA_ATOM_ID, :].numpy()[ca_mask] * COORD_SCALE

        try:
            with torch.no_grad():
                pred_ca_norm, _, _, _ = _stage1_run(
                    model, ex,
                    lambda x, ti: make_batch(x, ex_c, ti, device),
                    n_steps=args.n_steps,
                    n_recycle=0,
                    recycle_t_start=0.5,
                    seed=args.seed,
                    device=device,
                )
        except torch.cuda.OutOfMemoryError:
            print(f"[oom] {pid}: L={ex.seq_len}")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"[err] {pid}: {type(e).__name__}: {e}")
            continue

        pred_ca = pred_ca_norm.float().cpu().numpy()[ca_mask] * COORD_SCALE
        pred_ca = kabsch_align_np(pred_ca, true_ca)
        row = {
            "pdb_id": pid,
            "n_residues": int(len(true_ca)),
            "ca_lddt": ca_lddt(pred_ca, true_ca),
            "ca_rmsd": ca_rmsd(pred_ca, true_ca),
        }
        rows.append(row)
        print(
            f"[{len(rows):>3}] {pid:<6} L={len(true_ca):>5} "
            f"lDDT={row['ca_lddt']:.3f} RMSD={row['ca_rmsd']:.2f} "
            f"elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
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
        "rows": rows,
    }
    (out_dir / "stage1_ca_scores.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(f"  single N={len(rows):>3} | lDDT={summary['ca_lddt']:.3f} RMSD={summary['ca_rmsd']:.2f}")
    print(f"[done] {out_dir / 'stage1_ca_scores.json'}")


if __name__ == "__main__":
    main()
