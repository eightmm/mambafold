"""Run MambaFold inference on a benchmark id list (mono + multi).

Reads a text file of PDB IDs (one per line, e.g. benchmarks/sets/t1_quick.txt),
samples one prediction per id with EqM Euler, and writes paired PDBs:

    <out_dir>/<pdb_id>_pred.pdb   # model prediction (multi-chain, Kabsch-aligned to GT)
    <out_dir>/<pdb_id>_gt.pdb     # ground truth     (multi-chain)

Both files use the same per-residue → chain-letter mapping so DockQ /
TM-align / lDDT can pair them up directly without --mapping plumbing.

Usage:
    PYTHONPATH=src .venv/bin/python benchmarks/run_inference.py \
        --ckpt outputs/train/<phase>/ckpt_latest.pt \
        --ids  benchmarks/sets/t1_quick.txt \
        --out  benchmarks/results/<phase>_t1 \
        [--max_length 1024] [--n_steps 50] [--no_ema]
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
sys.path.insert(0, str(REPO / "scripts"))

from mambafold.data.constants import CA_ATOM_ID, COORD_SCALE, ID_TO_AA, RESIDUE_ATOMS
from mambafold.data.dataset import RCSBDataset
from mambafold.data.transforms import center_and_scale
from mambafold.data.types import ProteinBatch
from mambafold.sampling import sample_euler
from mambafold.train.distributed import enable_cuda_perf_flags
from mambafold.train.trainer import load_from_checkpoint
from mambafold.utils.geometry import kabsch_align


# A..Z then a..z then 0..9 — DockQ accepts any single-char chain id
_CHAIN_LETTERS = (
    [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [chr(c) for c in range(ord("0"), ord("9") + 1)]
)


def save_pdb_multichain(coords_aa, res_type_ids, atom_mask, b_factors, chain_id, out_path):
    """Write all-atom PDB with one chain per unique chain_id value.

    Args:
        coords_aa:    [L, A, 3] coords in Å.
        res_type_ids: [L] int residue indices (AA_TO_ID values).
        atom_mask:    [L, A] bool — atom slots to emit.
        b_factors:    [L, A] float — written into the B-factor column.
        chain_id:     [L] int — per-residue chain index (0..n_chains-1).
        out_path:     Path.
    """
    lines, serial = [], 1
    L = coords_aa.shape[0]
    # Group residues by chain in input order, emit TER between chains.
    boundaries = [0]
    for i in range(1, L):
        if chain_id[i] != chain_id[i - 1]:
            boundaries.append(i)
    boundaries.append(L)

    for bi in range(len(boundaries) - 1):
        a, b = boundaries[bi], boundaries[bi + 1]
        cid_int = int(chain_id[a])
        if cid_int >= len(_CHAIN_LETTERS):
            cid_int = cid_int % len(_CHAIN_LETTERS)  # paranoid wrap; >62 chains is exotic
        ch_letter = _CHAIN_LETTERS[cid_int]
        # Per-chain residue numbering starts at 1 (PDB convention)
        for r_local, r in enumerate(range(a, b), start=1):
            res_id = int(res_type_ids[r])
            res_name = ID_TO_AA.get(res_id, "UNK")
            atoms = RESIDUE_ATOMS.get(res_name, RESIDUE_ATOMS["UNK"])
            for slot, atom_name in enumerate(atoms):
                if slot >= atom_mask.shape[1] or not atom_mask[r, slot]:
                    continue
                x, y, z = (float(coords_aa[r, slot, i]) for i in range(3))
                b_fac = max(-99.99, min(999.99, float(b_factors[r, slot])))
                element = atom_name[0]
                an = atom_name if len(atom_name) >= 4 else f" {atom_name:<3s}"
                lines.append(
                    f"ATOM  {serial:>5d} {an:<4s} {res_name:>3s} {ch_letter}"
                    f"{r_local:>4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00{b_fac:6.2f}"
                    f"          {element:>2s}\n"
                )
                serial += 1
        # TER record marks the end of a chain
        lines.append(f"TER   {serial:>5d}      {res_name:>3s} {ch_letter}{r_local:>4d}\n")
        serial += 1
    lines.append("END\n")
    Path(out_path).write_text("".join(lines))


def make_batch(x, ex, gamma_cur, device):
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
        is_nterm=ex.is_nterm.unsqueeze(0).to(device),
        is_cterm=ex.is_cterm.unsqueeze(0).to(device),
        x_clean=ex.coords.unsqueeze(0).to(device),
        x_gamma=x.unsqueeze(0),
        eps=torch.zeros_like(x).unsqueeze(0),
        gamma=torch.tensor([[[[float(gamma_cur)]]]]).to(device),
        esm=ex.esm.unsqueeze(0).to(device) if ex.esm is not None else None,
    )


def kabsch_align_to_gt(pred_aa, true_aa, mask_la):
    flat_p = pred_aa.reshape(-1, 3)
    flat_t = true_aa.reshape(-1, 3)
    flat_m = mask_la.reshape(-1)
    if int(flat_m.sum()) < 3:
        return pred_aa
    _, R, _ = kabsch_align(flat_p[flat_m], flat_t[flat_m])
    cp = flat_p[flat_m].mean(0)
    ct = flat_t[flat_m].mean(0)
    aligned = ((flat_p - cp) @ R.T) + ct
    return aligned.reshape(pred_aa.shape)


def main():
    enable_cuda_perf_flags()
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ids", required=True, help="text file of PDB IDs, one per line")
    p.add_argument("--out", required=True)
    p.add_argument("--data_dir", default="data/rcsb")
    p.add_argument("--esm_dir", default=None,
                   help="optional ESM dir (must match training: leave unset for no-ESM models)")
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--n_seeds", type=int, default=1,
                   help="Number of independent samples per target. Each is written as "
                        "`<pid>_pred_seed<i>.pdb`; the seed-0 file is also linked at "
                        "`<pid>_pred.pdb` so existing scoring code keeps working.")
    p.add_argument("--seed_offset", type=int, default=0,
                   help="Sampling seeds = [seed_offset, seed_offset+1, ...]")
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--no_ema", dest="use_ema", action="store_false")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] ckpt={args.ckpt} ema={args.use_ema} device={device}")
    model = load_from_checkpoint(args.ckpt, device, use_ema=args.use_ema)
    model.eval()

    ids = [s.strip() for s in Path(args.ids).read_text().split() if s.strip()]
    print(f"[load] eval ids={len(ids)} from {args.ids}")

    # One dataset that we'll point at individual files.
    ds = RCSBDataset(
        args.data_dir,
        max_length=args.max_length,
        min_length=10,
        min_obs_ratio=0.0,
        esm_dir=args.esm_dir,
    )

    summary_rows: list[dict] = []
    t0 = time.time()
    for i, pid in enumerate(ids):
        npz_path = REPO / args.data_dir / pid[1:3] / f"{pid}.npz"
        if not npz_path.exists():
            print(f"[skip] {pid}: missing npz")
            continue
        ds.files = [npz_path]
        try:
            ex = ds[0]
        except Exception as e:
            print(f"[skip] {pid}: load fail {e}")
            continue
        if ex is None:
            print(f"[skip] {pid}: no valid example")
            continue

        ex_c = center_and_scale(ex)
        L = ex.seq_len
        n_chains = int(ex.chain_id.max().item()) + 1 if ex.chain_id is not None else 1

        aa_mask = (ex_c.atom_mask & ex_c.observed_mask).numpy().astype(bool)
        true_aa = ex_c.coords.numpy() * COORD_SCALE
        res_type = ex_c.res_type.numpy()
        atom_mask_np = ex_c.atom_mask.numpy().astype(bool)
        chain_id_np = ex_c.chain_id.numpy() if ex.chain_id is not None else np.zeros(L, dtype=np.int64)

        # B-factor: zero (no per-atom pLDDT calc here — scoring step computes lDDT)
        b_zero = np.zeros_like(aa_mask, dtype=np.float32)

        # GT written once
        save_pdb_multichain(true_aa, res_type, atom_mask_np, b_zero,
                            chain_id_np, out_dir / f"{pid}_gt.pdb")

        n_ok = 0
        seeds = list(range(args.seed_offset, args.seed_offset + args.n_seeds))
        for si, sd in enumerate(seeds):
            try:
                with torch.no_grad():
                    _, pred_aa, _, _ = sample_euler(
                        model, ex, lambda x, g: make_batch(x, ex_c, g, device),
                        n_steps=args.n_steps, seed=sd, device=device,
                    )
            except torch.cuda.OutOfMemoryError:
                print(f"[oom] {pid}: L={L} chains={n_chains} seed={sd}")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"[err] {pid} seed={sd}: {type(e).__name__}: {e}")
                continue

            pred_aa_aligned = kabsch_align_to_gt(pred_aa, true_aa, aa_mask)
            seed_path = out_dir / f"{pid}_pred_seed{si}.pdb"
            save_pdb_multichain(pred_aa_aligned, res_type, atom_mask_np, b_zero,
                                chain_id_np, seed_path)
            # First successful seed also written as the canonical "<pid>_pred.pdb"
            if n_ok == 0:
                save_pdb_multichain(pred_aa_aligned, res_type, atom_mask_np, b_zero,
                                    chain_id_np, out_dir / f"{pid}_pred.pdb")
            n_ok += 1

        if n_ok == 0:
            # All seeds failed → drop the GT too so the scorer doesn't see a half-pair
            (out_dir / f"{pid}_gt.pdb").unlink(missing_ok=True)
            continue

        summary_rows.append({"pdb_id": pid, "L": int(L), "n_chains": n_chains, "n_seeds_ok": n_ok})
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(ids) - i - 1)
        print(f"[{i+1:>4}/{len(ids)}] {pid}  L={L:>5}  chains={n_chains:>2}  "
              f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s")

    (out_dir / "manifest.json").write_text(json.dumps({
        "ckpt": str(args.ckpt),
        "ids_file": str(args.ids),
        "n_steps": args.n_steps,
        "max_length": args.max_length,
        "use_ema": args.use_ema,
        "n_predicted": len(summary_rows),
        "rows": summary_rows,
    }, indent=2))
    print(f"[done] predictions written: {len(summary_rows)} → {out_dir}")


if __name__ == "__main__":
    main()
