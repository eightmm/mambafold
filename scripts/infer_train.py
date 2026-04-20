"""Sanity check: inference on proteins from the TRAIN set with ESM embeddings.

Measures RMSD / lDDT for training proteins across multiple seeds.
If RMSDs are low → model has learned what it saw (plateau is real).
If RMSDs are high → model capacity issue, more training won't help.

CRUCIAL: this script passes real ESM embeddings to the model (unlike
infer_seq.py / infer_2025.py which pass zeros).

Usage:
    PYTHONPATH=src python scripts/infer_train.py \\
        --ckpt outputs/train/26367/ckpt_latest.pt \\
        --data_dir data/rcsb \\
        --esm_dir data/rcsb_esm \\
        --pdb_ids 2olz,1eid,7l7s,8ki5,7n4x \\
        --out outputs/infer_train/450k
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mambafold.data.constants import (
    AA_TO_ID, CA_ATOM_ID, COORD_SCALE, ID_TO_AA, RESIDUE_ATOMS,
)
from mambafold.data.dataset import RCSBDataset
from mambafold.data.transforms import center_and_scale
from mambafold.data.types import ProteinBatch
from mambafold.losses.eqm import eqm_reconstruction_scale
from mambafold.train.trainer import load_from_checkpoint
from mambafold.utils.geometry import kabsch_align, kabsch_rmsd


def hard_lddt(pred_ca, true_ca, mask, cutoff=15.0):
    p, t = pred_ca[mask], true_ca[mask]
    if len(p) < 2:
        return 0.0
    dp = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1) + 1e-8)
    dt = np.sqrt(((t[:, None] - t[None]) ** 2).sum(-1) + 1e-8)
    diff = np.abs(dp - dt)
    return float(np.mean([np.mean(diff < thr) for thr in (0.5, 1.0, 2.0, 4.0)]))


def per_atom_lddt(pred_aa, true_aa, mask_la, cutoff=15.0):
    """Per-atom lDDT over all heavy atoms (flattened).

    Args:
        pred_aa, true_aa: [L, A, 3] coordinates.
        mask_la:          [L, A] bool — valid observed atoms.
        cutoff:           Neighbor cutoff (Å), standard lDDT uses 15 Å.

    Returns:
        [L, A] float array; lDDT ∈ [0, 1] for valid atoms, 0 for masked.
    """
    p = pred_aa.reshape(-1, 3)            # [N, 3]
    t = true_aa.reshape(-1, 3)
    m = mask_la.reshape(-1)               # [N]
    N = len(p)

    dp = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1) + 1e-8)
    dt = np.sqrt(((t[:, None] - t[None]) ** 2).sum(-1) + 1e-8)

    neigh = (dt < cutoff) & m[:, None] & m[None, :]
    np.fill_diagonal(neigh, False)
    diff = np.abs(dp - dt)

    scores = np.zeros(N, dtype=np.float32)
    thr = np.array([0.5, 1.0, 2.0, 4.0])
    for i in range(N):
        if not m[i]:
            continue
        nb = neigh[i]
        if not nb.any():
            continue
        d = diff[i, nb]   # [K]
        scores[i] = float(np.mean([(d < t_).mean() for t_ in thr]))
    return scores.reshape(mask_la.shape)


def save_pdb_allatom(coords_aa, res_type_ids, atom_mask, b_factors, out_path,
                     chain_id="A"):
    """Write all-atom PDB with per-atom B-factor.

    Args:
        coords_aa:    [L, A, 3] coords in Å.
        res_type_ids: [L] int residue indices (AA_TO_ID values).
        atom_mask:    [L, A] bool — slots to emit.
        b_factors:    [L, A] float — written into PDB B-factor field (as-is).
        out_path:     Path.
        chain_id:     Single char chain id.
    """
    lines, serial = [], 1
    for r in range(coords_aa.shape[0]):
        res_id = int(res_type_ids[r])
        res_name = ID_TO_AA.get(res_id, "UNK")
        atoms = RESIDUE_ATOMS.get(res_name, RESIDUE_ATOMS["UNK"])
        for slot, atom_name in enumerate(atoms):
            if slot >= atom_mask.shape[1] or not atom_mask[r, slot]:
                continue
            x, y, z = (float(coords_aa[r, slot, i]) for i in range(3))
            b = float(b_factors[r, slot])
            # B-factor field is 6.2f (max 99.99); clamp defensively
            b = max(-99.99, min(999.99, b))
            element = atom_name[0]  # first char is element for heavy atoms
            # PDB fixed-width ATOM record (ASCII columns per 1996 spec):
            #   6 record | 5 serial | 1 sp | 4 name | 1 altLoc | 3 resName
            #   1 sp | 1 chain | 4 resSeq | 1 iCode | 3 sp | 8 x | 8 y | 8 z
            #   6 occ | 6 B | 10 sp | 2 elem
            # Pad atom name to 4 cols: 3-letter names get a leading space.
            an = atom_name if len(atom_name) >= 4 else f" {atom_name:<3s}"
            lines.append(
                f"ATOM  {serial:>5d} {an:<4s} {res_name:>3s} {chain_id}"
                f"{r+1:>4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}"
                f"          {element:>2s}\n"
            )
            serial += 1
    lines.append("END\n")
    Path(out_path).write_text("".join(lines))


def _batch_from_x(x, ex, gamma_cur, device):
    """ProteinBatch with REAL ESM embedding (if present)."""
    L = ex.seq_len
    esm = ex.esm.unsqueeze(0).to(device) if ex.esm is not None else None
    return ProteinBatch(
        res_type=ex.res_type.unsqueeze(0).to(device),
        res_seq_nums=ex.res_seq_nums.unsqueeze(0).to(device),
        atom_type=ex.atom_type.unsqueeze(0).to(device),
        pair_type=ex.pair_type.unsqueeze(0).to(device),
        res_mask=torch.ones(1, L, dtype=torch.bool, device=device),
        atom_mask=ex.atom_mask.unsqueeze(0).to(device),
        valid_mask=(ex.atom_mask & ex.observed_mask).unsqueeze(0).to(device),
        ca_mask=(ex.atom_mask[:, CA_ATOM_ID] & ex.observed_mask[:, CA_ATOM_ID]).unsqueeze(0).to(device),
        x_clean=ex.coords.unsqueeze(0).to(device),
        x_gamma=x.unsqueeze(0),
        eps=torch.zeros_like(x).unsqueeze(0),
        gamma=torch.tensor([[[[float(gamma_cur)]]]]).to(device),
        esm=esm,
    )


@torch.no_grad()
def _x_hat(model, x, ex, gamma_cur, device):
    batch = _batch_from_x(x, ex, gamma_cur, device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
        pred = model(batch)
    scale = eqm_reconstruction_scale(batch.gamma)
    return (x.unsqueeze(0) - scale * pred)[0]


@torch.no_grad()
def sample_euler(model, example, n_steps=50, seed=0, device="cuda"):
    """Euler ODE. Returns (final_ca [L,3], final_aa [L,A,3], traj_ca, sched)."""
    torch.manual_seed(seed)
    ex = center_and_scale(example)
    mask_f = ex.atom_mask.unsqueeze(-1).float().to(device)
    x = torch.randn(ex.seq_len, ex.atom_mask.shape[1], 3, device=device) * mask_f
    sched = torch.linspace(0.0, 0.99, n_steps + 1, device=device)
    traj = []
    for i in range(n_steps):
        g = float(sched[i].clamp(min=1e-4))
        dg = float(sched[i + 1] - sched[i])
        x_hat = _x_hat(model, x, ex, g, device)
        velocity = (x_hat - x) / max(1.0 - g, 1e-4)
        x = (x + dg * velocity) * mask_f
        traj.append(x[:, CA_ATOM_ID, :].float().cpu().numpy() * COORD_SCALE)
    x_final = _x_hat(model, x, ex, float(sched[-1]), device)
    final_aa = x_final.float().cpu().numpy() * COORD_SCALE           # [L, A, 3]
    final_ca = final_aa[:, CA_ATOM_ID, :]                             # [L, 3]
    return final_ca, final_aa, np.array(traj, dtype=np.float32), sched.cpu().numpy()


@torch.no_grad()
def sample_nag(model, example, n_steps=50, seed=0, momentum=0.35, device="cuda"):
    """NAG ODE. Returns (final_ca, final_aa, traj_ca, sched)."""
    torch.manual_seed(seed)
    ex = center_and_scale(example)
    mask_f = ex.atom_mask.unsqueeze(-1).float().to(device)
    x = torch.randn(ex.seq_len, ex.atom_mask.shape[1], 3, device=device) * mask_f
    x_prev = x.clone()
    sched = torch.linspace(0.0, 0.99, n_steps + 1, device=device)
    traj = []
    for i in range(n_steps):
        g = float(sched[i].clamp(min=1e-4))
        dg = float(sched[i + 1] - sched[i])
        y = (x + momentum * (x - x_prev)) * mask_f
        x_hat_y = _x_hat(model, y, ex, g, device)
        velocity = (x_hat_y - y) / max(1.0 - g, 1e-4)
        x_new = (x + dg * velocity) * mask_f
        x_prev, x = x, x_new
        traj.append(x[:, CA_ATOM_ID, :].float().cpu().numpy() * COORD_SCALE)
    x_final = _x_hat(model, x, ex, float(sched[-1]), device)
    final_aa = x_final.float().cpu().numpy() * COORD_SCALE
    final_ca = final_aa[:, CA_ATOM_ID, :]
    return final_ca, final_aa, np.array(traj, dtype=np.float32), sched.cpu().numpy()


def load_one(data_dir, esm_dir, pdb_id):
    ds = RCSBDataset(data_dir, max_length=1024, min_length=10, min_obs_ratio=0.0,
                     esm_dir=esm_dir)
    ds.files = [Path(data_dir) / f"{pdb_id.lower()}.npz"]
    return ds[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", default="data/rcsb")
    p.add_argument("--esm_dir", default="data/rcsb_esm")
    p.add_argument("--pdb_ids", required=True, help="comma-separated")
    p.add_argument("--out", required=True)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--no_ema", dest="use_ema", action="store_false")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    model = load_from_checkpoint(args.ckpt, device, use_ema=args.use_ema)

    pdb_ids = [s.strip() for s in args.pdb_ids.split(",") if s.strip()]
    results = []

    print(f"\n{'PDB':<6}{'L':>5}{'ESM':>4}  "
          f"Cα-RMSD (Å)           All-atom RMSD (Å)     lDDT")
    print("─" * 72)

    for pid in pdb_ids:
        try:
            ex = load_one(args.data_dir, args.esm_dir, pid)
        except Exception as e:
            print(f"{pid:<8}  load fail: {e}")
            continue
        if ex is None:
            print(f"{pid:<8}  (no valid example)")
            continue
        has_esm = ex.esm is not None

        ex_c = center_and_scale(ex)
        ca_mask = (ex_c.atom_mask[:, CA_ATOM_ID] & ex_c.observed_mask[:, CA_ATOM_ID]).numpy().astype(bool)
        # All-atom mask: present & observed heavy atoms
        aa_mask = (ex_c.atom_mask & ex_c.observed_mask).numpy().astype(bool)  # [L, A]
        true_ca = ex_c.coords[:, CA_ATOM_ID, :].numpy() * COORD_SCALE
        true_aa = ex_c.coords.numpy() * COORD_SCALE                            # [L, A, 3]
        L = ex.seq_len

        def _flat_rmsd(pred_aa, mask_la):
            """All-atom Kabsch RMSD: flatten [L,A,3] → [L*A,3], mask [L,A] → [L*A]."""
            return kabsch_rmsd(pred_aa.reshape(-1, 3),
                               true_aa.reshape(-1, 3),
                               mask_la.reshape(-1))

        def _align_and_save(aa, method, si):
            """Kabsch-align pred AA to true AA, write PDB with per-atom lDDT B-factor."""
            # per-atom lDDT first (invariant to alignment)
            latt = per_atom_lddt(aa, true_aa, aa_mask) * 100.0   # pLDDT scale
            # align predicted onto true using observed atoms
            flat_p, flat_t, flat_m = aa.reshape(-1, 3), true_aa.reshape(-1, 3), aa_mask.reshape(-1)
            if flat_m.sum() >= 3:
                aligned_obs, R, _ = kabsch_align(flat_p[flat_m], flat_t[flat_m])
                # apply same R+translation to ALL atoms (including unobserved pad=0 slots)
                cp = flat_p[flat_m].mean(0); ct = flat_t[flat_m].mean(0)
                aligned = ((flat_p - cp) @ R.T) + ct
                aa_aln = aligned.reshape(aa.shape)
            else:
                aa_aln = aa
            pdb_path = out_dir / f"{pid}_{method}_seed{si}.pdb"
            save_pdb_allatom(aa_aln, ex_c.res_type.numpy(),
                             ex_c.atom_mask.numpy().astype(bool), latt, pdb_path)
            return latt

        eu_ca, eu_aa, eu_ld, eu_platt = [], [], [], []
        nag_ca, nag_aa, nag_ld, nag_platt = [], [], [], []
        for si in range(args.n_seeds):
            ca_e, aa_e, _, _ = sample_euler(model, ex, n_steps=args.n_steps, seed=si, device=device)
            ca_n, aa_n, _, _ = sample_nag(model, ex, n_steps=args.n_steps, seed=si, device=device)
            eu_ca.append(kabsch_rmsd(ca_e, true_ca, ca_mask))
            nag_ca.append(kabsch_rmsd(ca_n, true_ca, ca_mask))
            eu_aa.append(_flat_rmsd(aa_e, aa_mask))
            nag_aa.append(_flat_rmsd(aa_n, aa_mask))
            eu_ld.append(hard_lddt(ca_e, true_ca, ca_mask))
            nag_ld.append(hard_lddt(ca_n, true_ca, ca_mask))
            eu_platt.append(_align_and_save(aa_e, "euler", si))
            nag_platt.append(_align_and_save(aa_n, "nag", si))
        # save ground-truth too for easy visual compare
        save_pdb_allatom(true_aa, ex_c.res_type.numpy(),
                         ex_c.atom_mask.numpy().astype(bool),
                         np.full_like(aa_mask, 100.0, dtype=np.float32),
                         out_dir / f"{pid}_true.pdb")
        eu_ca, nag_ca = np.array(eu_ca), np.array(nag_ca)
        eu_aa, nag_aa = np.array(eu_aa), np.array(nag_aa)
        eu_ld, nag_ld = np.array(eu_ld), np.array(nag_ld)

        print(f"{pid:<6}{L:>5}{('Y' if has_esm else 'N'):>4}  "
              f"CA[e={eu_ca.mean():5.2f} n={nag_ca.mean():5.2f}]  "
              f"AA[e={eu_aa.mean():5.2f} n={nag_aa.mean():5.2f}]  "
              f"lDDT={eu_ld.mean():.3f}")

        results.append({
            "pdb_id": pid, "L": L, "has_esm": has_esm,
            "eu_ca_rmsd": eu_ca,   "nag_ca_rmsd": nag_ca,
            "eu_aa_rmsd": eu_aa,   "nag_aa_rmsd": nag_aa,
            "eu_lddt":    eu_ld,   "nag_lddt":    nag_ld,
        })

    # summary
    if results:
        eu_ca = np.concatenate([r["eu_ca_rmsd"] for r in results])
        nag_ca = np.concatenate([r["nag_ca_rmsd"] for r in results])
        eu_aa = np.concatenate([r["eu_aa_rmsd"] for r in results])
        nag_aa = np.concatenate([r["nag_aa_rmsd"] for r in results])
        eu_ld = np.concatenate([r["eu_lddt"] for r in results])
        nag_ld = np.concatenate([r["nag_lddt"] for r in results])
        print("─" * 72)
        print(f"  Overall (N={len(results)} proteins × {args.n_seeds} seeds):")
        print(f"    Euler:  Cα={eu_ca.mean():5.2f} Å  AA={eu_aa.mean():5.2f} Å  lDDT={eu_ld.mean():.3f}")
        print(f"    NAG  :  Cα={nag_ca.mean():5.2f} Å  AA={nag_aa.mean():5.2f} Å  lDDT={nag_ld.mean():.3f}")

    np.savez_compressed(
        out_dir / "infer_train.npz",
        pdb_ids=np.array([r["pdb_id"] for r in results]),
        seq_lens=np.array([r["L"] for r in results]),
        eu_ca_rmsd=np.array([r["eu_ca_rmsd"] for r in results]),
        nag_ca_rmsd=np.array([r["nag_ca_rmsd"] for r in results]),
        eu_aa_rmsd=np.array([r["eu_aa_rmsd"] for r in results]),
        nag_aa_rmsd=np.array([r["nag_aa_rmsd"] for r in results]),
        eu_lddt=np.array([r["eu_lddt"] for r in results]),
        nag_lddt=np.array([r["nag_lddt"] for r in results]),
    )
    print(f"\nSaved: {out_dir/'infer_train.npz'}")


if __name__ == "__main__":
    main()
