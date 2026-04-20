"""Inference on a 2025 PDB structure using a trained MambaFold checkpoint.

Downloads the mmCIF, converts to Boltz npz, runs EqM Euler/NAG sampling,
and saves results + visualization.

Usage (GPU node):
    PYTHONPATH=src python scripts/infer_2025.py \
        --ckpt outputs/train/23508/ckpt_0200000.pt \
        --pdb_id 10AF \
        --out outputs/infer_2025/10af \
        [--use_ema]
"""

import argparse
import sys
import urllib.request
from io import StringIO
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mambafold.data.constants import CA_ATOM_ID, COORD_SCALE, RESIDUE_ATOMS, AA_TO_ID
from mambafold.utils.geometry import kabsch_rmsd
from mambafold.data.dataset import RCSBDataset
from mambafold.data.transforms import center_and_scale
from mambafold.data.types import ProteinBatch
from mambafold.losses.eqm import eqm_reconstruction_scale
from mambafold.train.trainer import load_from_checkpoint

# reuse sampler helpers from export_inference
from export_inference import (
    hard_lddt,
    sample_eqm_euler,
    sample_eqm_nag,
    _eqm_x_hat,
)

from mambafold.data.constants import (
    BOLTZ_RESIDUES_DTYPE as RESIDUES_DTYPE,
    BOLTZ_ATOMS_DTYPE as ATOMS_DTYPE,
    BOLTZ_CHAINS_DTYPE as CHAINS_DTYPE,
)


def download_and_convert(pdb_id: str, out_npz: Path) -> Path:
    """Download mmCIF and convert to Boltz npz. Returns path to npz."""
    if out_npz.exists():
        print(f"npz already exists: {out_npz}")
        return out_npz

    from Bio.PDB.MMCIFParser import MMCIFParser

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url, timeout=30) as resp:
        cif_text = resp.read().decode("utf-8")

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("prot", StringIO(cif_text))
    model = next(structure.get_models())

    all_res, all_atoms, all_chains = [], [], []
    global_atom_idx = global_res_idx = 0

    for chain in model.get_chains():
        chain_res = []
        for res in chain.get_residues():
            het, _, _ = res.get_id()
            if het.strip():
                continue
            res_name = res.get_resname().strip()
            if res_name not in AA_TO_ID or res_name == "UNK":
                continue
            canon_atoms = RESIDUE_ATOMS.get(res_name, [])
            atom_start = global_atom_idx
            for aname in canon_atoms:
                rec = np.zeros(1, dtype=ATOMS_DTYPE)[0]
                if res.has_id(aname):
                    rec["coords"] = res[aname].get_coord().astype(np.float32)
                    rec["is_present"] = True
                all_atoms.append(rec)
                global_atom_idx += 1
            rec = np.zeros(1, dtype=RESIDUES_DTYPE)[0]
            rec["name"] = res_name
            rec["res_type"] = AA_TO_ID.get(res_name, 20)
            rec["res_idx"] = global_res_idx
            rec["atom_idx"] = atom_start
            rec["atom_num"] = len(canon_atoms)
            rec["is_standard"] = True
            rec["is_present"] = True
            chain_res.append(rec)
            all_res.append(rec)
            global_res_idx += 1
        if not chain_res:
            continue
        crec = np.zeros(1, dtype=CHAINS_DTYPE)[0]
        crec["name"] = chain.get_id()
        crec["mol_type"] = 0
        crec["res_idx"] = chain_res[0]["res_idx"]
        crec["res_num"] = len(chain_res)
        crec["atom_idx"] = chain_res[0]["atom_idx"]
        crec["atom_num"] = sum(r["atom_num"] for r in chain_res)
        all_chains.append(crec)

    if not all_chains:
        raise ValueError(f"No protein residues in {pdb_id}")

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        residues=np.array(all_res, dtype=RESIDUES_DTYPE),
        atoms=np.array(all_atoms, dtype=ATOMS_DTYPE),
        chains=np.array(all_chains, dtype=CHAINS_DTYPE),
        bonds=np.zeros(0), connections=np.zeros(0),
        interfaces=np.zeros(0),
        mask=np.ones(len(all_res), dtype=bool),
        coords=np.zeros((len(all_atoms), 3), dtype=np.float32),
        ensemble=np.zeros(0),
    )
    print(f"Saved npz: {out_npz}  ({len(all_res)} residues)")
    return out_npz


def make_batch(example, gamma_val: float, device: str, seed: int) -> ProteinBatch:
    torch.manual_seed(seed)
    ex = center_and_scale(example)
    L = ex.seq_len
    mask_f = ex.atom_mask.unsqueeze(-1).to(ex.coords.dtype)
    eps = torch.randn_like(ex.coords)
    x_gamma = (gamma_val * ex.coords + (1 - gamma_val) * eps) * mask_f
    return ProteinBatch(
        res_type=ex.res_type.unsqueeze(0),
        res_seq_nums=ex.res_seq_nums.unsqueeze(0),
        atom_type=ex.atom_type.unsqueeze(0),
        pair_type=ex.pair_type.unsqueeze(0),
        res_mask=torch.ones(1, L, dtype=torch.bool),
        atom_mask=ex.atom_mask.unsqueeze(0),
        valid_mask=(ex.atom_mask & ex.observed_mask).unsqueeze(0),
        ca_mask=(ex.atom_mask[:, CA_ATOM_ID] & ex.observed_mask[:, CA_ATOM_ID]).unsqueeze(0),
        x_clean=ex.coords.unsqueeze(0),
        x_gamma=x_gamma.unsqueeze(0),
        eps=(eps * mask_f).unsqueeze(0),
        gamma=torch.tensor([[[[gamma_val]]]]),
        esm=None,
    ).to(torch.device(device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--pdb_id", default="10AF", help="2025 PDB ID")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--no_ema", dest="use_ema", action="store_false")
    parser.add_argument("--n_sample_steps", type=int, default=50)
    parser.add_argument("--n_sample_seeds", type=int, default=3)
    parser.add_argument("--n_seeds", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download and convert
    npz_path = out_dir / f"{args.pdb_id.lower()}.npz"
    download_and_convert(args.pdb_id, npz_path)

    # Step 2: Load via RCSBDataset
    ds = RCSBDataset(str(out_dir), max_length=512, min_length=10, min_obs_ratio=0.1)
    example = ds[0]
    if example is None:
        raise RuntimeError("RCSBDataset returned None — check npz format")
    ex_c = center_and_scale(example)
    L = ex_c.seq_len
    ca_mask = (ex_c.atom_mask[:, CA_ATOM_ID] & ex_c.observed_mask[:, CA_ATOM_ID]).numpy().astype(bool)
    true_ca = ex_c.coords[:, CA_ATOM_ID, :].numpy() * COORD_SCALE
    print(f"\nProtein: {args.pdb_id}  L={L}  Cα observed: {ca_mask.sum()}")

    # Step 3: Load model
    model = load_from_checkpoint(args.ckpt, device, use_ema=args.use_ema)

    # Step 4: Single-step reconstruction at hi-gamma (γ=0.9)
    print("\n── Single-step reconstruction (γ=0.9) ──")
    gammas = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    rmsds = np.zeros((len(gammas), args.n_seeds), dtype=np.float32)
    lddts = np.zeros_like(rmsds)

    for gi, g in enumerate(gammas):
        for si in range(args.n_seeds):
            with torch.no_grad():
                batch = make_batch(example, g, device, seed=si)
                pred = model(batch)
                scale = eqm_reconstruction_scale(batch.gamma)
                x_hat = (batch.x_gamma - scale * pred)[0].cpu().numpy()
            xh = x_hat[:, CA_ATOM_ID, :] * COORD_SCALE
            xc = batch.x_clean[0].cpu().numpy()[:, CA_ATOM_ID, :] * COORD_SCALE
            rmsds[gi, si] = kabsch_rmsd(xh, xc, ca_mask)
            lddts[gi, si] = hard_lddt(xh, xc, ca_mask)
        print(f"  γ={g:.2f}  RMSD={rmsds[gi].mean():.2f}±{rmsds[gi].std():.2f} Å"
              f"  LDDT={lddts[gi].mean():.3f}±{lddts[gi].std():.3f}")

    # Step 5: Euler ODE sampling (γ: 0→0.99)
    print(f"\n── Euler ODE sampling ({args.n_sample_steps} steps) ──")
    euler_finals, euler_rmsds, euler_lddts = [], [], []
    euler_trajs = []
    for si in range(args.n_sample_seeds):
        final_ca, traj_ca, sched = sample_eqm_euler(
            model, example, n_steps=args.n_sample_steps, seed=si, device=device
        )
        rm = kabsch_rmsd(final_ca, true_ca, ca_mask)
        ld = hard_lddt(final_ca, true_ca, ca_mask)
        euler_finals.append(final_ca)
        euler_trajs.append(traj_ca)
        euler_rmsds.append(rm)
        euler_lddts.append(ld)
        print(f"  seed {si}  RMSD={rm:.2f} Å  LDDT={ld:.3f}")

    # Step 6: NAG sampling
    print(f"\n── NAG sampling ({args.n_sample_steps} steps) ──")
    nag_finals, nag_rmsds, nag_lddts = [], [], []
    nag_trajs = []
    for si in range(args.n_sample_seeds):
        final_ca, traj_ca, sched = sample_eqm_nag(
            model, example, n_steps=args.n_sample_steps, seed=si, device=device
        )
        rm = kabsch_rmsd(final_ca, true_ca, ca_mask)
        ld = hard_lddt(final_ca, true_ca, ca_mask)
        nag_finals.append(final_ca)
        nag_trajs.append(traj_ca)
        nag_rmsds.append(rm)
        nag_lddts.append(ld)
        print(f"  seed {si}  RMSD={rm:.2f} Å  LDDT={ld:.3f}")

    # Step 7: Save results
    npz_out = out_dir / "inference.npz"
    np.savez_compressed(
        npz_out,
        pdb_id=np.array([args.pdb_id]),
        seq_len=np.array([L]),
        true_ca=true_ca,
        ca_mask=ca_mask,
        gammas=np.array(gammas, dtype=np.float32),
        rmsds=rmsds,
        lddts=lddts,
        euler_final_ca=np.array(euler_finals, dtype=np.float32),
        euler_traj_ca=np.array(euler_trajs, dtype=np.float32),
        euler_rmsd=np.array(euler_rmsds, dtype=np.float32),
        euler_lddt=np.array(euler_lddts, dtype=np.float32),
        nag_final_ca=np.array(nag_finals, dtype=np.float32),
        nag_traj_ca=np.array(nag_trajs, dtype=np.float32),
        nag_rmsd=np.array(nag_rmsds, dtype=np.float32),
        nag_lddt=np.array(nag_lddts, dtype=np.float32),
    )
    print(f"\nSaved: {npz_out}")

    # Step 8: Summary
    print("\n══ SUMMARY ══════════════════════════════")
    print(f"  PDB: {args.pdb_id}  L={L}  Step={Path(args.ckpt).stem}")
    print(f"  Single-step  γ=0.90  RMSD={rmsds[-2].mean():.2f} Å  LDDT={lddts[-2].mean():.3f}")
    print(f"  Single-step  γ=0.99  RMSD={rmsds[-1].mean():.2f} Å  LDDT={lddts[-1].mean():.3f}")
    print(f"  Euler ODE            RMSD={np.mean(euler_rmsds):.2f} Å  LDDT={np.mean(euler_lddts):.3f}")
    print(f"  NAG                  RMSD={np.mean(nag_rmsds):.2f} Å  LDDT={np.mean(nag_lddts):.3f}")
    print("═════════════════════════════════════════")

    # Step 9: Visualization
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"MambaFold · {args.pdb_id} (2025) · {Path(args.ckpt).stem}", fontsize=13)

        # RMSD vs gamma
        ax = axes[0]
        ax.plot(gammas, rmsds.mean(1), "o-", color="steelblue", label="1-step recon")
        ax.fill_between(gammas, rmsds.mean(1) - rmsds.std(1),
                         rmsds.mean(1) + rmsds.std(1), alpha=0.2, color="steelblue")
        ax.axhline(np.mean(euler_rmsds), color="tomato", ls="--", label=f"Euler ODE ({np.mean(euler_rmsds):.2f} Å)")
        ax.axhline(np.mean(nag_rmsds), color="seagreen", ls="--", label=f"NAG ({np.mean(nag_rmsds):.2f} Å)")
        ax.set_xlabel("γ (noise level)"); ax.set_ylabel("Kabsch RMSD (Å)")
        ax.set_title("RMSD vs γ"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # LDDT vs gamma
        ax = axes[1]
        ax.plot(gammas, lddts.mean(1), "o-", color="steelblue", label="1-step recon")
        ax.fill_between(gammas, lddts.mean(1) - lddts.std(1),
                         lddts.mean(1) + lddts.std(1), alpha=0.2, color="steelblue")
        ax.axhline(np.mean(euler_lddts), color="tomato", ls="--", label=f"Euler ODE ({np.mean(euler_lddts):.3f})")
        ax.axhline(np.mean(nag_lddts), color="seagreen", ls="--", label=f"NAG ({np.mean(nag_lddts):.3f})")
        ax.set_xlabel("γ"); ax.set_ylabel("Hard LDDT")
        ax.set_title("LDDT vs γ"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # Cα overlay (XY plane) — best Euler seed vs ground truth
        ax = axes[2]
        best_seed = int(np.argmin(euler_rmsds))
        ax.plot(true_ca[ca_mask, 0], true_ca[ca_mask, 1],
                "k-o", ms=2, lw=1, label="Ground truth", alpha=0.8)
        ax.plot(euler_finals[best_seed][ca_mask, 0], euler_finals[best_seed][ca_mask, 1],
                "r-o", ms=2, lw=1, label=f"Euler (seed {best_seed})", alpha=0.7)
        ax.plot(nag_finals[int(np.argmin(nag_rmsds))][ca_mask, 0],
                nag_finals[int(np.argmin(nag_rmsds))][ca_mask, 1],
                "g-o", ms=2, lw=1, label=f"NAG", alpha=0.7)
        ax.set_xlabel("X (Å)"); ax.set_ylabel("Y (Å)")
        ax.set_title("Cα trace (XY)"); ax.legend(fontsize=8)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)

        plt.tight_layout()
        fig_path = out_dir / "viz.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {fig_path}")
    except Exception as e:
        print(f"Visualization skipped: {e}")


if __name__ == "__main__":
    main()
