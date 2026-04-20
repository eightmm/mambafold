"""MambaFold inference from amino acid sequence only.

No reference structure needed — starts from pure noise and runs EqM ODE sampling.

Usage (GPU node):
    PYTHONPATH=src python scripts/infer_seq.py \
        --ckpt outputs/train/23508/ckpt_0200000.pt \
        --seq MAHHHHHHMSRPHVFF... \
        --out outputs/infer_2025/10af \
        [--pdb_id 10AF]   # optional label
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # for export_inference imports

from mambafold.data.constants import (
    AA_TO_ID, AA_3TO1, ATOM_NAME_TO_ID, CA_ATOM_ID, COORD_SCALE,
    MAX_ATOMS_PER_RES, PAIR_PAD_ID, PAIR_TO_ID,
    RESIDUE_ATOMS, RESIDUE_ATOM_TO_SLOT,
)
from mambafold.data.esm import ESMEmbedder
from mambafold.data.transforms import center_and_scale
from mambafold.data.types import ProteinBatch, ProteinExample
from mambafold.train.trainer import load_from_checkpoint
from infer_train import sample_euler, sample_nag, save_pdb_allatom   # ESM-aware


AA_1TO3 = {v: k for k, v in AA_3TO1.items()}


def seq_to_example(seq: str, esm: torch.Tensor | None = None) -> ProteinExample:
    """Build a ProteinExample from an amino acid sequence (no coordinates).

    Args:
        seq: 1-letter amino acid sequence.
        esm: Optional [L, d_esm] ESM embedding. If provided, attached to the
            example so the model receives the real PLM features.
    """
    # Convert 1-letter to 3-letter, skip unknowns
    residues = []
    for aa in seq.upper():
        name3 = AA_1TO3.get(aa)
        if name3 is None:
            continue
        residues.append(name3)

    L = len(residues)
    if L == 0:
        raise ValueError("No standard amino acids found in sequence")

    A = MAX_ATOMS_PER_RES
    res_type     = torch.zeros(L, dtype=torch.long)
    atom_type    = torch.full((L, A), ATOM_NAME_TO_ID["PAD"], dtype=torch.long)
    pair_type    = torch.full((L, A), PAIR_PAD_ID, dtype=torch.long)
    coords       = torch.zeros(L, A, 3, dtype=torch.float32)   # dummy — not used by ODE
    atom_mask    = torch.zeros(L, A, dtype=torch.bool)
    observed_mask = torch.zeros(L, A, dtype=torch.bool)
    res_seq_nums = torch.arange(L, dtype=torch.long)

    for i, res_name in enumerate(residues):
        res_type[i] = AA_TO_ID.get(res_name, AA_TO_ID["UNK"])
        canon_atoms = RESIDUE_ATOMS.get(res_name, [])
        for j, atom_name in enumerate(canon_atoms):
            if j >= A:
                break
            atom_type[i, j]    = ATOM_NAME_TO_ID.get(atom_name, ATOM_NAME_TO_ID["PAD"])
            pair_type[i, j]    = PAIR_TO_ID.get((res_name, atom_name), PAIR_PAD_ID)
            atom_mask[i, j]    = True
            observed_mask[i, j] = True   # treat all atoms as "present" for ODE

    return ProteinExample(
        res_type=res_type,
        atom_type=atom_type,
        pair_type=pair_type,
        coords=coords,
        atom_mask=atom_mask,
        observed_mask=observed_mask,
        res_seq_nums=res_seq_nums,
        seq_len=L,
        esm=esm,
    )


def save_pdb(ca_coords: np.ndarray, sequence: str, out_path: Path):
    """Save Cα-only structure as PDB (ATOM records)."""
    lines = []
    atom_num = 1
    for i, (xyz, aa1) in enumerate(zip(ca_coords, sequence)):
        aa3 = AA_1TO3.get(aa1.upper(), "UNK")
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        lines.append(
            f"ATOM  {atom_num:5d}  CA  {aa3:3s} A{i+1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
        )
        atom_num += 1
    lines.append("END\n")
    out_path.write_text("".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--seq", required=True, help="Amino acid sequence (1-letter)")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--pdb_id", default="query", help="Label for outputs")
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--no_ema", dest="use_ema", action="store_false")
    parser.add_argument("--n_steps", type=int, default=50, help="ODE steps")
    parser.add_argument("--n_seeds", type=int, default=3, help="Number of ODE runs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build example from sequence
    # Strip whitespace/numbers (handles FASTA-style or raw sequence)
    seq = "".join(c for c in args.seq if c.isalpha())

    # Compute ESM embedding (model was trained with use_plm=True, so this
    # matters a lot — passing zeros would cripple the prediction)
    print(f"\nComputing ESM3 embedding for L={len(seq)}...", flush=True)
    embedder = ESMEmbedder(model_name="esm3-open", device=device)
    esm = embedder([seq])[0, :len(seq)].float().cpu()   # [L, d_esm]
    print(f"  ESM shape: {tuple(esm.shape)}")

    example = seq_to_example(seq, esm=esm)
    L = example.seq_len
    ca_mask = example.atom_mask[:, CA_ATOM_ID].numpy().astype(bool)
    print(f"Sequence: {args.pdb_id}  L={L}")

    # Load model
    model = load_from_checkpoint(args.ckpt, device, use_ema=args.use_ema)

    # ODE sampling (Euler + NAG)
    results = {}
    atom_mask_np = example.atom_mask.numpy().astype(bool)   # [L, A]
    res_type_np = example.res_type.numpy()                  # [L]
    for method, fn in [("euler", sample_euler), ("nag", sample_nag)]:
        print(f"\n── {method.upper()} ODE ({args.n_steps} steps × {args.n_seeds} seeds) ──")
        finals_ca, finals_aa, trajs = [], [], []
        for si in range(args.n_seeds):
            final_ca, final_aa, traj_ca, sched = fn(
                model, example, n_steps=args.n_steps, seed=si, device=device
            )
            finals_ca.append(final_ca)
            finals_aa.append(final_aa)
            trajs.append(traj_ca)
            print(f"  seed {si} done  (L={final_ca.shape[0]})")
        results[method] = {
            "finals_ca": np.array(finals_ca),
            "finals_aa": np.array(finals_aa),
            "trajs":     np.array(trajs),
        }

    # Save npz
    npz_out = out_dir / "inference.npz"
    np.savez_compressed(
        npz_out,
        pdb_id=np.array([args.pdb_id]),
        sequence=np.array([seq]),
        seq_len=np.array([L]),
        ca_mask=ca_mask,
        atom_mask=atom_mask_np,
        euler_final_ca=results["euler"]["finals_ca"],
        euler_final_aa=results["euler"]["finals_aa"],
        euler_traj_ca=results["euler"]["trajs"],
        nag_final_ca=results["nag"]["finals_ca"],
        nag_final_aa=results["nag"]["finals_aa"],
        nag_traj_ca=results["nag"]["trajs"],
    )
    print(f"\nSaved: {npz_out}")

    # Save all-atom PDB files (B-factor = 0, no ground truth available)
    zero_b = np.zeros_like(atom_mask_np, dtype=np.float32)
    for method in ("euler", "nag"):
        for si, aa in enumerate(results[method]["finals_aa"]):
            pdb_path = out_dir / f"{args.pdb_id.lower()}_{method}_seed{si}.pdb"
            save_pdb_allatom(aa, res_type_np, atom_mask_np, zero_b, pdb_path)
        print(f"Saved {args.n_seeds} all-atom PDBs: {args.pdb_id.lower()}_{method}_seed*.pdb")

    # Visualization
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, args.n_seeds, figsize=(5 * args.n_seeds, 5))
        if args.n_seeds == 1:
            axes = [axes]
        fig.suptitle(f"MambaFold · {args.pdb_id} · {Path(args.ckpt).stem}\nCα trace (XY)", fontsize=11)

        colors = {"euler": "tomato", "nag": "steelblue"}
        for si in range(args.n_seeds):
            ax = axes[si]
            for method in ("euler", "nag"):
                ca = results[method]["finals_ca"][si]
                ax.plot(ca[ca_mask, 0], ca[ca_mask, 1], "-o", ms=2, lw=1,
                        color=colors[method], label=method, alpha=0.8)
            ax.set_title(f"seed {si}")
            ax.set_aspect("equal")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_xlabel("X (Å)")
            ax.set_ylabel("Y (Å)")

        plt.tight_layout()
        fig_path = out_dir / "viz.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {fig_path}")
    except Exception as e:
        print(f"Visualization skipped: {e}")

    print("\n══ Done ══")


if __name__ == "__main__":
    main()
