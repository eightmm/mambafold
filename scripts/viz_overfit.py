#!/usr/bin/env python
"""Overfit visualization: train on 1 protein, plot loss curve + reconstruction quality per gamma.

Usage:
    PYTHONPATH=src WANDB_MODE=offline .venv/bin/python -u scripts/viz_overfit.py \
        --data_dir afdb_data/train --out logs/overfit_viz.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mambafold.data.constants import CA_ATOM_ID, MAX_ATOMS_PER_RES
from mambafold.data.dataset import AFDBDataset
from mambafold.data.transforms import center_and_scale
from mambafold.data.types import ProteinBatch
from mambafold.losses.eqm import eqm_reconstruction_scale
from mambafold.model.mambafold import MambaFoldEqM
from mambafold.train.ema import EMA
from mambafold.train.engine import train_step


# ── helpers ────────────────────────────────────────────────────────────────

GAMMA_GRID = [(i + 0.5) / 50 for i in range(50)]   # 0.01, 0.03, ..., 0.99
COORD_SCALE = 10.0                                   # 1 unit = 10 Å


def load_example(data_dir):
    ds = AFDBDataset(data_dir=data_dir, max_length=128)
    for i, f in enumerate(ds.files):
        ex = ds[i]
        if ex is not None:
            print(f"Loaded: {f.name}  (L={ex.seq_len})")
            return ex
    raise RuntimeError("No valid protein found.")


def make_batch(example, gamma_val: float, device: str) -> ProteinBatch:
    ex = center_and_scale(example)
    L, A = ex.seq_len, MAX_ATOMS_PER_RES
    mask_f = ex.atom_mask.unsqueeze(-1).to(ex.coords.dtype)
    eps = torch.randn_like(ex.coords)
    x_gamma = (gamma_val * ex.coords + (1 - gamma_val) * eps) * mask_f
    eps = eps * mask_f
    valid_mask = (ex.atom_mask & ex.observed_mask).unsqueeze(0)
    return ProteinBatch(
        res_type=ex.res_type.unsqueeze(0),
        atom_type=ex.atom_type.unsqueeze(0),
        res_mask=torch.ones(1, L, dtype=torch.bool),
        atom_mask=ex.atom_mask.unsqueeze(0),
        valid_mask=valid_mask,
        ca_mask=ex.atom_mask[:, CA_ATOM_ID].unsqueeze(0),
        x_clean=ex.coords.unsqueeze(0),
        x_gamma=x_gamma.unsqueeze(0),
        eps=eps.unsqueeze(0),
        gamma=torch.tensor([[[[gamma_val]]]]),
        esm=None,
    ).to(torch.device(device))


def build_model(device):
    return MambaFoldEqM(
        d_atom=256, d_res=256, d_plm=32,
        n_atom_enc=1, n_trunk=2, n_atom_dec=1,
        use_plm=False,
        atom_d_state=32, atom_mimo_rank=1, atom_headdim=64,
        d_state=32, mimo_rank=1, headdim=64,
    ).to(torch.device(device))


def hard_lddt_ca(pred_ca, true_ca, ca_mask, cutoff=1.5):
    """Compute hard LDDT for Cα atoms.
    pred_ca, true_ca: [L, 3]  (normalized coords)
    ca_mask: [L] bool
    cutoff: distance cutoff in normalized units (1.5 = 15Å)
    Returns scalar in [0, 1].
    """
    coords_p = pred_ca[ca_mask]   # [N, 3]
    coords_t = true_ca[ca_mask]   # [N, 3]
    N = coords_p.shape[0]
    if N < 2:
        return 0.0

    # Pairwise distances
    diff_p = coords_p.unsqueeze(0) - coords_p.unsqueeze(1)   # [N, N, 3]
    diff_t = coords_t.unsqueeze(0) - coords_t.unsqueeze(1)
    dist_p = diff_p.norm(dim=-1)   # [N, N]
    dist_t = diff_t.norm(dim=-1)

    # Only pairs within cutoff in true structure, excluding diagonal
    pair_mask = (dist_t < cutoff) & (torch.eye(N, device=dist_t.device).bool().logical_not())
    if pair_mask.sum() == 0:
        return 0.0

    dist_err = (dist_p - dist_t).abs()
    # Thresholds in normalized units (0.5, 1.0, 2.0, 4.0 Å → /10)
    thresholds = [t / COORD_SCALE for t in [0.5, 1.0, 2.0, 4.0]]
    score = sum((dist_err[pair_mask] < thr).float().mean().item() for thr in thresholds) / 4
    return score


def rmsd_ca(pred_ca, true_ca, ca_mask):
    """Cα RMSD in Å."""
    p = pred_ca[ca_mask]
    t = true_ca[ca_mask]
    return ((p - t).pow(2).sum(-1).mean().sqrt() * COORD_SCALE).item()


# ── main ───────────────────────────────────────────────────────────────────

def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    example = load_example(args.data_dir)
    model = build_model(device)
    ema = EMA(model, decay=0.999)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {n_params:.2f}M params")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # ── training ──
    n_steps = args.n_steps
    # per-gamma loss tracking: [50] running sums + counts
    gamma_loss_sum = np.zeros(50)
    gamma_loss_cnt = np.zeros(50)
    loss_curve = []   # (step, gamma_idx, loss)

    print(f"\nTraining {n_steps} steps with 50 fixed gammas...")
    for step in range(1, n_steps + 1):
        g_idx = (step - 1) % 50
        gamma_val = GAMMA_GRID[g_idx]
        batch = make_batch(example, gamma_val, device)
        metrics = train_step(model, batch, optimizer, grad_clip=1.0,
                             alpha_mode="const", use_amp=(device == "cuda"))
        ema.update(model)
        gamma_loss_sum[g_idx] += metrics["loss"]
        gamma_loss_cnt[g_idx] += 1
        loss_curve.append((step, g_idx, metrics["loss"]))
        if step % 500 == 0:
            avg = gamma_loss_sum.sum() / gamma_loss_cnt.sum()
            print(f"  step {step:>5d}/{n_steps} | avg_loss={avg:.4f}")

    print("Training done. Evaluating reconstruction...")

    # ── per-gamma evaluation ──
    ex_centered = center_and_scale(example)
    true_ca = ex_centered.coords[:, CA_ATOM_ID, :]        # [L, 3]
    ca_mask_1d = ex_centered.atom_mask[:, CA_ATOM_ID]     # [L]

    gamma_vals = np.array(GAMMA_GRID)
    lddts, rmsds, eqm_losses = [], [], []

    model.eval()
    with torch.no_grad():
        for g_val in GAMMA_GRID:
            # Average over 3 noise samples to reduce variance
            lddt_acc, rmsd_acc, loss_acc = 0, 0, 0
            n_eval = 3
            for _ in range(n_eval):
                batch = make_batch(example, g_val, device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=(device == "cuda")):
                    pred = model(batch)
                scale = eqm_reconstruction_scale(batch.gamma)
                x_hat = (batch.x_gamma - scale * pred)[0].float().cpu()  # [L, A, 3]
                pred_ca = x_hat[:, CA_ATOM_ID, :]                          # [L, 3]
                lddt_acc += hard_lddt_ca(pred_ca, true_ca, ca_mask_1d)
                rmsd_acc += rmsd_ca(pred_ca, true_ca, ca_mask_1d)
                loss_acc += gamma_loss_sum[GAMMA_GRID.index(g_val)] / max(1, gamma_loss_cnt[GAMMA_GRID.index(g_val)])
            lddts.append(lddt_acc / n_eval)
            rmsds.append(rmsd_acc / n_eval)
            eqm_losses.append(gamma_loss_sum[GAMMA_GRID.index(g_val)] /
                               max(1, gamma_loss_cnt[GAMMA_GRID.index(g_val)]))

    lddts = np.array(lddts)
    rmsds = np.array(rmsds)
    eqm_losses = np.array(eqm_losses)

    # ── structure overlay at gamma=0.9 ──
    model.eval()
    with torch.no_grad():
        batch_09 = make_batch(example, 0.91, device)   # closest to 0.9 in grid
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            pred_09 = model(batch_09)
        scale_09 = eqm_reconstruction_scale(batch_09.gamma)
        x_hat_09 = (batch_09.x_gamma - scale_09 * pred_09)[0].float().cpu()
        pred_ca_09 = x_hat_09[:, CA_ATOM_ID, :][ca_mask_1d].numpy()
        true_ca_np = true_ca[ca_mask_1d].numpy()
        noisy_ca_09 = batch_09.x_gamma[0, :, CA_ATOM_ID, :].float().cpu()[ca_mask_1d].numpy()

    # ── plotting ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("MambaFold EqM — Overfit Test (1 protein, 5000 steps, SISO)", fontsize=13)

    # 1. Loss curve (rolling average over 50-step window)
    ax = axes[0, 0]
    steps_arr = [x[0] for x in loss_curve]
    losses_arr = [x[2] for x in loss_curve]
    window = 50
    roll = np.convolve(losses_arr, np.ones(window) / window, mode="valid")
    ax.plot(steps_arr[window-1:], roll, lw=1.5, color="steelblue")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (50-step avg)")
    ax.set_title("Training Loss Curve")
    ax.grid(alpha=0.3)

    # 2. Per-gamma avg training loss
    ax = axes[0, 1]
    ax.bar(gamma_vals, eqm_losses, width=0.018, color="coral", alpha=0.8)
    ax.set_xlabel("γ")
    ax.set_ylabel("Avg EqM Loss")
    ax.set_title("Avg Loss per Gamma (training)")
    ax.grid(alpha=0.3, axis="y")

    # 3. LDDT vs gamma
    ax = axes[0, 2]
    ax.plot(gamma_vals, lddts, "o-", color="seagreen", ms=4, lw=1.5)
    ax.axhline(0.5, ls="--", color="gray", lw=1, label="LDDT=0.5")
    ax.axhline(0.7, ls=":", color="gray", lw=1, label="LDDT=0.7")
    ax.set_xlabel("γ")
    ax.set_ylabel("Hard LDDT (Cα)")
    ax.set_title("LDDT vs Gamma (eval)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 4. RMSD vs gamma
    ax = axes[1, 0]
    ax.plot(gamma_vals, rmsds, "s-", color="darkorange", ms=4, lw=1.5)
    ax.set_xlabel("γ")
    ax.set_ylabel("Cα RMSD (Å)")
    ax.set_title("RMSD vs Gamma (eval)")
    ax.grid(alpha=0.3)

    # 5. Cα overlay at gamma≈0.9 (XY plane)
    ax = axes[1, 1]
    ax.plot(true_ca_np[:, 0], true_ca_np[:, 1], "b-o", ms=3, lw=1, label="True", alpha=0.8)
    ax.plot(noisy_ca_09[:, 0], noisy_ca_09[:, 1], "g--", ms=2, lw=0.8, label=f"Noisy (γ=0.91)", alpha=0.5)
    ax.plot(pred_ca_09[:, 0], pred_ca_09[:, 1], "r-o", ms=3, lw=1, label="Reconstructed", alpha=0.8)
    ax.set_title("Cα overlay at γ≈0.9 (XY projection)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    # 6. LDDT + RMSD summary text
    ax = axes[1, 2]
    ax.axis("off")
    hi_gamma = gamma_vals > 0.7
    lo_gamma = gamma_vals < 0.3
    summary = (
        f"Model: SISO, d=256, L=128\n"
        f"Steps: {n_steps}\n\n"
        f"γ > 0.7 (near-clean):\n"
        f"  LDDT = {lddts[hi_gamma].mean():.3f} ± {lddts[hi_gamma].std():.3f}\n"
        f"  RMSD = {rmsds[hi_gamma].mean():.2f} ± {rmsds[hi_gamma].std():.2f} Å\n\n"
        f"γ < 0.3 (near-noisy):\n"
        f"  LDDT = {lddts[lo_gamma].mean():.3f} ± {lddts[lo_gamma].std():.3f}\n"
        f"  RMSD = {rmsds[lo_gamma].mean():.2f} ± {rmsds[lo_gamma].std():.2f} Å\n\n"
        f"Overall avg LDDT = {lddts.mean():.3f}\n"
        f"Overall avg RMSD = {rmsds.mean():.2f} Å"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            va="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"γ > 0.7  LDDT={lddts[hi_gamma].mean():.3f}  RMSD={rmsds[hi_gamma].mean():.2f}Å")
    print(f"γ < 0.3  LDDT={lddts[lo_gamma].mean():.3f}  RMSD={rmsds[lo_gamma].mean():.2f}Å")
    print(f"Overall  LDDT={lddts.mean():.3f}  RMSD={rmsds.mean():.2f}Å")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="afdb_data/train")
    parser.add_argument("--n_steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--out", default="logs/overfit_viz.png")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
