#!/usr/bin/env python
"""Unified overfit script: train + evaluate + visualize in one pass.

Usage:
    PYTHONPATH=src python -u scripts/overfit.py \
        --data_dir afdb_data/train \
        --out_dir outputs/overfit/test1

Output (all in --out_dir):
    config.json    - experiment config
    checkpoint.pt  - model + ema + optimizer state
    metrics.json   - per-gamma LDDT / RMSD
    viz.png        - 2x3 visualization
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import wandb
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mambafold.data.constants import CA_ATOM_ID, MAX_ATOMS_PER_RES, PAIR_PAD_ID
from mambafold.data.dataset import AFDBDataset
from mambafold.data.transforms import center_and_scale
from mambafold.data.types import ProteinBatch
from mambafold.losses.eqm import eqm_reconstruction_scale
from mambafold.model.mambafold import MambaFoldEqM
from mambafold.train.ema import EMA
from mambafold.train.engine import train_step

# ── constants ──────────────────────────────────────────────────────────────

GAMMA_GRID = [(i + 0.5) / 50 for i in range(50)]   # 0.01, 0.03, ..., 0.99
COORD_SCALE = 10.0


# ── data ───────────────────────────────────────────────────────────────────

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
    L = ex.seq_len
    mask_f = ex.atom_mask.unsqueeze(-1).to(ex.coords.dtype)
    eps = torch.randn_like(ex.coords)
    x_gamma = (gamma_val * ex.coords + (1 - gamma_val) * eps) * mask_f
    eps = eps * mask_f
    valid_mask = (ex.atom_mask & ex.observed_mask).unsqueeze(0)
    return ProteinBatch(
        res_type=ex.res_type.unsqueeze(0),
        atom_type=ex.atom_type.unsqueeze(0),
        pair_type=ex.pair_type.unsqueeze(0),
        res_mask=torch.ones(1, L, dtype=torch.bool),
        atom_mask=ex.atom_mask.unsqueeze(0),
        valid_mask=valid_mask,
        ca_mask=(ex.atom_mask[:, CA_ATOM_ID] & ex.observed_mask[:, CA_ATOM_ID]).unsqueeze(0),
        x_clean=ex.coords.unsqueeze(0),
        x_gamma=x_gamma.unsqueeze(0),
        eps=eps.unsqueeze(0),
        gamma=torch.tensor([[[[gamma_val]]]]),
        esm=None,
    ).to(torch.device(device))


# ── model ──────────────────────────────────────────────────────────────────

def build_model(args, device):
    return MambaFoldEqM(
        d_atom=args.d_atom,
        d_res=args.d_res,
        d_plm=32,
        n_atom_enc=args.n_atom_enc,
        n_trunk=args.n_trunk,
        n_atom_dec=args.n_atom_dec,
        use_plm=False,
        atom_d_state=args.d_state,
        atom_mimo_rank=args.mimo_rank,
        atom_headdim=args.headdim,
        d_state=args.d_state,
        mimo_rank=args.mimo_rank,
        headdim=args.headdim,
    ).to(torch.device(device))


# ── metrics ────────────────────────────────────────────────────────────────

def hard_lddt_ca(pred_ca, true_ca, ca_mask, cutoff=1.5):
    coords_p = pred_ca[ca_mask]
    coords_t = true_ca[ca_mask]
    N = coords_p.shape[0]
    if N < 2:
        return 0.0
    diff_p = coords_p.unsqueeze(0) - coords_p.unsqueeze(1)
    diff_t = coords_t.unsqueeze(0) - coords_t.unsqueeze(1)
    dist_p = diff_p.norm(dim=-1)
    dist_t = diff_t.norm(dim=-1)
    pair_mask = (dist_t < cutoff) & (~torch.eye(N, device=dist_t.device).bool())
    if pair_mask.sum() == 0:
        return 0.0
    dist_err = (dist_p - dist_t).abs()
    thresholds = [t / COORD_SCALE for t in [0.5, 1.0, 2.0, 4.0]]
    return sum((dist_err[pair_mask] < thr).float().mean().item() for thr in thresholds) / 4


def rmsd_ca(pred_ca, true_ca, ca_mask):
    p = pred_ca[ca_mask]
    t = true_ca[ca_mask]
    return ((p - t).pow(2).sum(-1).mean().sqrt() * COORD_SCALE).item()


# ── train ──────────────────────────────────────────────────────────────────

def run_training(args, example, model, ema, optimizer, device):
    gamma_loss_sum = np.zeros(50)
    gamma_loss_cnt = np.zeros(50)
    loss_curve = []

    print(f"\nTraining {args.n_steps} steps with 50 fixed gammas...")
    for step in range(1, args.n_steps + 1):
        g_idx = (step - 1) % 50
        gamma_val = GAMMA_GRID[g_idx]
        batch = make_batch(example, gamma_val, device)
        metrics = train_step(model, batch, optimizer, grad_clip=args.grad_clip,
                             alpha_mode="const", use_amp=(device == "cuda"))
        ema.update(model)
        gamma_loss_sum[g_idx] += metrics["loss"]
        gamma_loss_cnt[g_idx] += 1
        loss_curve.append((step, g_idx, metrics["loss"]))
        if step % 50 == 0 or step == 1:
            avg = gamma_loss_sum.sum() / gamma_loss_cnt.sum()
            print(f"  step {step:>5d}/{args.n_steps} | avg_loss={avg:.4f}", flush=True)
            if wandb.run is not None:
                wandb.log({"train/avg_loss": avg}, step=step)
        if wandb.run is not None:
            wandb.log({"train/loss": metrics["loss"], "train/gamma": gamma_val}, step=step)

    return loss_curve, gamma_loss_sum, gamma_loss_cnt


# ── evaluate ───────────────────────────────────────────────────────────────


# ── EqM Samplers ────────────────────────────────────────────────────────────
#
# Forward process: x_γ = γ·x_clean + (1-γ)·ε   (γ=0: noise, γ=1: clean)
# Model predicts:  f(x_γ) ≈ (ε - x_clean)·c(γ)
# 1-step clean:    x_hat = x_γ - scale(γ)·f      scale=(1-γ)/c(γ)
# ODE velocity:    dx/dγ = x_clean - ε ≈ (x_hat - x_γ)/(1-γ)
# Euler step:      x_{γ+Δγ} = x_γ + Δγ·(x_hat - x_γ)/(1-γ)


def _eqm_x_hat(model, x, ex, gamma_cur, device, a=0.8, lam=4.0):
    """Single forward pass: predict x_clean from current position x."""
    L = ex.seq_len
    batch = ProteinBatch(
        res_type=ex.res_type.unsqueeze(0).to(device),
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
        esm=None,
    )
    amp_on = device == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_on):
        pred = model(batch)
    scale = eqm_reconstruction_scale(batch.gamma, a=a, lam=lam)
    return (x.unsqueeze(0) - scale * pred)[0]


@torch.no_grad()
def sample_eqm_euler(model, example, n_steps: int = 50, seed: int = 0,
                     device: str = "cuda", a: float = 0.8, lam: float = 4.0):
    """EqM probability flow ODE — Euler integrator.

    Integrates dx/dγ = (x_hat - x) / (1-γ) from γ=0 to γ≈1.
    x_hat = x - scale(γ)·f(x)  is the one-step clean prediction.

    Returns:
        final_ca  [L, 3]           final Cα (Å)
        traj_ca   [n_steps, L, 3]  Cα at each Euler step (Å)
        sched     [n_steps+1]      gamma schedule used
    """
    torch.manual_seed(seed)
    ex = center_and_scale(example)
    L = ex.seq_len
    mask_f = ex.atom_mask.unsqueeze(-1).float().to(device)

    x = torch.randn(L, ex.atom_mask.shape[1], 3, device=device) * mask_f
    sched = torch.linspace(0.0, 0.99, n_steps + 1, device=device)
    traj_ca = []

    for i in range(n_steps):
        gamma_cur = float(sched[i].clamp(min=1e-4))
        dg = float(sched[i + 1] - sched[i])

        x_hat = _eqm_x_hat(model, x, ex, gamma_cur, device, a, lam)
        # ODE velocity: (x_hat - x) / (1 - γ)
        velocity = (x_hat - x) / max(1.0 - gamma_cur, 1e-4)
        x = (x + dg * velocity) * mask_f
        traj_ca.append(x[:, CA_ATOM_ID, :].float().cpu().numpy() * COORD_SCALE)

    # One final reconstruction step: x_hat at γ=sched[-1] removes residual noise floor
    x_hat_final = _eqm_x_hat(model, x, ex, float(sched[-1]), device, a, lam)
    final_ca = x_hat_final[:, CA_ATOM_ID, :].float().cpu().numpy() * COORD_SCALE
    return final_ca, np.array(traj_ca, dtype=np.float32), sched.cpu().numpy()


@torch.no_grad()
def sample_eqm_nag(model, example, n_steps: int = 50, seed: int = 0,
                   momentum: float = 0.35, device: str = "cuda",
                   a: float = 0.8, lam: float = 4.0):
    """EqM sampler — Nesterov Accelerated Gradient (NAG Form 1, paper Eq.9).

    Evaluates velocity at the Nesterov lookahead, steps from x_k:
        y_k      = x_k + μ·(x_k - x_{k-1})            lookahead
        x_{k+1}  = x_k + Δγ·(x_hat(y_k) - y_k)/(1-γ)  Euler from x_k

    momentum μ is γ-independent (constant throughout trajectory).
    μ=0.35 follows paper's empirical best (Table 2).

    Returns:
        final_ca  [L, 3]           final Cα (Å)
        traj_ca   [n_steps, L, 3]  Cα at each step (Å)
        sched     [n_steps+1]      gamma schedule used
    """
    torch.manual_seed(seed)
    ex = center_and_scale(example)
    L = ex.seq_len
    mask_f = ex.atom_mask.unsqueeze(-1).float().to(device)

    x = torch.randn(L, ex.atom_mask.shape[1], 3, device=device) * mask_f
    x_prev = x.clone()
    sched = torch.linspace(0.0, 0.99, n_steps + 1, device=device)
    traj_ca = []

    for i in range(n_steps):
        gamma_cur = float(sched[i].clamp(min=1e-4))
        dg = float(sched[i + 1] - sched[i])

        # Nesterov lookahead
        y = (x + momentum * (x - x_prev)) * mask_f
        x_hat_y = _eqm_x_hat(model, y, ex, gamma_cur, device, a, lam)

        # Euler step from x_k (NAG Form 1: paper Eq.9)
        velocity = (x_hat_y - y) / max(1.0 - gamma_cur, 1e-4)
        x_new = (x + dg * velocity) * mask_f
        x_prev, x = x, x_new
        traj_ca.append(x[:, CA_ATOM_ID, :].float().cpu().numpy() * COORD_SCALE)

    # One final reconstruction step: x_hat at γ=sched[-1] removes residual noise floor
    x_hat_final = _eqm_x_hat(model, x, ex, float(sched[-1]), device, a, lam)
    final_ca = x_hat_final[:, CA_ATOM_ID, :].float().cpu().numpy() * COORD_SCALE
    return final_ca, np.array(traj_ca, dtype=np.float32), sched.cpu().numpy()


def run_evaluation(example, model, device, out_dir=None):
    ex_centered = center_and_scale(example)
    true_ca = ex_centered.coords[:, CA_ATOM_ID, :]
    ca_mask_1d = ex_centered.atom_mask[:, CA_ATOM_ID] & ex_centered.observed_mask[:, CA_ATOM_ID]
    L = ex_centered.seq_len
    G = len(GAMMA_GRID)
    n_eval = 3

    lddts, rmsds = [], []

    # Arrays for npz export: [G, S, L, 3] in Angstrom
    x_clean_ca_arr = np.zeros((G, n_eval, L, 3), dtype=np.float32)
    x_noisy_ca_arr = np.zeros((G, n_eval, L, 3), dtype=np.float32)
    x_hat_ca_arr   = np.zeros((G, n_eval, L, 3), dtype=np.float32)
    rmsds_arr      = np.zeros((G, n_eval), dtype=np.float32)
    lddts_arr      = np.zeros((G, n_eval), dtype=np.float32)

    print("\nEvaluating reconstruction per gamma...")
    model.eval()
    with torch.no_grad():
        for g_idx, g_val in enumerate(GAMMA_GRID):
            lddt_acc, rmsd_acc = 0.0, 0.0
            for si in range(n_eval):
                batch = make_batch(example, g_val, device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=(device == "cuda")):
                    pred = model(batch)
                scale = eqm_reconstruction_scale(batch.gamma)
                x_hat = (batch.x_gamma - scale * pred)[0].float().cpu()
                pred_ca = x_hat[:, CA_ATOM_ID, :]
                lddt_val = hard_lddt_ca(pred_ca, true_ca, ca_mask_1d)
                rmsd_val = rmsd_ca(pred_ca, true_ca, ca_mask_1d)
                lddt_acc += lddt_val
                rmsd_acc += rmsd_val

                # collect Angstrom-scale Cα for npz
                x_clean_ca_arr[g_idx, si] = true_ca.numpy() * COORD_SCALE
                x_noisy_ca_arr[g_idx, si] = (
                    batch.x_gamma[0, :, CA_ATOM_ID, :].float().cpu().numpy() * COORD_SCALE
                )
                x_hat_ca_arr[g_idx, si] = pred_ca.numpy() * COORD_SCALE
                rmsds_arr[g_idx, si] = rmsd_val
                lddts_arr[g_idx, si] = lddt_val

            lddts.append(lddt_acc / n_eval)
            rmsds.append(rmsd_acc / n_eval)

    # Structure overlay at gamma≈0.9
    with torch.no_grad():
        batch_09 = make_batch(example, 0.91, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            pred_09 = model(batch_09)
        scale_09 = eqm_reconstruction_scale(batch_09.gamma)
        x_hat_09 = (batch_09.x_gamma - scale_09 * pred_09)[0].float().cpu()
        pred_ca_09 = x_hat_09[:, CA_ATOM_ID, :][ca_mask_1d].numpy()
        true_ca_np = true_ca[ca_mask_1d].numpy()
        noisy_ca_09 = batch_09.x_gamma[0, :, CA_ATOM_ID, :].float().cpu()[ca_mask_1d].numpy()

    # ── EqM iterative sampling: Euler ODE + NAG ──
    N_SEEDS = 3
    N_STEPS = 50
    true_ca_ang = true_ca.numpy() * COORD_SCALE
    ca_mask_np  = ca_mask_1d.numpy()

    def _run_sampler(sampler_fn, label):
        final_arr = np.zeros((N_SEEDS, L, 3), dtype=np.float32)
        traj_arr  = None
        rmsd_arr  = np.zeros(N_SEEDS, dtype=np.float32)
        print(f"\n{label} sampling from noise...")
        for si in range(N_SEEDS):
            final_ca, traj_ca, sched = sampler_fn(
                model, example, n_steps=N_STEPS, seed=si, device=device
            )
            final_arr[si] = final_ca
            if traj_arr is None:
                traj_arr = np.zeros((N_SEEDS, len(traj_ca), L, 3), dtype=np.float32)
            traj_arr[si] = traj_ca
            rmsd = float(np.sqrt(((final_ca[ca_mask_np] - true_ca_ang[ca_mask_np]) ** 2).sum(-1).mean()))
            rmsd_arr[si] = rmsd
            print(f"  seed {si}  RMSD={rmsd:.2f} Å")
        return final_arr, traj_arr, rmsd_arr, sched

    euler_final_arr, euler_traj_arr, euler_rmsd_arr, sched = _run_sampler(
        sample_eqm_euler, "EqM Euler"
    )
    nag_final_arr, nag_traj_arr, nag_rmsd_arr, _ = _run_sampler(
        sample_eqm_nag, "EqM NAG"
    )

    # Save npz so notebook can load without re-running the model
    if out_dir is not None:
        npz_path = Path(out_dir) / "inference.npz"
        np.savez_compressed(
            npz_path,
            x_clean_ca=x_clean_ca_arr,
            x_noisy_ca=x_noisy_ca_arr,
            x_hat_ca=x_hat_ca_arr,
            ca_mask=ca_mask_1d.numpy(),
            gammas=np.array(GAMMA_GRID, dtype=np.float32),
            rmsds=rmsds_arr,
            lddts=lddts_arr,
            euler_traj_ca=euler_traj_arr,
            euler_final_ca=euler_final_arr,
            euler_gammas=sched.astype(np.float32),
            euler_rmsd=euler_rmsd_arr,
            nag_traj_ca=nag_traj_arr,
            nag_final_ca=nag_final_arr,
            nag_gammas=sched.astype(np.float32),
            nag_rmsd=nag_rmsd_arr,
        )
        print(f"Saved: {npz_path}  ({npz_path.stat().st_size / 1024:.0f} KB)")

    return (np.array(lddts), np.array(rmsds),
            true_ca_np, pred_ca_09, noisy_ca_09,
            euler_rmsd_arr, nag_rmsd_arr)


# ── plot ───────────────────────────────────────────────────────────────────

def plot_results(loss_curve, gamma_loss_sum, gamma_loss_cnt,
                 lddts, rmsds, true_ca_np, pred_ca_09, noisy_ca_09,
                 args, out_dir):
    gamma_vals = np.array(GAMMA_GRID)
    eqm_losses = gamma_loss_sum / np.maximum(gamma_loss_cnt, 1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"MambaFold EqM — Overfit (1 protein, {args.n_steps} steps, "
        f"MIMO rank={args.mimo_rank}, d_res={args.d_res})",
        fontsize=13,
    )

    # 1. Loss curve
    ax = axes[0, 0]
    steps_arr = [x[0] for x in loss_curve]
    losses_arr = [x[2] for x in loss_curve]
    window = 50
    roll = np.convolve(losses_arr, np.ones(window) / window, mode="valid")
    ax.plot(steps_arr[window - 1:], roll, lw=1.5, color="steelblue")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss (50-step avg)")
    ax.set_title("Training Loss Curve"); ax.grid(alpha=0.3)

    # 2. Per-gamma avg training loss
    ax = axes[0, 1]
    ax.bar(gamma_vals, eqm_losses, width=0.018, color="coral", alpha=0.8)
    ax.set_xlabel("γ"); ax.set_ylabel("Avg EqM Loss")
    ax.set_title("Avg Loss per Gamma (training)"); ax.grid(alpha=0.3, axis="y")

    # 3. LDDT vs gamma
    ax = axes[0, 2]
    ax.plot(gamma_vals, lddts, "o-", color="seagreen", ms=4, lw=1.5)
    ax.axhline(0.5, ls="--", color="gray", lw=1, label="LDDT=0.5")
    ax.axhline(0.7, ls=":", color="gray", lw=1, label="LDDT=0.7")
    ax.set_xlabel("γ"); ax.set_ylabel("Hard LDDT (Cα)")
    ax.set_title("LDDT vs Gamma (eval)"); ax.set_ylim(0, 1)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 4. RMSD vs gamma
    ax = axes[1, 0]
    ax.plot(gamma_vals, rmsds, "s-", color="darkorange", ms=4, lw=1.5)
    ax.set_xlabel("γ"); ax.set_ylabel("Cα RMSD (Å)")
    ax.set_title("RMSD vs Gamma (eval)"); ax.grid(alpha=0.3)

    # 5. Cα overlay at gamma≈0.9
    ax = axes[1, 1]
    ax.plot(true_ca_np[:, 0], true_ca_np[:, 1], "b-o", ms=3, lw=1,
            label="True", alpha=0.8)
    ax.plot(noisy_ca_09[:, 0], noisy_ca_09[:, 1], "g--", ms=2, lw=0.8,
            label="Noisy (γ=0.91)", alpha=0.5)
    ax.plot(pred_ca_09[:, 0], pred_ca_09[:, 1], "r-o", ms=3, lw=1,
            label="Reconstructed", alpha=0.8)
    ax.set_title("Cα overlay at γ≈0.9 (XY projection)")
    ax.legend(fontsize=8); ax.set_aspect("equal"); ax.grid(alpha=0.2)

    # 6. Summary text
    ax = axes[1, 2]
    ax.axis("off")
    hi = gamma_vals > 0.7
    lo = gamma_vals < 0.3
    summary = (
        f"d_atom={args.d_atom}, d_res={args.d_res}\n"
        f"n_trunk={args.n_trunk}, mimo_rank={args.mimo_rank}\n"
        f"Steps: {args.n_steps}\n\n"
        f"γ > 0.7 (near-clean):\n"
        f"  LDDT = {lddts[hi].mean():.3f} ± {lddts[hi].std():.3f}\n"
        f"  RMSD = {rmsds[hi].mean():.2f} ± {rmsds[hi].std():.2f} Å\n\n"
        f"γ < 0.3 (near-noisy):\n"
        f"  LDDT = {lddts[lo].mean():.3f} ± {lddts[lo].std():.3f}\n"
        f"  RMSD = {rmsds[lo].mean():.2f} ± {rmsds[lo].std():.2f} Å\n\n"
        f"Overall avg LDDT = {lddts.mean():.3f}\n"
        f"Overall avg RMSD = {rmsds.mean():.2f} Å"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            va="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    viz_path = out_dir / "viz.png"
    plt.savefig(viz_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {viz_path}")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    import os, time

    # ── 1. config file (pre-parse) ──────────────────────────────────────────
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()

    cfg = {}
    if pre_args.config:
        with open(pre_args.config) as f:
            cfg = yaml.safe_load(f) or {}
        print(f"Config: {pre_args.config}")

    # ── 2. full arg parser (yaml values are defaults) ───────────────────────
    parser = argparse.ArgumentParser(description="MambaFold unified overfit+viz")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    # Data
    parser.add_argument("--data_dir", default="afdb_data/train")
    # Output
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: outputs/overfit/<job_id|timestamp>)")
    # Training
    parser.add_argument("--n_steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    # Model
    parser.add_argument("--d_atom", type=int, default=256)
    parser.add_argument("--d_res", type=int, default=256)
    parser.add_argument("--d_state", type=int, default=32)
    parser.add_argument("--mimo_rank", type=int, default=2)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--n_atom_enc", type=int, default=2)
    parser.add_argument("--n_trunk", type=int, default=6)
    parser.add_argument("--n_atom_dec", type=int, default=2)
    # W&B
    parser.add_argument("--wandb_project", default="mambafold")
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=[])
    parser.add_argument("--wandb_offline", action="store_true", default=False)
    parser.add_argument("--no_wandb", action="store_true", default=False)

    # Apply yaml as defaults (CLI flags override)
    parser.set_defaults(**cfg)
    args = parser.parse_args()

    # ── 3. output dir ───────────────────────────────────────────────────────
    if args.out_dir is None:
        job_id = os.environ.get("SLURM_JOB_ID", None)
        tag = job_id if job_id else time.strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"outputs/overfit/{tag}"
    else:
        tag = Path(args.out_dir).name
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # Save config
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    example = load_example(args.data_dir)
    model = build_model(args, device)
    ema = EMA(model, decay=0.999)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {n_params:.2f}M params")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # ── 4. wandb init ───────────────────────────────────────────────────────
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or tag,
            tags=args.wandb_tags or [],
            config={k: v for k, v in vars(args).items()
                    if not k.startswith("wandb") and k != "no_wandb"},
            mode="offline" if args.wandb_offline else "online",
        )
        wandb.config.update({"n_params_M": round(n_params, 2)})

    # ── 5. train ────────────────────────────────────────────────────────────
    loss_curve, gamma_loss_sum, gamma_loss_cnt = run_training(
        args, example, model, ema, optimizer, device
    )

    # ── 6. save checkpoint ──────────────────────────────────────────────────
    ckpt_path = out_dir / "checkpoint.pt"
    torch.save({
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "loss_curve": loss_curve,
        "gamma_loss_sum": gamma_loss_sum,
        "gamma_loss_cnt": gamma_loss_cnt,
    }, ckpt_path)
    print(f"Saved: {ckpt_path}")

    # ── 7. evaluate ─────────────────────────────────────────────────────────
    lddts, rmsds, true_ca_np, pred_ca_09, noisy_ca_09, euler_rmsd_arr, nag_rmsd_arr = (
        run_evaluation(example, model, device, out_dir=out_dir)
    )

    # ── 8. save metrics ─────────────────────────────────────────────────────
    hi_mask = np.array(GAMMA_GRID) > 0.7
    metrics = {
        "gamma": GAMMA_GRID,
        "lddt": lddts.tolist(),
        "rmsd": rmsds.tolist(),
        "lddt_hi_gamma_mean": float(lddts[hi_mask].mean()),
        "rmsd_hi_gamma_mean": float(rmsds[hi_mask].mean()),
        "lddt_overall_mean": float(lddts.mean()),
        "rmsd_overall_mean": float(rmsds.mean()),
        "euler_rmsd_mean": float(euler_rmsd_arr.mean()),
        "nag_rmsd_mean": float(nag_rmsd_arr.mean()),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved: {out_dir / 'metrics.json'}")

    # ── 9. wandb eval logging ───────────────────────────────────────────────
    if wandb.run is not None:
        gamma_vals = np.array(GAMMA_GRID)
        # per-gamma table
        table = wandb.Table(columns=["gamma", "rmsd", "lddt"])
        for gv, rv, lv in zip(gamma_vals, rmsds, lddts):
            table.add_data(float(gv), float(rv), float(lv))
        wandb.log({
            "eval/rmsd_vs_gamma": wandb.plot.line(table, "gamma", "rmsd",
                                                   title="RMSD vs γ"),
            "eval/lddt_vs_gamma": wandb.plot.line(table, "gamma", "lddt",
                                                   title="LDDT vs γ"),
            "eval/rmsd_hi_gamma": float(rmsds[hi_mask].mean()),
            "eval/lddt_hi_gamma": float(lddts[hi_mask].mean()),
            "eval/rmsd_overall": float(rmsds.mean()),
            "eval/lddt_overall": float(lddts.mean()),
            "eval/euler_rmsd_mean": float(euler_rmsd_arr.mean()),
            "eval/nag_rmsd_mean": float(nag_rmsd_arr.mean()),
        })

    # ── 10. plot ─────────────────────────────────────────────────────────────
    plot_results(loss_curve, gamma_loss_sum, gamma_loss_cnt,
                 lddts, rmsds, true_ca_np, pred_ca_09, noisy_ca_09,
                 args, out_dir)
    if wandb.run is not None:
        wandb.log({"eval/viz": wandb.Image(str(out_dir / "viz.png"))})

    # ── 11. PASS/FAIL ────────────────────────────────────────────────────────
    losses_list = [x[2] for x in loss_curve]
    loss_init = losses_list[0]
    loss_min = min(losses_list)
    drop = (loss_init - loss_min) / loss_init * 100

    print(f"\n{'='*50}")
    print(f"γ > 0.7  LDDT={lddts[hi_mask].mean():.3f}  "
          f"RMSD={rmsds[hi_mask].mean():.2f}Å")
    print(f"Overall  LDDT={lddts.mean():.3f}  RMSD={rmsds.mean():.2f}Å")
    print(f"Euler RMSD={euler_rmsd_arr.mean():.2f}Å  NAG RMSD={nag_rmsd_arr.mean():.2f}Å")
    print(f"Loss drop: {drop:.1f}% (init={loss_init:.4f}, min={loss_min:.4f})")

    if wandb.run is not None:
        wandb.summary.update({
            "loss_drop_pct": drop,
            "pass": drop >= 50.0,
        })
        wandb.finish()

    if drop >= 50.0:
        print(f"\nPASS: loss dropped {drop:.1f}% >= 50%")
        sys.exit(0)
    else:
        print(f"\nFAIL: loss dropped only {drop:.1f}% (expected >= 50%)")
        sys.exit(1)


if __name__ == "__main__":
    main()
