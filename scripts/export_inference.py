"""Export per-gamma inference results to .npz for notebook visualization.

Usage (GPU node):
    PYTHONPATH=src python scripts/export_inference.py \
        --ckpt outputs/overfit/22627/checkpoint.pt \
        --data_dir afdb_data/train \
        --out outputs/overfit/22627/inference.npz

Outputs a single .npz with arrays per gamma:
    x_clean_ca      [G, S, L, 3]  ground truth Cα (Å)         — single-step
    x_noisy_ca      [G, S, L, 3]  noisy input Cα (Å)          — single-step
    x_hat_ca        [G, S, L, 3]  model reconstruction Cα (Å) — single-step
    ca_mask         [L]            bool mask
    gammas          [G]            gamma values
    rmsds           [G, S]         per-gamma per-seed RMSD (Å)
    lddts           [G, S]         per-gamma per-seed hard LDDT
    euler_traj_ca  [DS, T, L, 3]  EqM Euler trajectory Cα (Å)
    euler_final_ca [DS, L, 3]     EqM Euler final structure Cα (Å)
    euler_gammas   [T]             EqM Euler gamma schedule
    euler_rmsd     [DS]            EqM Euler final RMSD (Å)
    nag_traj_ca    [DS, T, L, 3]  EqM NAG trajectory Cα (Å)
    nag_final_ca   [DS, L, 3]     EqM NAG final structure Cα (Å)
    nag_gammas     [T]             EqM NAG gamma schedule
    nag_rmsd       [DS]            EqM NAG final RMSD (Å)

G = n_gammas, S = n_seeds, DS = n_sample_seeds, T = n_sample_steps, L = protein length
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mambafold.data.constants import CA_ATOM_ID, COORD_SCALE, PAIR_PAD_ID
from mambafold.data.dataset import AFDBDataset
from mambafold.data.transforms import center_and_scale
from mambafold.data.types import ProteinBatch
from mambafold.losses.eqm import eqm_reconstruction_scale
from mambafold.model.mambafold import MambaFoldEqM


# ── helpers ────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = MambaFoldEqM(
        d_atom=a["d_atom"],
        d_res=a["d_res"],
        d_plm=32,
        n_atom_enc=a["n_atom_enc"],
        n_trunk=a["n_trunk"],
        n_atom_dec=a["n_atom_dec"],
        use_plm=False,
        atom_d_state=a["d_state"],
        atom_mimo_rank=a["mimo_rank"],
        atom_headdim=a["headdim"],
        d_state=a["d_state"],
        mimo_rank=a["mimo_rank"],
        headdim=a["headdim"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def load_example(data_dir: str, idx: int = 0):
    dataset = AFDBDataset(data_dir=data_dir, max_length=256)
    count = 0
    for i in range(len(dataset)):
        ex = dataset[i]
        if ex is None:
            continue
        if count == idx:
            print(f"Loaded [{i}]: {dataset.files[i].name}  L={ex.seq_len}")
            return ex
        count += 1
    raise RuntimeError(f"Protein index {idx} not found in {data_dir}")


def make_batch(example, gamma_val: float, device: str, seed: int) -> ProteinBatch:
    torch.manual_seed(seed)
    ex = center_and_scale(example)
    L = ex.seq_len
    mask_f = ex.atom_mask.unsqueeze(-1).to(ex.coords.dtype)
    eps = torch.randn_like(ex.coords)
    x_gamma = (gamma_val * ex.coords + (1 - gamma_val) * eps) * mask_f
    eps = eps * mask_f
    return ProteinBatch(
        res_type=ex.res_type.unsqueeze(0),
        atom_type=ex.atom_type.unsqueeze(0),
        pair_type=ex.pair_type.unsqueeze(0),
        res_mask=torch.ones(1, L, dtype=torch.bool),
        atom_mask=ex.atom_mask.unsqueeze(0),
        valid_mask=(ex.atom_mask & ex.observed_mask).unsqueeze(0),
        ca_mask=ex.atom_mask[:, CA_ATOM_ID].unsqueeze(0),
        x_clean=ex.coords.unsqueeze(0),
        x_gamma=x_gamma.unsqueeze(0),
        eps=eps.unsqueeze(0),
        gamma=torch.tensor([[[[gamma_val]]]]),
        esm=None,
    ).to(torch.device(device))


def hard_lddt(pred_ca, true_ca, mask, cutoff=15.0):
    """Hard LDDT for Cα in Angstrom."""
    p = pred_ca[mask]  # [N, 3]
    t = true_ca[mask]
    if len(p) < 2:
        return 0.0
    diff = np.abs(
        np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1) + 1e-8) -
        np.sqrt(((t[:, None] - t[None]) ** 2).sum(-1) + 1e-8)
    )
    thresholds = [0.5, 1.0, 2.0, 4.0]
    score = np.mean([np.mean(diff < thr) for thr in thresholds])
    return float(score)



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
        ca_mask=ex.atom_mask[:, CA_ATOM_ID].unsqueeze(0).to(device),
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

    x_hat_final = _eqm_x_hat(model, x, ex, float(sched[-1]), device, a, lam)
    final_ca = x_hat_final[:, CA_ATOM_ID, :].float().cpu().numpy() * COORD_SCALE
    return final_ca, np.array(traj_ca, dtype=np.float32), sched.cpu().numpy()


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--protein_idx", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=5,
                        help="Number of noise seeds per gamma")
    parser.add_argument("--gammas", type=str, default=None,
                        help="Comma-separated gamma values (default: 30 evenly spaced 0.05→0.99)")
    parser.add_argument("--n_sample_steps", type=int, default=30,
                        help="Number of EqM Euler/NAG sampling steps (default: 50)")
    parser.add_argument("--n_sample_seeds", type=int, default=3,
                        help="Number of EqM sampling seeds (default: 3)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = load_model(args.ckpt, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {n_params:.2f}M params")

    example = load_example(args.data_dir, args.protein_idx)
    ex_c = center_and_scale(example)
    L = ex_c.seq_len
    ca_mask = ex_c.atom_mask[:, CA_ATOM_ID].numpy().astype(bool)  # [L]

    if args.gammas:
        gammas = [float(g) for g in args.gammas.split(",")]
    else:
        gammas = list(np.linspace(0.05, 0.99, 30))

    G, S = len(gammas), args.n_seeds

    # output arrays
    x_clean_ca = np.zeros((G, S, L, 3), dtype=np.float32)
    x_noisy_ca = np.zeros((G, S, L, 3), dtype=np.float32)
    x_hat_ca   = np.zeros((G, S, L, 3), dtype=np.float32)
    rmsds      = np.zeros((G, S), dtype=np.float32)
    lddts      = np.zeros((G, S), dtype=np.float32)

    print(f"\nRunning inference: {G} gammas × {S} seeds = {G*S} forward passes")
    print("-" * 50)

    for gi, g in enumerate(gammas):
        for si in range(S):
            with torch.no_grad():
                batch = make_batch(example, g, device, seed=si)
                pred = model(batch)
                scale = eqm_reconstruction_scale(batch.gamma)
                x_hat = (batch.x_gamma - scale * pred)[0].cpu().numpy()

            def ca_ang(arr):
                return arr[:, CA_ATOM_ID, :] * COORD_SCALE

            xc = ca_ang(batch.x_clean[0].cpu().numpy())
            xn = ca_ang(batch.x_gamma[0].cpu().numpy())
            xh = ca_ang(x_hat)

            x_clean_ca[gi, si] = xc
            x_noisy_ca[gi, si] = xn
            x_hat_ca[gi, si]   = xh

            rmsd = float(np.sqrt(((xh[ca_mask] - xc[ca_mask]) ** 2).sum(-1).mean()))
            lddt = hard_lddt(xh, xc, ca_mask)
            rmsds[gi, si] = rmsd
            lddts[gi, si] = lddt

        print(f"  γ={g:.2f}  RMSD={rmsds[gi].mean():.2f}±{rmsds[gi].std():.2f} Å"
              f"  LDDT={lddts[gi].mean():.3f}±{lddts[gi].std():.3f}")

    # ── EqM iterative sampling: Euler ODE + NAG ──
    DS    = args.n_sample_seeds
    STEPS = args.n_sample_steps
    true_ca_ang = ex_c.coords[:, CA_ATOM_ID, :].numpy() * COORD_SCALE

    def _run_sampler(sampler_fn, label):
        final_arr = np.zeros((DS, L, 3), dtype=np.float32)
        traj_arr  = None
        rmsd_arr  = np.zeros(DS, dtype=np.float32)
        sched_out = None
        print(f"\n{label} sampling: {STEPS} steps × {DS} seeds")
        print("-" * 50)
        for si in range(DS):
            final_ca, traj_ca, sched = sampler_fn(
                model, example, n_steps=STEPS, seed=si, device=device
            )
            final_arr[si] = final_ca
            if traj_arr is None:
                traj_arr = np.zeros((DS, len(traj_ca), L, 3), dtype=np.float32)
            traj_arr[si] = traj_ca
            rmsd = float(np.sqrt(((final_ca[ca_mask] - true_ca_ang[ca_mask]) ** 2).sum(-1).mean()))
            lddt = hard_lddt(final_ca, true_ca_ang, ca_mask)
            rmsd_arr[si] = rmsd
            sched_out = sched
            print(f"  seed {si}  RMSD={rmsd:.2f} Å  LDDT={lddt:.3f}")
        return final_arr, traj_arr, rmsd_arr, sched_out

    euler_final, euler_traj, euler_rmsd, euler_sched = _run_sampler(sample_eqm_euler, "EqM Euler")
    nag_final,   nag_traj,   nag_rmsd,   nag_sched   = _run_sampler(sample_eqm_nag,   "EqM NAG  ")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        x_clean_ca=x_clean_ca,
        x_noisy_ca=x_noisy_ca,
        x_hat_ca=x_hat_ca,
        ca_mask=ca_mask,
        gammas=np.array(gammas, dtype=np.float32),
        rmsds=rmsds,
        lddts=lddts,
        euler_traj_ca=euler_traj,
        euler_final_ca=euler_final,
        euler_gammas=euler_sched.astype(np.float32),
        euler_rmsd=euler_rmsd,
        nag_traj_ca=nag_traj,
        nag_final_ca=nag_final,
        nag_gammas=nag_sched.astype(np.float32),
        nag_rmsd=nag_rmsd,
    )
    print(f"\nSaved: {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
