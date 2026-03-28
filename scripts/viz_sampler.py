"""Euler/NAG 샘플러 결과 시각화 — GPU 불필요, matplotlib만 사용."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


# ── Kabsch alignment (Cα superposition) ─────────────────────────────────────
def kabsch(P: np.ndarray, Q: np.ndarray):
    """Kabsch RMSD alignment: translate & rotate P onto Q.

    P, Q: [N, 3]  (only valid residues)
    Returns aligned P [N, 3] and RMSD float.
    """
    P = P - P.mean(0)
    Q = Q - Q.mean(0)
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    P_rot = P @ R.T
    rmsd = float(np.sqrt(((P_rot - Q) ** 2).sum(-1).mean()))
    return P_rot, rmsd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", nargs="?",
                        default="outputs/overfit/gpu2_uniform/inference.npz")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    npz = Path(args.npz)
    out = Path(args.out) if args.out else npz.parent / "viz_sampler.png"
    d = np.load(npz)

    true_ca       = d["x_clean_ca"][0, 0]              # [L, 3]
    mask          = d["ca_mask"].astype(bool)           # [L]
    atom_mask     = d["atom_mask"].astype(bool) if "atom_mask" in d else None
    atom_mask_flat= atom_mask.reshape(-1) if atom_mask is not None else None
    euler_traj    = d["euler_traj_ca"]                  # [3, 50, L, 3]
    euler_fin     = d["euler_final_ca"]                 # [3, L, 3]
    euler_fin_all = d["euler_final_all"] if "euler_final_all" in d else None
    euler_gs      = d["euler_gammas"]                   # [51]
    nag_fin       = d["nag_final_ca"]                   # [3, L, 3]
    nag_fin_all   = d["nag_final_all"] if "nag_final_all" in d else None
    scale = 1.0
    T = true_ca[mask]   # [N_valid, 3]

    # Cα Kabsch RMSD
    euler_rmsd_k = np.array([kabsch(euler_fin[i, mask], T)[1] for i in range(3)])
    nag_rmsd_k   = np.array([kabsch(nag_fin[i, mask],   T)[1] for i in range(3)])

    # all-atom Kabsch RMSD (npz에 저장된 값 사용, 없으면 Cα로 대체)
    euler_rmsd_aa_k = np.array(d["euler_rmsd_aa"]) if "euler_rmsd_aa" in d else euler_rmsd_k
    nag_rmsd_aa_k   = np.array(d["nag_rmsd_aa"])   if "nag_rmsd_aa"   in d else nag_rmsd_k
    has_all = euler_fin_all is not None and atom_mask_flat is not None

    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"EqM Sampler Analysis  ({npz.parent.name})", fontsize=14, y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

    colors = ["tab:blue", "tab:orange", "tab:green"]
    seed_labels = [f"seed {i}" for i in range(3)]

    # ── Panel 0: True vs Euler final (seed 0) — 2D projection (PC1 vs PC2) ──
    ax = fig.add_subplot(gs[0, 0])
    # PCA on true structure
    T_c = T - T.mean(0)
    _, _, Vt_pca = np.linalg.svd(T_c, full_matrices=False)
    proj = lambda x: (x - x.mean(0)) @ Vt_pca[:2].T

    ax.plot(*proj(T).T, "k-", lw=1.5, label="True", alpha=0.8)
    for i in range(3):
        F = euler_fin[i, mask] * scale
        F_aligned, rmsd = kabsch(F, T)
        ax.plot(*proj(F_aligned).T, "-", color=colors[i],
                lw=1.0, alpha=0.6, label=f"{seed_labels[i]} {rmsd:.1f}Å")
    ax.set_title("Euler final (Kabsch aligned, PCA proj)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Panel 1: Trajectory RMSD vs step ─────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    n_steps = euler_traj.shape[1]
    for i in range(3):
        traj_rmsds = []
        for t in range(n_steps):
            pred = euler_traj[i, t, mask] * scale
            _, r = kabsch(pred, T)
            traj_rmsds.append(r)
        ax.plot(range(n_steps), traj_rmsds, color=colors[i],
                label=f"{seed_labels[i]}", lw=1.5)
    ax.set_title("Euler: RMSD along trajectory")
    ax.set_xlabel("Step"); ax.set_ylabel("RMSD vs true (Å)")
    ax.legend(fontsize=8)
    ax.axhline(2.0, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: True vs NAG final (2D projection) ───────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(*proj(T).T, "k-", lw=1.5, label="True", alpha=0.8)
    for i in range(3):
        F = nag_fin[i, mask] * scale
        F_aligned, rmsd = kabsch(F, T)
        ax.plot(*proj(F_aligned).T, "-", color=colors[i],
                lw=1.0, alpha=0.6, label=f"{seed_labels[i]} {rmsd:.1f}Å")
    ax.set_title("NAG final (Kabsch aligned, PCA proj)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Panel 3: Per-residue RMSD (Euler seed 1) ─────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    for i in range(3):
        F = euler_fin[i, mask] * scale
        F_aligned, _ = kabsch(F, T)
        per_res = np.sqrt(((F_aligned - T) ** 2).sum(-1))
        ax.plot(per_res, color=colors[i], lw=1.0,
                label=f"{seed_labels[i]}", alpha=0.7)
    ax.set_title("Euler: Per-residue RMSD after Kabsch")
    ax.set_xlabel("Residue"); ax.set_ylabel("RMSD (Å)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: γ schedule used during trajectory ───────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(euler_gs, "b-o", ms=3, label="γ schedule")
    ax.set_title("Euler γ schedule (sampling path)")
    ax.set_xlabel("Step"); ax.set_ylabel("γ")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ── Panel 5: RMSD summary bar (Kabsch, Cα vs all-atom) ──────────────────
    ax = fig.add_subplot(gs[1, 2])
    x = np.arange(3)
    w = 0.22
    ax.bar(x - 1.5*w, euler_rmsd_k,    w, label="Euler Cα",    color="steelblue",  alpha=0.85)
    ax.bar(x - 0.5*w, euler_rmsd_aa_k, w, label="Euler all",   color="cornflowerblue", alpha=0.85)
    ax.bar(x + 0.5*w, nag_rmsd_k,      w, label="NAG Cα",      color="darkorange",  alpha=0.85)
    ax.bar(x + 1.5*w, nag_rmsd_aa_k,   w, label="NAG all",     color="peachpuff",   alpha=0.85, edgecolor="darkorange", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(seed_labels)
    ax.set_ylabel("Final RMSD (Å, Kabsch)")
    ax.set_title("Final RMSD by seed (Kabsch aligned)")
    ax.grid(axis="y", alpha=0.3)
    import json
    metrics_path = npz.parent / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        for key, color, lbl in [
            ("rmsd_ca_hi_gamma_mean", "green",  "1-step Cα γ>0.7"),
            ("rmsd_aa_hi_gamma_mean", "purple", "1-step all γ>0.7"),
        ]:
            ref = m.get(key)
            if ref is not None:
                ax.axhline(ref, color=color, ls="--", lw=1.0, label=f"{lbl} ({ref:.2f}Å)")
    ax.legend(fontsize=7)

    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
