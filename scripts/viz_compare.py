"""Visualize MambaFold predictions aligned to reference structure.

Usage (CPU OK):
    PYTHONPATH=src python scripts/viz_compare.py \
        --ref outputs/infer_2025/10af/10af.npz \
        --inf outputs/infer_2025/10af/inference.npz \
        --out outputs/infer_2025/10af/compare.png \
        [--pdb_id 10AF]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mambafold.data.constants import CA_ATOM_ID, COORD_SCALE
from mambafold.data.dataset import RCSBDataset
from mambafold.data.transforms import center_and_scale
from mambafold.utils.geometry import kabsch_align


def load_ref_ca(npz_path: str):
    """Extract Cα coords (Å) from Boltz-format reference npz."""
    ds = RCSBDataset(str(Path(npz_path).parent),
                     max_length=1024, min_length=1, min_obs_ratio=0.0)
    data = np.load(npz_path)
    ex = ds._canonicalize(data)
    if ex is None:
        raise ValueError(f"Failed to parse {npz_path}")
    ex_c = center_and_scale(ex)
    ca_mask = (ex_c.atom_mask[:, CA_ATOM_ID] & ex_c.observed_mask[:, CA_ATOM_ID]).numpy().astype(bool)
    ca_coords = ex_c.coords[:, CA_ATOM_ID, :].numpy() * COORD_SCALE   # [L, 3]
    return ca_coords, ca_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="Reference Boltz npz")
    parser.add_argument("--inf", required=True, help="inference.npz from infer_seq.py")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--pdb_id", default="")
    args = parser.parse_args()

    # ── Load reference ──────────────────────────────────────────────────────
    ref_ca, ref_mask = load_ref_ca(args.ref)
    ref_ca_valid = ref_ca[ref_mask]
    L = ref_ca.shape[0]
    print(f"Reference: L={L}, observed Cα={ref_mask.sum()}")

    # ── Load predictions ────────────────────────────────────────────────────
    inf = np.load(args.inf, allow_pickle=True)
    euler_preds = inf["euler_final_ca"]   # [S, L', 3]
    nag_preds   = inf["nag_final_ca"]     # [S, L', 3]
    pred_mask   = inf["ca_mask"]          # [L']

    # Align prediction mask to reference mask (take min length)
    n = min(ref_mask.sum(), pred_mask.sum())
    ref_aligned_ref = ref_ca_valid[:n]

    methods = {"Euler": euler_preds, "NAG": nag_preds}
    colors  = {"Euler": "tomato", "NAG": "steelblue"}

    # ── Compute RMSDs ───────────────────────────────────────────────────────
    results = {}
    for method, preds in methods.items():
        results[method] = []
        for si, pred in enumerate(preds):
            pred_valid = pred[pred_mask][:n]
            aligned, _, rmsd = kabsch_align(pred_valid, ref_aligned_ref)
            results[method].append({"aligned": aligned, "rmsd": rmsd, "seed": si})
        rmsds = [r["rmsd"] for r in results[method]]
        print(f"{method}: RMSD = {np.mean(rmsds):.2f} ± {np.std(rmsds):.2f} Å  "
              f"(best={min(rmsds):.2f}, worst={max(rmsds):.2f})")

    # ── Plot ────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n_seeds = euler_preds.shape[0]
    fig = plt.figure(figsize=(5 + 4 * n_seeds, 10))
    title = f"MambaFold · {args.pdb_id or Path(args.ref).stem} (2025) — Cα alignment"
    fig.suptitle(title, fontsize=12, fontweight="bold")

    gs = GridSpec(2, 1 + n_seeds, figure=fig, hspace=0.35, wspace=0.3)

    # ── Column 0: RMSD bar chart ─────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[:, 0])
    bar_x, bar_h, bar_c, bar_lbl = [], [], [], []
    xi = 0
    for method in ("Euler", "NAG"):
        for r in results[method]:
            bar_x.append(xi)
            bar_h.append(r["rmsd"])
            bar_c.append(colors[method])
            bar_lbl.append(f"{method}\ns{r['seed']}")
            xi += 1
        xi += 0.5

    bars = ax_bar.bar(bar_x, bar_h, color=bar_c, width=0.7, edgecolor="white", linewidth=0.5)
    for bar, h in zip(bars, bar_h):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax_bar.set_xticks(bar_x)
    ax_bar.set_xticklabels(bar_lbl, fontsize=8)
    ax_bar.set_ylabel("Kabsch RMSD (Å)")
    ax_bar.set_title("RMSD vs Reference", fontsize=10)
    ax_bar.grid(axis="y", alpha=0.3)
    # Legend
    from matplotlib.patches import Patch
    ax_bar.legend(handles=[Patch(color=colors[m], label=m) for m in colors],
                  fontsize=8, loc="upper right")

    # ── Columns 1+: per-seed XY and XZ overlays ──────────────────────────────
    for si in range(n_seeds):
        for row, (plane, dims) in enumerate([("XY", (0, 1)), ("XZ", (0, 2))]):
            ax = fig.add_subplot(gs[row, si + 1])
            d0, d1 = dims

            # Reference
            ax.plot(ref_ca_valid[:n, d0], ref_ca_valid[:n, d1],
                    "k-", lw=1.2, alpha=0.9, label="Reference", zorder=3)
            ax.scatter(ref_ca_valid[0, d0], ref_ca_valid[0, d1],
                       color="black", s=40, zorder=4, marker="^")

            for method in ("Euler", "NAG"):
                r = results[method][si]
                al = r["aligned"]
                ax.plot(al[:, d0], al[:, d1], "-", lw=0.8, alpha=0.7,
                        color=colors[method],
                        label=f"{method} ({r['rmsd']:.2f} Å)")
                ax.scatter(al[0, d0], al[0, d1], color=colors[method],
                           s=25, zorder=4, marker="^")

            ax.set_aspect("equal")
            ax.grid(alpha=0.25)
            if row == 0:
                ax.set_title(f"seed {si}", fontsize=10)
                ax.legend(fontsize=7, loc="upper right")
            ax.set_xlabel(f"{'XYZ'[d0]} (Å)", fontsize=8)
            ax.set_ylabel(f"{'XYZ'[d1]} (Å)", fontsize=8)
            ax.tick_params(labelsize=7)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
