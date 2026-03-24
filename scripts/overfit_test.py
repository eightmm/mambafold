#!/usr/bin/env python
"""Overfit test: verify model can memorize a single fixed batch.

Usage:
    # Synthetic batch (no data needed):
    PYTHONPATH=src python scripts/overfit_test.py

    # Real data (loads first .pt file from dir):
    PYTHONPATH=src python scripts/overfit_test.py --data_dir /path/to/afdb_pt

    # No wandb:
    PYTHONPATH=src python scripts/overfit_test.py --no_wandb
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import wandb

from mambafold.data.collate import ProteinCollator
from mambafold.data.constants import CA_ATOM_ID, MAX_ATOMS_PER_RES
from mambafold.data.dataset import AFDBDataset
from mambafold.data.transforms import center_and_scale, eqm_corrupt
from mambafold.data.types import ProteinBatch
from mambafold.model.mambafold import MambaFoldEqM
from mambafold.train.ema import EMA
from mambafold.train.engine import train_step


def make_synthetic_batch(B: int = 2, L: int = 32, device: str = "cuda") -> ProteinBatch:
    """Fixed synthetic batch for overfit testing (no data files needed)."""
    A = MAX_ATOMS_PER_RES

    # Use fixed seed so batch is identical every call
    gen = torch.Generator().manual_seed(42)
    rng = lambda *shape, **kw: torch.empty(*shape, **kw).normal_(generator=gen)

    res_type = torch.zeros(B, L, dtype=torch.long)
    atom_type = torch.zeros(B, L, A, dtype=torch.long)
    res_mask = torch.ones(B, L, dtype=torch.bool)

    # Only first 5 atom slots active (backbone + CB) for realism
    atom_mask = torch.zeros(B, L, A, dtype=torch.bool)
    atom_mask[:, :, :5] = True
    valid_mask = atom_mask.clone()
    ca_mask = torch.ones(B, L, dtype=torch.bool)

    x_clean = rng(B, L, A, 3)
    x_clean = x_clean * atom_mask.unsqueeze(-1)

    gamma = torch.full((B, 1, 1, 1), 0.8)  # near-clean; easy target
    eps = rng(B, L, A, 3)
    x_gamma = gamma * x_clean + (1 - gamma) * eps
    x_gamma = x_gamma * atom_mask.unsqueeze(-1)
    eps = eps * atom_mask.unsqueeze(-1)

    return ProteinBatch(
        res_type=res_type,
        atom_type=atom_type,
        res_mask=res_mask,
        atom_mask=atom_mask,
        valid_mask=valid_mask,
        ca_mask=ca_mask,
        x_clean=x_clean,
        x_gamma=x_gamma,
        eps=eps,
        gamma=gamma,
        esm=None,
    ).to(torch.device(device if torch.cuda.is_available() else "cpu"))


def load_real_example(data_dir: str):
    """Load first protein from data_dir as a ProteinExample (pre-corruption)."""
    dataset = AFDBDataset(data_dir=data_dir, max_length=128)
    for i in range(len(dataset)):
        example = dataset[i]
        if example is not None:
            print(f"Loaded: {dataset.files[i].name} (L={example.seq_len})")
            return example
    raise RuntimeError(f"No valid proteins found in {data_dir}")


def build_small_model(device: str) -> MambaFoldEqM:
    """Small model for fast overfit test.

    Uses standard Mamba3-compatible dimensions (d_inner must be large enough
    for tilelang warp partition: d_model * expand >= 8 * headdim).
    """
    return MambaFoldEqM(
        d_atom=256,
        d_res=256,
        d_plm=32,
        n_atom_enc=1,
        n_trunk=2,
        n_atom_dec=1,
        use_plm=False,
        atom_d_state=32,
        atom_mimo_rank=1,  # SISO: bypass tilelang MIMO (A5000 shared mem limit)
        atom_headdim=64,
        d_state=32,
        mimo_rank=1,       # SISO
        headdim=64,
    ).to(torch.device(device if torch.cuda.is_available() else "cpu"))


def make_batch_from_example(example, device: str, fixed_gamma: float = None) -> ProteinBatch:
    """Apply center+scale and corrupt from a ProteinExample.

    Args:
        fixed_gamma: if given, use this gamma instead of sampling randomly.
    """
    ex = center_and_scale(example)
    L = ex.seq_len

    x_gamma, eps, gamma_val = eqm_corrupt(ex.coords, ex.atom_mask)

    if fixed_gamma is not None:
        gamma_val = fixed_gamma
        mask_f = ex.atom_mask.unsqueeze(-1).to(ex.coords.dtype)
        eps_fresh = torch.randn_like(ex.coords)
        x_gamma = (gamma_val * ex.coords + (1 - gamma_val) * eps_fresh) * mask_f
        eps = eps_fresh * mask_f

    res_mask = torch.ones(1, L, dtype=torch.bool)
    atom_mask = ex.atom_mask.unsqueeze(0)
    valid_mask = (ex.atom_mask & ex.observed_mask).unsqueeze(0)
    ca_mask = ex.atom_mask[:, CA_ATOM_ID].unsqueeze(0)

    return ProteinBatch(
        res_type=ex.res_type.unsqueeze(0),
        atom_type=ex.atom_type.unsqueeze(0),
        res_mask=res_mask,
        atom_mask=atom_mask,
        valid_mask=valid_mask,
        ca_mask=ca_mask,
        x_clean=ex.coords.unsqueeze(0),
        x_gamma=x_gamma.unsqueeze(0),
        eps=eps.unsqueeze(0),
        gamma=torch.tensor([[[[gamma_val]]]]),
        esm=None,
    ).to(torch.device(device))


def run_overfit(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data source
    if args.data_dir:
        print(f"Loading protein from {args.data_dir}")
        example = load_real_example(args.data_dir)
        resample_per_step = True
    else:
        print("Using synthetic batch (B=2, L=32)")
        example = None
        resample_per_step = False
        batch = make_synthetic_batch(device=device)
        B, L = batch.batch_size, batch.max_len
        print(f"Batch: B={B}, L={L}, device={batch.device}")

    model = build_small_model(device)
    ema = EMA(model, decay=0.999)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {n_params:.2f}M parameters")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    if not args.no_wandb:
        wandb.init(
            project="mambafold",
            name="overfit-test",
            config=vars(args),
            tags=["overfit"],
        )

    losses = []
    # 50 evenly spaced gammas: (0.5/50, 1.5/50, ..., 49.5/50)
    gamma_grid = [(i + 0.5) / 50 for i in range(50)]
    use_grid = resample_per_step and args.fixed_gammas
    if use_grid:
        print(f"Gamma grid: {len(gamma_grid)} values, {args.n_steps // 50} cycles")

    print(f"\nOverfit test: {args.n_steps} steps, lr={args.lr}, "
          f"{'fixed_gammas(50)' if use_grid else 'resample_gamma' if resample_per_step else 'synthetic'}")
    print("-" * 60)

    for step in range(1, args.n_steps + 1):
        if resample_per_step:
            fixed_g = gamma_grid[(step - 1) % 50] if use_grid else None
            batch = make_batch_from_example(example, device, fixed_gamma=fixed_g)

        metrics = train_step(
            model, batch, optimizer,
            grad_clip=1.0,
            alpha_mode="const",
            use_amp=(device == "cuda"),
        )
        ema.update(model)
        losses.append(metrics["loss"])

        if step % 20 == 0 or step == 1:
            print(
                f"  step {step:>4d}/{args.n_steps} | "
                f"loss={metrics['loss']:.4f} "
                f"eqm={metrics['eqm']:.4f} "
                f"lddt={metrics['lddt']:.4f}"
            )
            if not args.no_wandb:
                wandb.log({
                    "overfit/loss": metrics["loss"],
                    "overfit/eqm": metrics["eqm"],
                    "overfit/lddt": metrics["lddt"],
                }, step=step)

    # Summary
    loss_init = losses[0]
    loss_final = losses[-1]
    drop_pct = (loss_init - loss_final) / loss_init * 100

    print("-" * 60)
    print(f"Initial loss : {loss_init:.4f}")
    print(f"Final loss   : {loss_final:.4f}")
    print(f"Drop         : {drop_pct:.1f}%")

    if not args.no_wandb:
        wandb.summary["loss_initial"] = loss_init
        wandb.summary["loss_final"] = loss_final
        wandb.summary["loss_drop_pct"] = drop_pct
        wandb.finish()

    # With variable gamma sampling, final loss variance is high.
    # Use minimum loss achieved as the signal instead.
    loss_min = min(losses)
    drop_min_pct = (loss_init - loss_min) / loss_init * 100
    print(f"Min loss     : {loss_min:.4f} (drop {drop_min_pct:.1f}%)")

    threshold = 50.0  # expect min loss to drop >50% from initial
    if drop_min_pct >= threshold:
        print(f"\nPASS: min loss dropped {drop_min_pct:.1f}% >= {threshold}%")
        return 0
    else:
        print(f"\nFAIL: min loss dropped only {drop_min_pct:.1f}% (expected >= {threshold}%)")
        return 1


def main():
    parser = argparse.ArgumentParser(description="MambaFold overfit test")
    parser.add_argument("--data_dir", default=None, help="Path to .pt protein files (optional)")
    parser.add_argument("--n_steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-3, help="High LR for fast overfit")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--fixed_gammas", action="store_true",
                        help="Cycle through 50 evenly-spaced gammas instead of random sampling")
    args = parser.parse_args()

    sys.exit(run_overfit(args))


if __name__ == "__main__":
    main()
