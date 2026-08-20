"""Benchmark sampler step-count trade-offs with one loaded checkpoint."""

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
sys.path.insert(0, str(REPO / "benchmarks"))

from run_inference import (  # noqa: E402
    kabsch_align_to_gt,
    make_sampling_batch,
    prepare_static_batch,
    save_pdb,
)

from mambafold.data.constants import COORD_SCALE  # noqa: E402
from mambafold.data.dataset import RCSBDataset  # noqa: E402
from mambafold.data.transforms import center_and_scale  # noqa: E402
from mambafold.sampling import GeometryGuidanceConfig, sample  # noqa: E402
from mambafold.train.distributed import enable_cuda_perf_flags  # noqa: E402
from mambafold.train.trainer import load_from_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--pid", default="t1061")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--data_dir", default="data/casp_official/npz_70")
    parser.add_argument("--esm_dir", default="data/casp_official/esmc6b_70")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--steps", type=int, nargs="+", default=[500, 300, 200, 100])
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--solver", choices=("sde", "ode"), default="sde")
    parser.add_argument(
        "--timestep_schedule",
        choices=("log", "uniform"),
        default="log",
        help="SDE integration grid; uniform disables the SimpleFold log grid.",
    )
    parser.add_argument(
        "--geometry_guidance_scales",
        type=float,
        nargs="+",
        default=[0.0],
        help="Sweep scales in one loaded model process (for example: 0 0.02 0.05 0.1)",
    )
    parser.add_argument("--geometry_guidance_start", type=float, default=0.5)
    parser.add_argument("--geometry_guidance_every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise SystemExit(f"Output already exists: {args.out}")
    if not args.steps or any(step <= 0 for step in args.steps):
        raise SystemExit("--steps must contain positive integers")
    guidance_configs = [
        GeometryGuidanceConfig(
            scale=scale,
            start=args.geometry_guidance_start,
            every_n_steps=args.geometry_guidance_every,
        )
        for scale in args.geometry_guidance_scales
    ]
    try:
        for config in guidance_configs:
            config.validate()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    enable_cuda_perf_flags()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True)

    print(f"[load] ckpt={args.ckpt} device={device}", flush=True)
    model = load_from_checkpoint(args.ckpt, device, use_ema=True)
    model.eval()

    npz_path = REPO / args.data_dir / args.pid[1:3] / f"{args.pid}.npz"
    if not npz_path.is_file():
        raise SystemExit(f"Missing target: {npz_path}")
    dataset = RCSBDataset(
        args.data_dir,
        max_length=args.max_length,
        min_length=10,
        min_obs_ratio=0.0,
        esm_dir=args.esm_dir,
        single_chain_only=True,
    )
    dataset.files = [npz_path]
    example = dataset[0]
    if example is None:
        raise SystemExit(f"Invalid single-chain target: {args.pid}")

    centered = center_and_scale(example)
    static_batch = prepare_static_batch(centered, device)

    def batch_fn(x, ti):
        return make_sampling_batch(static_batch, x, ti)

    length = example.seq_len
    atom_mask = centered.atom_mask.numpy().astype(bool)
    observed_mask = (centered.atom_mask & centered.observed_mask).numpy().astype(bool)
    true_coords = centered.coords.numpy() * COORD_SCALE
    residue_types = centered.res_type.numpy()
    chain_ids = centered.chain_id.numpy()
    zero_b_factors = np.zeros_like(observed_mask, dtype=np.float32)

    common = {
        "seed": args.seed,
        "device": device,
        "sampler": args.solver,
        "sde_tau": 0.01,
        "sde_eps": 0.01,
        "sde_w_cutoff": 0.99,
        "sde_log_timesteps": args.timestep_schedule == "log",
        "record_trajectory": False,
    }

    print(f"[warmup] pid={args.pid} L={length} steps={args.warmup_steps}", flush=True)
    sample(model, example, batch_fn, n_steps=args.warmup_steps, **common)
    if device == "cuda":
        torch.cuda.synchronize()

    rows = []
    for geometry_guidance in guidance_configs:
        scale_label = f"{geometry_guidance.scale:g}".replace(".", "p")
        for step_count in args.steps:
            if device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            _, prediction, _, _, confidence = sample(
                model,
                example,
                batch_fn,
                n_steps=step_count,
                geometry_guidance=geometry_guidance,
                **common,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started

            step_dir = args.out / f"guidance_{scale_label}" / f"steps_{step_count}"
            step_dir.mkdir(parents=True)
            prediction = kabsch_align_to_gt(prediction, true_coords, observed_mask)
            b_factors = (
                np.asarray(confidence, dtype=np.float32)[:, None]
                * 100.0
                * atom_mask.astype(np.float32)
            )
            save_pdb(
                true_coords,
                residue_types,
                observed_mask,
                zero_b_factors,
                chain_ids,
                step_dir / f"{args.pid}_gt.pdb",
            )
            save_pdb(
                prediction,
                residue_types,
                atom_mask,
                b_factors,
                chain_ids,
                step_dir / f"{args.pid}_pred.pdb",
            )
            rows.append(
                {
                    "geometry_guidance_scale": geometry_guidance.scale,
                    "steps": step_count,
                    "elapsed_seconds": elapsed,
                    "seconds_per_step": elapsed / step_count,
                    "output_dir": str(step_dir),
                }
            )
            print(
                f"[result] guidance={geometry_guidance.scale:g} steps={step_count} "
                f"elapsed={elapsed:.3f}s per_step={elapsed / step_count:.5f}s",
                flush=True,
            )

    manifest = {
        "checkpoint": args.ckpt,
        "use_ema": True,
        "pid": args.pid,
        "length": length,
        "seed": args.seed,
        "sampler": args.solver,
        "timestep_schedule": args.timestep_schedule,
        "warmup_steps": args.warmup_steps,
        "geometry_guidance": {
            "scales": args.geometry_guidance_scales,
            "start": guidance_configs[0].start,
            "every_n_steps": guidance_configs[0].every_n_steps,
            "bond_weight": guidance_configs[0].bond_weight,
            "angle_weight": guidance_configs[0].angle_weight,
            "clash_weight": guidance_configs[0].clash_weight,
        },
        "rows": rows,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[done] {args.out}", flush=True)


if __name__ == "__main__":
    main()
