#!/usr/bin/env python3
"""Controlled single-process sweep of residue-coherent self-avoidance guidance."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks"))

from run_inference import (  # noqa: E402
    kabsch_align_to_gt,
    make_sampling_batch,
    prepare_static_batch,
    save_cif,
    save_pdb,
)

from mambafold.data.constants import COORD_SCALE  # noqa: E402
from mambafold.data.dataset import RCSBDataset  # noqa: E402
from mambafold.data.transforms import center_and_scale  # noqa: E402
from mambafold.losses.stereochemistry import all_atom_vdw_clash_loss  # noqa: E402
from mambafold.sampling import GeometryGuidanceConfig, sample  # noqa: E402
from mambafold.train.distributed import enable_cuda_perf_flags  # noqa: E402
from mambafold.train.trainer import load_from_checkpoint  # noqa: E402


def _scale_label(scale: float) -> str:
    return f"steric_{scale:g}".replace(".", "p")


def _vdw_channel_label(scale: float, every_n_steps: int) -> str:
    scale_label = f"{scale:.2f}".replace(".", "p")
    return f"steric_1_vdw_sep_s{scale_label}_e{every_n_steps}"


def _parse_vdw_channel_grid(values: list[str] | None) -> list[tuple[float, int]]:
    grid: list[tuple[float, int]] = []
    for value in values or []:
        try:
            scale_text, interval_text = value.split(":", maxsplit=1)
            scale = float(scale_text)
            interval = int(interval_text)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid independent VDW dose {value!r}; expected SCALE:INTERVAL"
            ) from exc
        if scale <= 0.0 or interval < 1:
            raise ValueError(f"invalid independent VDW dose {value!r}; values must be positive")
        grid.append((scale, interval))
    if len(set(grid)) != len(grid):
        raise ValueError("independent VDW dose grid contains duplicates")
    return grid


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _conditions(
    scales: list[float],
    *,
    physics_ablation: bool = False,
    vdw_weight: float = 0.2,
    segment_weight: float = 0.25,
    segment_every_n_steps: int = 4,
    vdw_channel_grid: list[tuple[float, int]] | None = None,
) -> list[tuple[str, GeometryGuidanceConfig]]:
    conditions = [
        ("baseline", GeometryGuidanceConfig(scale=0.0)),
        (
            "local_only",
            GeometryGuidanceConfig.stereochemical(
                scale=0.03,
                start=0.6,
                every_n_steps=2,
            ),
        ),
        (
            "split_local_control",
            GeometryGuidanceConfig.self_avoidance(
                local_scale=0.03,
                steric_scale=0.0,
                local_start=0.65,
                local_every_n_steps=2,
                steric_start=0.35,
                steric_every_n_steps=1,
                steric_smoothing_radius=4,
            ),
        ),
    ]
    conditions.extend(
        (
            _scale_label(scale),
            GeometryGuidanceConfig.self_avoidance(
                local_scale=0.03,
                steric_scale=scale,
                local_start=0.65,
                local_every_n_steps=2,
                steric_start=0.35,
                steric_every_n_steps=1,
                steric_smoothing_radius=4,
            ),
        )
        for scale in scales
    )
    if physics_ablation:
        by_name = dict(conditions)
        steric_control = by_name.get("steric_1")
        if steric_control is None:
            steric_control = GeometryGuidanceConfig.self_avoidance(
                local_scale=0.03,
                steric_scale=1.0,
                local_start=0.65,
                local_every_n_steps=2,
                steric_start=0.35,
                steric_every_n_steps=1,
                steric_smoothing_radius=4,
            )
            conditions.append(("steric_1", steric_control))
        # The interval is inert while the segment weight is zero, but keeping
        # it identical in control and treatment makes the paired manifest
        # contract vary only the intended guidance weight.
        steric_control = replace(
            steric_control,
            steric_segment_every_n_steps=segment_every_n_steps,
        )
        conditions = [
            (name, steric_control if name == "steric_1" else config) for name, config in conditions
        ]
        conditions.extend(
            (
                (
                    "steric_1_vdw",
                    replace(steric_control, all_atom_clash_weight=vdw_weight),
                ),
                (
                    "steric_1_segment",
                    replace(
                        steric_control,
                        steric_segment_weight=segment_weight,
                        steric_segment_every_n_steps=segment_every_n_steps,
                    ),
                ),
                (
                    "steric_1_vdw_segment",
                    replace(
                        steric_control,
                        all_atom_clash_weight=vdw_weight,
                        steric_segment_weight=segment_weight,
                        steric_segment_every_n_steps=segment_every_n_steps,
                    ),
                ),
            )
        )
    if vdw_channel_grid:
        by_name = dict(conditions)
        steric_control = by_name.get("steric_1")
        if steric_control is None:
            raise ValueError("independent VDW doses require steric scale 1.0 in --steric_scales")
        conditions.extend(
            (
                _vdw_channel_label(scale, interval),
                replace(
                    steric_control,
                    vdw_scale=scale,
                    vdw_every_n_steps=interval,
                ),
            )
            for scale, interval in vdw_channel_grid
        )
    for _, config in conditions:
        config.validate()
    return conditions


def main() -> None:
    enable_cuda_perf_flags()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--ckpt_provenance", type=Path, default=None)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--esm_dir", type=Path, required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--steric_scales",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.4, 0.8],
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Optional subset of condition labels to run (useful for a full-step smoke).",
    )
    parser.add_argument(
        "--physics_ablation",
        action="store_true",
        help="Expose steric-1 plus VDW, segment, and combined conservative refinements.",
    )
    parser.add_argument("--vdw_weight", type=float, default=0.2)
    parser.add_argument("--segment_weight", type=float, default=0.25)
    parser.add_argument("--segment_every_n_steps", type=int, default=4)
    parser.add_argument(
        "--vdw_channel_grid",
        nargs="+",
        default=None,
        metavar="SCALE:INTERVAL",
        help=(
            "Independent severe-overlap VDW doses. The channel uses a 1.5-A "
            "overlap tolerance and a 0.01-A per-call step cap."
        ),
    )
    parser.add_argument(
        "--record_vdw_losses",
        action="store_true",
        help=(
            "Record final VDW losses at the legacy 0.6-A and "
            "OpenStructure-aligned 1.5-A overlap tolerances."
        ),
    )
    args = parser.parse_args()

    if (args.out / "inference").exists() or (args.out / "sweep_manifest.json").exists():
        raise SystemExit(f"Refusing to overwrite existing sweep outputs: {args.out}")
    if not args.ckpt.is_file():
        raise SystemExit(f"Missing checkpoint: {args.ckpt}")
    if any(scale <= 0.0 for scale in args.steric_scales):
        parser.error("--steric_scales must contain only positive values")
    if args.vdw_weight <= 0.0 or args.segment_weight <= 0.0:
        parser.error("--vdw_weight and --segment_weight must be positive")
    if args.segment_every_n_steps < 1:
        parser.error("--segment_every_n_steps must be positive")
    try:
        vdw_channel_grid = _parse_vdw_channel_grid(args.vdw_channel_grid)
    except ValueError as exc:
        parser.error(str(exc))

    conditions = _conditions(
        args.steric_scales,
        physics_ablation=args.physics_ablation,
        vdw_weight=args.vdw_weight,
        segment_weight=args.segment_weight,
        segment_every_n_steps=args.segment_every_n_steps,
        vdw_channel_grid=vdw_channel_grid,
    )
    if args.conditions is not None:
        conditions_by_name = dict(conditions)
        unknown = sorted(set(args.conditions) - conditions_by_name.keys())
        if unknown:
            parser.error(f"unknown --conditions {unknown}; available={list(conditions_by_name)}")
        conditions = [(name, conditions_by_name[name]) for name in args.conditions]
    target_ids = [value.strip() for value in args.ids.read_text().split() if value.strip()]
    if not target_ids:
        raise SystemExit(f"No target IDs in {args.ids}")
    args.out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_device_name = torch.cuda.get_device_name() if device == "cuda" else None
    autocast_dtype = (
        ("bfloat16" if torch.cuda.is_bf16_supported() else "float16") if device == "cuda" else None
    )
    print(
        f"[load] checkpoint={args.ckpt} device={device} "
        f"cuda_device={cuda_device_name} autocast={autocast_dtype}",
        flush=True,
    )
    model = load_from_checkpoint(str(args.ckpt), device, use_ema=True)
    model.eval()
    checkpoint_sha256 = _sha256(args.ckpt)

    dataset = RCSBDataset(
        str(args.data_dir),
        max_length=args.max_length,
        min_length=10,
        min_obs_ratio=0.0,
        esm_dir=str(args.esm_dir),
        single_chain_only=True,
    )
    targets = []
    for target_id in target_ids:
        npz = REPO / args.data_dir / target_id[1:3] / f"{target_id}.npz"
        if not npz.is_file():
            raise SystemExit(f"Missing target NPZ: {npz}")
        dataset.files = [npz]
        example = dataset[0]
        if example is None:
            raise SystemExit(f"Invalid single-chain target: {target_id}")
        centered = center_and_scale(example)
        targets.append(
            {
                "id": target_id,
                "example": example,
                "centered": centered,
                "static_batch": prepare_static_batch(centered, device),
            }
        )
    print(
        f"[ready] model loaded once; targets={len(targets)} conditions={len(conditions)}",
        flush=True,
    )

    # Compile/warm each target length before timing any condition.  Every
    # measured sample resets its seed, so this does not alter the predictions.
    for target in targets:
        print(f"[warmup] {target['id']} L={target['example'].seq_len}", flush=True)
        sample(
            model,
            target["example"],
            lambda x, t, static=target["static_batch"]: make_sampling_batch(static, x, t),
            n_steps=2,
            seed=997,
            device=device,
            sampler="sde",
            sde_tau=0.01,
            sde_eps=0.01,
            sde_w_cutoff=0.99,
            sde_log_timesteps=True,
            record_trajectory=False,
            geometry_guidance=None,
        )
    if device == "cuda":
        torch.cuda.synchronize()

    condition_manifests = []
    for condition_index, (condition, guidance) in enumerate(conditions, start=1):
        condition_dir = args.out / "inference" / condition
        condition_dir.mkdir(parents=True)
        rows = []
        print(
            f"[condition {condition_index}/{len(conditions)}] {condition} "
            f"local={guidance.scale:g} steric={guidance.steric_scale:g} "
            f"vdw={guidance.all_atom_clash_weight:g} "
            f"vdw_sep={guidance.vdw_scale:g}/{guidance.vdw_every_n_steps} "
            f"segment={guidance.steric_segment_weight:g}/"
            f"{guidance.steric_segment_every_n_steps}",
            flush=True,
        )
        for target in targets:
            target_id = target["id"]
            example = target["example"]
            centered = target["centered"]
            static_batch = target["static_batch"]
            length = example.seq_len
            atom_mask = centered.atom_mask.numpy().astype(bool)
            observed_mask = (centered.atom_mask & centered.observed_mask).numpy().astype(bool)
            true_aa = centered.coords.numpy() * COORD_SCALE
            res_type = centered.res_type.numpy()
            chain_id = centered.chain_id.numpy()
            b_zero = np.zeros_like(observed_mask, dtype=np.float32)

            save_pdb(
                true_aa,
                res_type,
                observed_mask,
                b_zero,
                chain_id,
                condition_dir / f"{target_id}_gt.pdb",
            )
            save_cif(
                true_aa,
                res_type,
                observed_mask,
                b_zero,
                chain_id,
                condition_dir / f"{target_id}_gt.cif",
                f"{target_id}_gt",
            )

            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            _, pred_aa, _, _, confidence = sample(
                model,
                example,
                lambda x, t: make_sampling_batch(static_batch, x, t),
                n_steps=args.n_steps,
                seed=args.seed,
                device=device,
                sampler="sde",
                sde_tau=0.01,
                sde_eps=0.01,
                sde_w_cutoff=0.99,
                sde_log_timesteps=True,
                record_trajectory=False,
                geometry_guidance=guidance,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            runtime_s = time.perf_counter() - started
            peak_vram_gib = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0

            final_vdw_losses = None
            if args.record_vdw_losses:
                pred_internal = torch.as_tensor(
                    pred_aa / COORD_SCALE,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                final_vdw_losses = {
                    f"final_vdw_loss_tol_{str(tolerance).replace('.', 'p')}": float(
                        all_atom_vdw_clash_loss(
                            pred_internal,
                            static_batch.res_type,
                            static_batch.atom_mask.bool(),
                            static_batch.res_mask.bool(),
                            chain_id=static_batch.chain_id,
                            res_seq_nums=static_batch.res_seq_nums,
                            overlap_tolerance_A=tolerance,
                        ).item()
                    )
                    for tolerance in (0.6, 1.5)
                }

            pred_aa = kabsch_align_to_gt(pred_aa, true_aa, observed_mask)
            b_factor = (
                np.asarray(confidence, dtype=np.float32)[:, None]
                * 100.0
                * atom_mask.astype(np.float32)
            )
            for suffix in ("pred_seed0", "pred"):
                save_pdb(
                    pred_aa,
                    res_type,
                    atom_mask,
                    b_factor,
                    chain_id,
                    condition_dir / f"{target_id}_{suffix}.pdb",
                )
                save_cif(
                    pred_aa,
                    res_type,
                    atom_mask,
                    b_factor,
                    chain_id,
                    condition_dir / f"{target_id}_{suffix}.cif",
                    f"{target_id}_{suffix}",
                )
            row = {
                "pdb_id": target_id,
                "L": int(length),
                "n_chains": 1,
                "runtime_s": runtime_s,
                "peak_vram_gib": peak_vram_gib,
            }
            if final_vdw_losses is not None:
                row.update(final_vdw_losses)
            rows.append(row)
            print(
                f"[result] {condition}/{target_id} L={length} "
                f"time={runtime_s:.2f}s peak={peak_vram_gib:.2f}GiB",
                flush=True,
            )

        manifest = {
            "schema_version": 1,
            "condition": condition,
            "checkpoint": str(args.ckpt_provenance or args.ckpt),
            "checkpoint_staged": args.ckpt_provenance is not None,
            "checkpoint_sha256": checkpoint_sha256,
            "ids_file": str(args.ids),
            "sampler": "sde",
            "n_steps": args.n_steps,
            "seed": args.seed,
            "sde_tau": 0.01,
            "sde_eps": 0.01,
            "sde_w_cutoff": 0.99,
            "sde_log_timesteps": True,
            "cuda_device_name": cuda_device_name,
            "autocast_dtype": autocast_dtype,
            "geometry_guidance": asdict(guidance),
            "rows": rows,
        }
        (condition_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        condition_manifests.append(manifest)

    sweep_manifest = {
        "schema_version": 1,
        "experiment": "self_overlap_guidance_v1",
        "single_process_model_load": True,
        "conditions": [manifest["condition"] for manifest in condition_manifests],
        "target_count": len(targets),
        "condition_manifests": [
            str(Path("inference") / manifest["condition"] / "manifest.json")
            for manifest in condition_manifests
        ],
    }
    (args.out / "sweep_manifest.json").write_text(json.dumps(sweep_manifest, indent=2) + "\n")
    print(f"[done] {args.out}", flush=True)


if __name__ == "__main__":
    main()
