#!/usr/bin/env python
"""Generate frozen-backbone rollout shards for a standalone pLDDT head.

Every target is sampled with all requested seeds in one or more same-length
batches.  The resulting final trunk latents and exact hard lDDT-Ca labels are
stored offline, so confidence training never loads or updates the folding
model.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from mambafold.confidence import hard_lddt_ca
from mambafold.confidence.rollout import (
    sequence_sha256,
    sha256_file,
    write_rollout_manifest,
    write_rollout_shard,
)
from mambafold.data.constants import AA_3TO1, CA_ATOM_ID, COORD_SCALE
from mambafold.data.dataset import RCSBDataset
from mambafold.data.esm import ESMC_6B_DIM, ESMC_6B_REVISION
from mambafold.data.transforms import center_and_scale
from mambafold.sampling import prepare_inference_batch, sample
from mambafold.train.distributed import enable_cuda_perf_flags
from mambafold.train.trainer import build_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="pLDDT YAML configuration")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--esm-dir", default=None)
    parser.add_argument("--file-list", default=None)
    parser.add_argument("--chain-list", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--expected-checkpoint-step", type=int, default=None)
    parser.add_argument("--expected-checkpoint-sha256", default=None)
    parser.add_argument(
        "--checkpoint-sha256-file",
        default=None,
        help="Optional sha256sum-style sidecar computed once by the launcher. "
        "Avoids every independent rollout rank rereading a multi-GB checkpoint.",
    )
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--seed-batch-size", type=int, default=None)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    return parser


def _value(args: argparse.Namespace, rollout: dict[str, Any], name: str, *, required=False):
    arg_name = name.replace("-", "_")
    value = getattr(args, arg_name, None)
    if value is None:
        value = rollout.get(arg_name)
    if required and (value is None or value == ""):
        raise ValueError(f"missing rollout setting {arg_name!r}; pass --{name}")
    return value


def _portable_target_id(path: Path, data_dir: Path, chain_origin: int | None = None) -> str:
    try:
        relative = path.relative_to(data_dir).as_posix()
    except ValueError:
        relative = path.name
    identity = relative if chain_origin is None else f"{relative}\0chain={chain_origin}"
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
    chain = "" if chain_origin is None else f"-ch{chain_origin}"
    return f"{path.stem}{chain}-{suffix}"


def _crop_seed(base_seed: int, logical_file: str) -> int:
    digest = hashlib.sha256(f"{base_seed}\0{logical_file}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _polymer_geometry_quality(pred_ca_A: torch.Tensor) -> tuple[float, float]:
    """Return median adjacent C-alpha distance and polymer-like step fraction."""
    if pred_ca_A.ndim != 2 or pred_ca_A.shape[-1] != 3 or pred_ca_A.shape[0] < 2:
        raise ValueError(f"pred_ca_A must have shape [L,3] with L >= 2, got {pred_ca_A.shape}")
    ca_step = (pred_ca_A[1:] - pred_ca_A[:-1]).norm(dim=-1)
    median_ca_step = float(ca_step.median())
    normal_ca_fraction = float(((ca_step - 3.8).abs() < 0.5).float().mean())
    return median_ca_step, normal_ca_fraction


def _load_deterministic_example(
    dataset: RCSBDataset,
    index: int,
    *,
    crop_seed: int,
    data_dir: Path,
):
    path, chain_origin = _dataset_target(dataset, index)
    try:
        logical_file = path.relative_to(data_dir).as_posix()
    except ValueError:
        logical_file = path.name
    if chain_origin is not None:
        logical_file = f"{logical_file}\0chain={chain_origin}"
    # RCSBDataset uses torch.randint only to choose the valid crop start.  A
    # target-derived seed makes that crop invariant to rank/world-size and
    # index-range sharding while restoring the caller's RNG state afterward.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_crop_seed(crop_seed, logical_file))
        example = dataset[index]
    return path, example


def _dataset_target(dataset: RCSBDataset, index: int) -> tuple[Path, int | None]:
    if getattr(dataset, "extract_monomer_chains", False) and dataset.chain_index is not None:
        file_index, origin, _ = dataset.chain_index[index]
        return Path(dataset.files[file_index]), int(origin)
    return Path(dataset.files[index]), None


def _full_single_chain_sequence(
    path: Path, min_length: int, chain_origin: int | None = None
) -> str:
    """Return the uncropped canonical sequence used as the leakage identity."""
    with np.load(path) as data:
        residues = data["residues"]
        sequences: list[str] = []
        origin = -1
        for chain in data["chains"]:
            if int(chain["mol_type"]) != RCSBDataset.MOL_TYPE_PROTEIN:
                continue
            origin += 1
            if chain_origin is not None and origin != chain_origin:
                continue
            start = int(chain["res_idx"])
            end = start + int(chain["res_num"])
            names = [
                str(residue["name"])
                for residue in residues[start:end]
                if bool(residue["is_standard"]) and str(residue["name"]) in AA_3TO1
            ]
            if len(names) >= min_length:
                sequences.append("".join(AA_3TO1[name] for name in names))
    if len(sequences) != 1:
        raise ValueError(
            f"expected exactly one canonical protein chain in {path.name}"
            f" at origin={chain_origin}, found {len(sequences)}"
        )
    return sequences[0]


def _checkpoint_step(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict):
        return -1
    try:
        return int(payload.get("step", -1))
    except (TypeError, ValueError):
        return -1


def _load_folding_model(path: Path, device: torch.device, *, use_ema: bool):
    """Build the exact checkpoint architecture and load one strict weight set."""
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("args"), dict):
        raise ValueError("folding checkpoint must contain its training args")
    weight_key = "ema" if use_ema else "model"
    weights = payload.get(weight_key)
    if not isinstance(weights, dict):
        raise ValueError(f"folding checkpoint lacks {weight_key!r} weights")
    model = build_model(payload["args"], str(device))
    model.load_state_dict(weights, strict=True)
    del weights, payload
    gc.collect()
    return model.requires_grad_(False).eval()


def _flush_shard(
    out_dir: Path,
    rank: int,
    shard_index: int,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    path = out_dir / f"rollout-r{rank:05d}-{shard_index:06d}.pt"
    entry = write_rollout_shard(path, records)
    print(
        f"[shard] {path.name}: records={entry['records']} bytes={entry['bytes']}",
        flush=True,
    )
    return entry


def main() -> None:
    args = _parser().parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("pLDDT config schema_version must be 1")
    rollout = dict(config.get("rollout") or {})

    checkpoint = Path(_value(args, rollout, "checkpoint", required=True))
    data_dir = Path(_value(args, rollout, "data-dir", required=True))
    esm_dir = Path(_value(args, rollout, "esm-dir", required=True))
    file_list_value = _value(args, rollout, "file-list")
    chain_list_value = _value(args, rollout, "chain-list")
    if (file_list_value is None) == (chain_list_value is None):
        raise ValueError("set exactly one of rollout.file_list or rollout.chain_list")
    target_list = Path(file_list_value if file_list_value is not None else chain_list_value)
    chain_mode = chain_list_value is not None
    out_dir = Path(_value(args, rollout, "out-dir", required=True))
    use_ema = bool(_value(args, rollout, "use-ema") if args.use_ema is None else args.use_ema)
    rank = int(args.rank if args.rank is not None else os.environ.get("RANK", 0))
    world_size = int(
        args.world_size if args.world_size is not None else os.environ.get("WORLD_SIZE", 1)
    )
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"invalid shard rank/world-size: {rank}/{world_size}")

    device_value = str(_value(args, rollout, "device") or "cuda")
    if device_value == "cuda" and world_size > 1:
        device_value = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
    device = torch.device(device_value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA rollout requested but torch.cuda.is_available() is false")
    if device.type == "cuda":
        torch.cuda.set_device(device if device.index is not None else 0)
    enable_cuda_perf_flags()

    max_length = int(rollout.get("max_length", 1024))
    min_length = int(rollout.get("min_length", 20))
    min_obs_ratio = float(rollout.get("min_obs_ratio", 0.5))
    crop_seed = int(rollout.get("crop_seed", 0))
    n_seeds_value = _value(args, rollout, "n-seeds")
    n_seeds = int(4 if n_seeds_value is None else n_seeds_value)
    seed_offset = int(rollout.get("seed_offset", 0))
    seed_batch_size_value = _value(args, rollout, "seed-batch-size")
    seed_batch_size = int(n_seeds if seed_batch_size_value is None else seed_batch_size_value)
    shard_records = int(rollout.get("shard_records", 128))
    length_bin = int(rollout.get("length_bin", 0))
    n_steps_value = _value(args, rollout, "n-steps")
    n_steps = int(50 if n_steps_value is None else n_steps_value)
    sampler_name = str(rollout.get("sampler", "sde"))
    sde_tau = float(rollout.get("sde_tau", 0.01))
    sde_eps = float(rollout.get("sde_eps", 0.01))
    sde_w_cutoff = float(rollout.get("sde_w_cutoff", 0.99))
    sde_log_timesteps = bool(rollout.get("sde_log_timesteps", True))
    filter_invalid_geometry = bool(rollout.get("filter_invalid_geometry", False))
    conditioning = dict(rollout.get("conditioning") or {})
    expected_conditioning = {
        "model": "biohub/ESMC-6B",
        "revision": ESMC_6B_REVISION,
        "embedding_dimensions": ESMC_6B_DIM,
    }
    if conditioning != expected_conditioning:
        raise ValueError(
            "rollout conditioning must match the pinned public ESMC-6B contract: "
            f"expected={expected_conditioning}, found={conditioning}"
        )
    expected_checkpoint_step = int(_value(args, rollout, "expected-checkpoint-step", required=True))
    expected_checkpoint_sha256_value = _value(
        args, rollout, "expected-checkpoint-sha256", required=False
    )
    expected_checkpoint_sha256 = (
        None
        if expected_checkpoint_sha256_value in (None, "")
        else str(expected_checkpoint_sha256_value)
    )
    if expected_checkpoint_sha256 is not None and (
        len(expected_checkpoint_sha256) != 64
        or any(value not in "0123456789abcdef" for value in expected_checkpoint_sha256)
    ):
        raise ValueError("expected_checkpoint_sha256 must be a lowercase SHA-256 digest")
    if n_seeds < 1 or seed_batch_size < 1 or shard_records < 1:
        raise ValueError("n_seeds, seed_batch_size, and shard_records must be positive")

    manifest_name = f"manifest-r{rank:05d}-of-{world_size:05d}.json"
    manifest_path = out_dir / manifest_name
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {manifest_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = RCSBDataset(
        str(data_dir),
        max_length=max_length,
        min_length=min_length,
        min_obs_ratio=min_obs_ratio,
        file_list=None if chain_mode else str(target_list),
        chain_list=str(target_list) if chain_mode else None,
        esm_dir=str(esm_dir),
        single_chain_only=not chain_mode,
        extract_monomer_chains=chain_mode,
    )
    start_index = int(args.start_index if args.start_index is not None else 0)
    end_index = int(args.end_index if args.end_index is not None else len(dataset))
    if not 0 <= start_index <= end_index <= len(dataset):
        raise ValueError(
            f"invalid index range [{start_index},{end_index}) for dataset size {len(dataset)}"
        )
    selected_indices = [
        index
        for index in range(start_index, end_index)
        if (index - start_index) % world_size == rank
    ]
    if not selected_indices:
        raise ValueError("this shard selects no dataset indices")

    print(
        f"[load] checkpoint={checkpoint.name} ema={use_ema} device={device} "
        f"targets={len(selected_indices)} rank={rank}/{world_size}",
        flush=True,
    )
    if args.checkpoint_sha256_file is None:
        checkpoint_hash = sha256_file(checkpoint)
    else:
        hash_fields = Path(args.checkpoint_sha256_file).read_text(encoding="utf-8").split()
        if not hash_fields:
            raise ValueError("checkpoint SHA-256 sidecar is empty")
        checkpoint_hash = hash_fields[0]
        if len(checkpoint_hash) != 64 or any(
            value not in "0123456789abcdef" for value in checkpoint_hash
        ):
            raise ValueError("checkpoint SHA-256 sidecar has an invalid digest")
    if expected_checkpoint_sha256 is not None and checkpoint_hash != expected_checkpoint_sha256:
        raise RuntimeError(
            "folding checkpoint SHA-256 mismatch: "
            f"expected {expected_checkpoint_sha256}, found {checkpoint_hash}"
        )
    checkpoint_step = _checkpoint_step(checkpoint)
    if checkpoint_step != expected_checkpoint_step:
        raise RuntimeError(
            "folding checkpoint step mismatch: "
            f"expected {expected_checkpoint_step}, found {checkpoint_step}"
        )
    model = _load_folding_model(checkpoint, device, use_ema=use_ema)

    buffer: list[dict[str, Any]] = []
    shard_entries: list[dict[str, Any]] = []
    shard_index = 0
    skipped = 0
    skipped_geometry = 0
    sequence_inventory: set[str] = set()
    seeds_all = list(range(seed_offset, seed_offset + n_seeds))
    for position, dataset_index in enumerate(selected_indices, start=1):
        path, example = _load_deterministic_example(
            dataset,
            dataset_index,
            crop_seed=crop_seed,
            data_dir=data_dir,
        )
        _, chain_origin = _dataset_target(dataset, dataset_index)
        if example is None:
            skipped += 1
            print(f"[skip] index={dataset_index} file={path.name}: filtered sample", flush=True)
            continue
        centered = center_and_scale(example)
        length = int(centered.seq_len)
        if centered.esm is None or tuple(centered.esm.shape) != (length, ESMC_6B_DIM):
            actual_shape = None if centered.esm is None else tuple(centered.esm.shape)
            raise ValueError(
                f"target {path.name} has incompatible ESMC features: "
                f"expected={(length, ESMC_6B_DIM)}, found={actual_shape}"
            )
        target_id = _portable_target_id(path, data_dir, chain_origin)
        sequence_hash = sequence_sha256(
            _full_single_chain_sequence(path, min_length, chain_origin)
        )
        true_ca_A = centered.coords[:, CA_ATOM_ID].float().cpu() * COORD_SCALE
        true_ca_mask = (
            centered.atom_mask[:, CA_ATOM_ID] & centered.observed_mask[:, CA_ATOM_ID]
        ).bool()

        target_latent_max_abs = 0.0
        for seed_start in range(0, len(seeds_all), seed_batch_size):
            seeds = seeds_all[seed_start : seed_start + seed_batch_size]
            static_batch = prepare_inference_batch(
                [centered] * len(seeds),
                device=device,
                length_bin=length_bin,
            )
            with torch.inference_mode():
                result = sample(
                    model,
                    static_batch,
                    seeds=seeds,
                    n_steps=n_steps,
                    method=sampler_name,
                    sde_tau=sde_tau,
                    sde_eps=sde_eps,
                    sde_w_cutoff=sde_w_cutoff,
                    sde_log_timesteps=sde_log_timesteps,
                    return_trunk_latent=True,
                )
            if result.trunk_latent is None:
                raise RuntimeError("sample did not return requested trunk_latent")
            pred_ca_A = result.final_ca[:, :length].float().cpu()
            latent = result.trunk_latent[:, :length].float().cpu()
            if not torch.isfinite(latent).all():
                raise RuntimeError(
                    f"non-finite trunk_latent from sampler for target={target_id!r}, seeds={seeds}"
                )
            target_latent_max_abs = max(target_latent_max_abs, float(latent.abs().max()))
            label_input_mask = result.residue_mask[:, :length].bool().cpu() & true_ca_mask
            expanded_true_ca_A = true_ca_A.unsqueeze(0).expand(len(seeds), -1, -1)
            scores, label_mask = hard_lddt_ca(
                pred_ca_A,
                expanded_true_ca_A,
                label_input_mask,
            )
            for row, seed in enumerate(seeds):
                median_ca_step, normal_ca_fraction = _polymer_geometry_quality(pred_ca_A[row])
                if filter_invalid_geometry and (
                    not 3.0 <= median_ca_step <= 4.6 or normal_ca_fraction < 0.8
                ):
                    skipped_geometry += 1
                    print(
                        f"[skip-geometry] target={target_id} seed={seed} "
                        f"median_ca_step_A={median_ca_step:.4f} "
                        f"normal_ca_fraction={normal_ca_fraction:.4f}",
                        flush=True,
                    )
                    continue
                sequence_inventory.add(sequence_hash)
                buffer.append(
                    {
                        "target_id": target_id,
                        "sequence_sha256": sequence_hash,
                        "seed": seed,
                        "trunk_latent": latent[row],
                        "pred_ca_A": pred_ca_A[row],
                        "true_ca_A": true_ca_A,
                        "plddt_target": scores[row],
                        "target_mask": label_mask[row],
                    }
                )
                if len(buffer) >= shard_records:
                    shard_entries.append(_flush_shard(out_dir, rank, shard_index, buffer))
                    buffer = []
                    shard_index += 1
        print(
            f"[target {position}/{len(selected_indices)}] {target_id} L={length} "
            f"seeds={len(seeds_all)} latent_max_abs={target_latent_max_abs:.6g}",
            flush=True,
        )

    if buffer:
        shard_entries.append(_flush_shard(out_dir, rank, shard_index, buffer))
    if not shard_entries:
        raise RuntimeError("no rollout records were generated")

    sampler_config = {
        "sampler": sampler_name,
        "n_steps": n_steps,
        "sde_tau": sde_tau,
        "sde_eps": sde_eps,
        "sde_w_cutoff": sde_w_cutoff,
        "sde_log_timesteps": sde_log_timesteps,
        "geometry_guidance": None,
        "filter_invalid_geometry": filter_invalid_geometry,
        "max_length": max_length,
        "min_length": min_length,
        "min_obs_ratio": min_obs_ratio,
        "crop_seed": crop_seed,
        "n_seeds": n_seeds,
        "seed_offset": seed_offset,
        "length_bin": length_bin,
        "label": "hard_lDDT-Ca",
        "coordinate_unit": "Angstrom",
        "expected_checkpoint_step": expected_checkpoint_step,
        "expected_checkpoint_sha256": expected_checkpoint_sha256,
    }
    provenance = {
        "checkpoint": {
            "basename": checkpoint.name,
            "sha256": checkpoint_hash,
            "step": checkpoint_step,
            "weights": "ema" if use_ema else "model",
        },
        "config": sampler_config,
        "conditioning": expected_conditioning,
        "split": {
            # Retain the established field names so old and chain-aware
            # manifests share the confidence trainer's provenance contract.
            "file_list_basename": target_list.name,
            "file_list_sha256": sha256_file(target_list),
            "target_mode": "chain" if chain_mode else "single_chain_entry",
        },
    }
    selection = {
        "rank": rank,
        "world_size": world_size,
        "start_index": start_index,
        "end_index": end_index,
        "selected_indices": len(selected_indices),
        "skipped_filtered": skipped,
        "skipped_geometry": skipped_geometry,
    }
    write_rollout_manifest(
        manifest_path,
        shards=shard_entries,
        provenance=provenance,
        selection=selection,
        sequence_sha256s=sorted(sequence_inventory),
    )
    print(
        f"[done] manifest={manifest_path.name} shards={len(shard_entries)} "
        f"records={sum(entry['records'] for entry in shard_entries)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
