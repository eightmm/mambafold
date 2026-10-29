#!/usr/bin/env python3
"""Audit full-backbone fixed/sampled refinement with the unmodified trainer.

Training requires an uninterrupted, single-GPU allocation with worker-backed
loading. Worker RNG is isolated from model depth draws; all delivered CPU
batch fields are hashed to verify paired crops, conditioning and corruption.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
from collections import Counter
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def tensor_digest(items):
    digest = hashlib.sha256()
    for name, value in items:
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(
            value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def batch_digest(batch):
    return tensor_digest(
        (field.name, value)
        for field in fields(batch)
        if isinstance(value := getattr(batch, field.name), torch.Tensor)
    )


def transplant_component(weights, reference, component):
    """Cross-swap EMA components for diagnosis without changing the forward path."""
    if component not in ("backbone", "refiner"):
        raise ValueError("component must be backbone or refiner")
    if weights.keys() != reference.keys() or any(
        value.shape != reference[name].shape or value.dtype != reference[name].dtype
        for name, value in weights.items()
    ):
        raise ValueError("component transplant requires identical complete model layouts")
    selected = {
        name for name in weights
        if name.startswith("latent_residual.") == (component == "refiner")
    }
    if not selected or len(selected) == len(weights):
        raise ValueError("component transplant requires both backbone and refiner")
    return {name: reference[name] if name in selected else value
            for name, value in weights.items()}


def train(args):
    from mambafold.data.types import ProteinBatch
    from mambafold.train.ema import EMA
    from mambafold.train.trainer import capture_rng_state, restore_rng_state

    spec = importlib.util.spec_from_file_location(
        "full_refinement_train", ROOT / "scripts/train.py"
    )
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    original_loader = trainer.build_dataloaders
    original_optimizer = trainer.build_optimizer
    original_prewarm = trainer.prewarm_sequence_kernels
    original_forward = trainer.allatom_forward_and_loss
    original_log = trainer.log_metrics
    original_clip = torch.nn.utils.clip_grad_norm_
    original_ema = EMA.update
    original_to = ProteinBatch.to
    state = dict(
        updates=0,
        attempts=0,
        microbatches=0,
        finite_losses=0,
        depth_counts=Counter(),
        rows=[],
        stream=hashlib.sha256(),
    )
    model_ref = None
    initial_groups = {}
    audit_dir = args.out / "audit"
    audit_dir.mkdir(parents=True, exist_ok=False)

    def groups(model):
        prefixes = ("atom_encoder.", "residue_trunk.", "latent_residual.", "atom_decoder.")
        return {
            prefix: tensor_digest(
                (n, p) for n, p in model.named_parameters() if n.startswith(prefix)
            )
            for prefix in prefixes
        }

    def loader(cfg, is_dist):
        if is_dist or cfg.num_workers < 1:
            raise ValueError("paired audit requires one GPU and at least one loader worker")
        result, sampler, dataset = original_loader(cfg, is_dist)
        result.generator = torch.Generator().manual_seed(cfg.seed)
        write_json(
            audit_dir / "loader.json",
            dict(
                dataset_size=len(dataset),
                workers=cfg.num_workers,
                prefetch=cfg.prefetch_factor,
                sampler=type(sampler).__name__,
                seed=cfg.seed,
            ),
        )
        return result, sampler, dataset

    def optimizer(model, cfg):
        nonlocal model_ref, initial_groups
        model_ref = model
        if any(not p.requires_grad for p in model.parameters()):
            raise RuntimeError("full-backbone comparison forbids frozen parameters")
        if model.latent_residual.atom_refiner is not None:
            raise RuntimeError("atom recurrence must stay disabled")
        initial_groups = groups(model)
        write_json(
            audit_dir / "initial.json",
            dict(
                model_sha256=tensor_digest(model.state_dict().items()),
                group_sha256=initial_groups,
                parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                backbone_parameters=sum(
                    p.numel()
                    for n, p in model.named_parameters()
                    if not n.startswith("latent_residual.")
                ),
            ),
        )
        if not cfg["latent_loop_sampling"]:
            torch.save(dict(args=cfg, ema=model.state_dict(), step=0), args.out / "initial_ema.pt")
        return original_optimizer(model, cfg)

    def prewarm(*a, **kw):
        rng = capture_rng_state()
        try:
            return original_prewarm(*a, **kw)
        finally:
            restore_rng_state(rng)
            torch.cuda.reset_peak_memory_stats()

    def transfer(batch, *a, **kw):
        if batch.device.type == "cpu":
            digest = batch_digest(batch)
            state["stream"].update(bytes.fromhex(digest))
            state["microbatches"] += 1
            with (audit_dir / "inputs.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(micro=state["microbatches"], sha256=digest, length=batch.max_len)
                    )
                    + "\n"
                )
        return original_to(batch, *a, **kw)

    def forward(*a, **kw):
        try:
            loss, metrics = original_forward(*a, **kw)
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError("OOM invalidates the uninterrupted paired comparison") from exc
        if math.isfinite(float(metrics["loss"])):
            state["finite_losses"] += 1
        else:
            raise RuntimeError(
                "nonfinite loss; stop paired experiment instead of silently retrying"
            )
        return loss, metrics

    def clip(parameters, *a, **kw):
        result = original_clip(parameters, *a, **kw)
        state["attempts"] += 1
        value = float(result)
        if not math.isfinite(value) or value >= 1e4:
            raise RuntimeError(f"gradient guard would skip update: {value}")
        if state["attempts"] in (1, 1000, 2500, 5000):
            rows = {}
            for prefix in initial_groups:
                params = [p for n, p in model_ref.named_parameters() if n.startswith(prefix)]
                rows[prefix] = dict(
                    tensors=len(params),
                    connected=sum(p.grad is not None for p in params),
                    clipped_norm=math.sqrt(
                        sum(
                            float(p.grad.float().square().sum())
                            for p in params
                            if p.grad is not None
                        )
                    ),
                )
            write_json(audit_dir / f"gradients_{state['attempts']:07d}.json", rows)
        return result

    def update(ema, model):
        original_ema(ema, model)
        state["updates"] += 1
        depth = model.refinement_training_loops or model.latent_residual.n_loops
        state["depth_counts"][depth] += 1

    def log(step, total, avgs, *a, **kw):
        original_log(step, total, avgs, *a, **kw)
        row = dict(
            step=step,
            **avgs,
            applied_updates=state["updates"],
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            input_stream_sha256=state["stream"].hexdigest(),
        )
        state["rows"].append(row)
        with (audit_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    trainer.build_dataloaders = loader
    trainer.build_optimizer = optimizer
    trainer.prewarm_sequence_kernels = prewarm
    trainer.allatom_forward_and_loss = forward
    trainer.log_metrics = log
    ProteinBatch.to = transfer
    torch.nn.utils.clip_grad_norm_ = clip
    EMA.update = update
    sys.argv = [
        str(ROOT / "scripts/train.py"),
        "--config",
        str(args.config),
        "--out_dir",
        str(args.out),
        "--no_wandb",
    ]
    started = time.perf_counter()
    trainer.main()
    final_groups = groups(model_ref)
    if any(final_groups[key] == initial_groups[key] for key in initial_groups):
        raise RuntimeError("a whole model group remained unchanged")
    rows = state["rows"]
    measured = [row for row in rows if row["step"] > min(1000, rows[-1]["step"] // 2)]
    if state["microbatches"] != state["finite_losses"]:
        raise RuntimeError("not every delivered microbatch completed its loss")
    write_json(
        audit_dir / "complete.json",
        dict(
            updates=state["updates"],
            attempts=state["attempts"],
            microbatches=state["microbatches"],
            finite_losses=state["finite_losses"],
            skipped_updates=state["attempts"] - state["updates"],
            input_stream_sha256=state["stream"].hexdigest(),
            depth_counts=dict(state["depth_counts"]),
            group_sha256=final_groups,
            group_changed={key: True for key in initial_groups},
            end_to_end_seconds=time.perf_counter() - started,
            steady_mean_step_ms=statistics.mean(row["perf_train_step_ms_max"] for row in measured),
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            timing_scope="real loader/H2D, full loss/backward, clipping, optimizer and EMA; "
            "step timer excludes checkpoint writes; end-to-end includes startup and writes",
        ),
    )


def infer(args):
    from mambafold.sampling import prepare_inference_batch, sample
    from mambafold.structure_io import write_pdb
    from mambafold.train.distributed import enable_cuda_perf_flags
    from mambafold.train.trainer import build_model
    from scripts.rollout import benchmark_seeds, example_from_sequence

    enable_cuda_perf_flags()
    state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    restored_from = getattr(args, "restore_from", None)
    restored_component = getattr(args, "restore_component", None)
    if bool(restored_from) != bool(restored_component):
        raise ValueError("restore_from and restore_component must be specified together")
    transplant = None
    if restored_from:
        reference = torch.load(restored_from, map_location="cpu", mmap=True, weights_only=False)
        state["ema"] = transplant_component(state["ema"], reference["ema"], restored_component)
        transplant = dict(component=restored_component, source=str(restored_from),
                          source_step=reference["step"],
                          mixed_ema_sha256=tensor_digest(state["ema"].items()))
        del reference
    model = build_model(state["args"], "cuda").eval()
    model.load_state_dict(state["ema"], strict=True)
    checkpoint_step = state["step"]
    del state
    args.out.mkdir(parents=True, exist_ok=False)
    if transplant:
        write_json(args.out / "transplant.json", transplant)
    targets = json.loads((args.panel / "panel.json").read_text())["targets"]
    rows = []
    for index, target in enumerate(targets):
        with np.load(args.panel / "inputs" / f"{target['target']}.npz") as packed:
            example = example_from_sequence(
                target["sequence"], torch.from_numpy(packed["esm"].copy())
            )
        if torch.count_nonzero(example.coords):
            raise RuntimeError("inference must not receive reference coordinates")
        batch = prepare_inference_batch([example], "cuda", length_bin=64)
        depths = args.depths[index % len(args.depths) :] + args.depths[: index % len(args.depths)]
        for depth in depths:
            model.latent_residual.n_loops = depth
            calls = Counter()
            handles = [
                model.latent_residual.residue_stack.register_forward_hook(
                    lambda *unused: calls.update(residue=1)
                ),
                model.latent_residual.stack.register_forward_hook(
                    lambda *unused: calls.update(latent=1)
                ),
                model.atom_decoder.register_forward_hook(lambda *unused: calls.update(decoder=1)),
            ]
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model(batch)
            finally:
                for handle in handles:
                    handle.remove()
            if calls != Counter(residue=depth, latent=depth, decoder=1):
                raise RuntimeError(f"incorrect inference recurrence: {calls}")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(batch)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            result = sample(
                model,
                batch,
                [benchmark_seeds(0, [target["target"]])[0]],
                n_steps=500,
                method="sde",
                sde_tau=0.01,
                return_trunk_latent=False,
            )
            torch.cuda.synchronize()
            row = dict(
                target=target["target"],
                K=depth,
                checkpoint_step=checkpoint_step,
                seconds=time.perf_counter() - started,
                observed_calls=dict(calls),
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            )
            pairs = args.out / f"K{depth}" / "pairs"
            pairs.mkdir(parents=True, exist_ok=True)
            name = target["target"] + "_seed0"
            write_pdb(
                result.final_aa[0, : target["length"]].numpy(),
                example.res_type.numpy(),
                example.atom_mask.numpy(),
                np.zeros((target["length"], 14)),
                np.zeros(target["length"], dtype=np.int64),
                pairs / f"{name}_pred.pdb",
            )
            (pairs / f"{name}_gt.pdb").symlink_to(
                (args.panel / "references" / f"{target['target']}.pdb").resolve()
            )
            rows.append(row)
            write_json(args.out / "sampling.json", rows)
            print(json.dumps(row), flush=True)
    for depth in args.depths:
        selected = [row for row in rows if row["K"] == depth]
        if len(selected) != len(targets):
            raise RuntimeError("incomplete inference coverage")
        write_json(
            args.out / f"K{depth}" / "pairs/manifest.json",
            dict(
                expected=len(targets),
                checkpoint=str(args.checkpoint),
                weight_source="ema",
                sampler=dict(method="sde", n_steps=500, tau=0.01, seed=0),
                rows=selected,
            ),
        )


def check_pair(directory):
    initial = [
        json.loads((directory / arm / "audit/initial.json").read_text())
        for arm in ("fixed", "sampled")
    ]
    complete = [
        json.loads((directory / arm / "audit/complete.json").read_text())
        for arm in ("fixed", "sampled")
    ]
    if initial[0] != initial[1]:
        raise RuntimeError("initial full model/trainable sets differ between arms")
    for key in ("updates", "microbatches", "finite_losses", "input_stream_sha256"):
        if complete[0][key] != complete[1][key]:
            raise RuntimeError(f"paired training differs: {key}")
    if any(row["skipped_updates"] or row["attempts"] != row["updates"] for row in complete):
        raise RuntimeError("paired comparison contains skipped updates")
    if initial[0]["parameters"] != initial[0]["trainable_parameters"]:
        raise RuntimeError("backbone was frozen")
    write_json(directory / "paired_audit.json", dict(initial=initial[0], arms=complete))
    return complete


def report(args):
    complete = check_pair(args.out)
    rows = []
    scores = {}
    for label, depths in [
        ("baseline", [2]),
        ("fixed-2500", [2]),
        ("sampled-2500", [2]),
        ("fixed-5000", [1, 2, 3]),
        ("sampled-5000", [1, 2, 3]),
    ]:
        timings = json.loads((args.out / "evaluation" / label / "sampling.json").read_text())
        for depth in depths:
            score = json.loads(
                (args.out / "evaluation" / label / f"K{depth}/scores/summary.json").read_text()
            )
            if not score["evaluation_complete"] or score["successful_targets"] != 12:
                raise RuntimeError("incomplete standard scoring coverage")
            scores[label, depth] = {r["target_id"]: r["lddt"] for r in score["rows"]}
            selected = [r for r in timings if r["K"] == depth]
            rows.append(
                dict(
                    model=label,
                    K=depth,
                    coverage=score["successful_targets"],
                    lddt=score["metrics"]["lddt"]["mean"],
                    bb_lddt=score["metrics"]["bb_lddt"]["mean"],
                    tm_score=score["metrics"]["tm_score"]["mean"],
                    mean_seconds=statistics.mean(r["seconds"] for r in selected),
                    peak_allocated_gib=max(r["peak_allocated_gib"] for r in selected),
                )
            )
    fixed, sampled = scores["fixed-5000", 2], scores["sampled-5000", 2]
    if fixed.keys() != sampled.keys():
        raise RuntimeError("unpaired scoring targets")
    deltas = {key: sampled[key] - fixed[key] for key in fixed}
    entries = sorted({key.split("_")[0] for key in deltas})
    entry_deltas = {
        entry: statistics.mean(v for k, v in deltas.items() if k.split("_")[0] == entry)
        for entry in entries
    }
    result = dict(
        rows=rows,
        training=dict(zip(("fixed", "sampled"), complete)),
        primary="5000-step EMA at inference K2; sampled minus fixed",
        paired_delta_by_target=deltas,
        paired_delta_by_entry=entry_deltas,
        mean_chain_delta=statistics.mean(deltas.values()),
        mean_entry_delta=statistics.mean(entry_deltas.values()),
        limits="one training seed, 12 development chains with known training homologs; "
        "not a convergence/generalization claim; training loss differs with LiFT K",
    )
    write_json(args.out / "comparison.json", result)
    lines = [
        "# Full-backbone fixed versus sampled refinement",
        "",
        "All parameters trainable; atom recurrence off; matched initial weights and input stream.",
        "Final checkpoint step 5000, EMA. Standard OpenStructure on 12 development chains.",
        "",
        "| Model | K | Coverage | all-atom lDDT | backbone lDDT | Mean seconds | Peak GiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['K']} | {row['coverage']}/12 | "
            f"{row['lddt']:.5f} | {row['bb_lddt']:.5f} | "
            f"{row['mean_seconds']:.2f} | {row['peak_allocated_gib']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Sampled minus fixed K2: chain mean {result['mean_chain_delta']:+.5f}; "
            f"entry mean {result['mean_entry_delta']:+.5f}.",
            "",
            result["limits"],
            "",
        ]
    )
    (args.out / "comparison.md").write_text("\n".join(lines))
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    training = sub.add_parser("train")
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--out", type=Path, required=True)
    inference = sub.add_parser("infer")
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--panel", type=Path, required=True)
    inference.add_argument("--out", type=Path, required=True)
    inference.add_argument("--depths", type=int, nargs="+", choices=(1, 2, 3), required=True)
    inference.add_argument("--restore-from", type=Path)
    inference.add_argument("--restore-component", choices=("backbone", "refiner"))
    checking = sub.add_parser("check")
    checking.add_argument("--out", type=Path, required=True)
    reporting = sub.add_parser("report")
    reporting.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.stage in ("check", "report"):
        (check_pair(args.out) if args.stage == "check" else report(args))
        return
    if not torch.cuda.is_available():
        parser.error("a Slurm CUDA allocation is required")
    torch.set_num_threads(1)
    (train if args.stage == "train" else infer)(args)


if __name__ == "__main__":
    main()
