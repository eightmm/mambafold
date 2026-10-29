#!/usr/bin/env python3
"""Isolated warm-start pilot; never resumes or modifies the production run.

The fixed post-cutoff panel is development data with known training homologs.
Training uses admitted pre-cutoff chains only. Report all predefined cases,
including negative results. K=2 is the training budget; K=1/3 are evaluations.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from benchmarks.compare_latent_loops import write_json  # noqa: E402
from benchmarks.probe_latent_contribution import load_batch  # noqa: E402
from mambafold.data.collate import ProteinCollator  # noqa: E402
from mambafold.data.dataset import RCSBDataset  # noqa: E402
from mambafold.train.distributed import enable_cuda_perf_flags  # noqa: E402
from mambafold.train.engine import allatom_forward_and_loss, allatom_loss_surface  # noqa: E402
from mambafold.train.optim import build_optimizer  # noqa: E402
from mambafold.train.trainer import (  # noqa: E402
    build_model,
    initialize_refiner_from_checkpoint,
    seed_all,
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def rms(x):
    return x.detach().double().square().mean().sqrt().item()


@contextmanager
def scales(model):
    rows = []
    handles = []
    for name, module in model.named_modules():
        if name.startswith(("latent_residual", "residue_trunk")) and (
            name.endswith(("mamba_f", "mamba_b")) or name.endswith("stack")
        ):

            def record(module, args, out, name=name):
                rows.append(
                    dict(
                        module=name,
                        rms=rms(out),
                        maximum=out.detach().abs().max().item(),
                        finite=bool(torch.isfinite(out).all()),
                    )
                )

            handles.append(module.register_forward_hook(record))
    try:
        yield rows
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def route(model, mode):
    branch = model.latent_residual
    loops = branch.n_loops
    try:
        handles = []
        if mode == "zero_latent":
            handles.append(
                branch.stack.register_forward_hook(
                    lambda module, inputs, output: torch.zeros_like(output)
                )
            )
        elif mode == "reset_history":
            anchor = []
            refine_once = branch.refine_once

            def reset(x, mask, condition):
                if not anchor:
                    anchor.append(x)
                return refine_once(anchor[0], mask, condition)

            branch.refine_once = reset
        elif mode == "bypass":
            branch.forward = lambda x, mask, temb, **kw: x
        else:
            branch.n_loops = int(mode[1:])
        yield
    finally:
        branch.n_loops = loops
        for handle in handles:
            handle.remove()
        if mode == "bypass":
            del branch.forward
        if mode == "reset_history":
            del branch.refine_once


def evaluate(model, args, label):
    model.eval()
    panel = json.loads((args.panel / "panel.json").read_text())
    rows = []
    trace = []
    modes = ("K2", "K1", "K3", "bypass")
    if model.residue_refiner:
        modes += ("zero_latent", "reset_history")
    for index, target in enumerate(panel["targets"]):
        for t in (0.1, 0.5, 0.9):
            batch = load_batch(target, args.panel, t)
            outputs = {}
            for mode in modes:
                with (
                    route(model, mode),
                    torch.no_grad(),
                    torch.autocast("cuda", dtype=torch.bfloat16),
                ):
                    if index == 0 and t == 0.5 and mode == "K2":
                        with scales(model) as trace:
                            output = model(batch)
                    else:
                        output = model(batch)
                _, metrics = allatom_loss_surface(output, batch, w_lddt_atom=1)
                outputs[mode] = output["v_atom"].float()
                valid = batch.valid_mask
                diff = (outputs[mode] - outputs["K2"])[valid]
                rows.append(
                    dict(
                        target=target["target"],
                        t=t,
                        route=mode,
                        **metrics,
                        velocity_relative_rms=rms(diff) / max(rms(outputs["K2"][valid]), 1e-12),
                        recon_delta_A=rms(diff) * (1 - t) * 10,
                    )
                )
    summary = {
        mode: {
            key: statistics.mean(r[key] for r in rows if r["route"] == mode)
            for key in ("fm_atom", "lddt_atom", "velocity_relative_rms", "recon_delta_A")
        }
        for mode in modes
    }
    result = dict(label=label, rows=rows, summary=summary, scales=trace)
    write_json(args.out / f"{label}.json", result)
    print(json.dumps(dict(stage="evaluation", label=label, summary=summary)), flush=True)
    return result


def training_examples(args):
    admitted = set(Path("data/splits/admitted_all.txt").read_text().splitlines())
    candidates = []
    for line in Path("data/splits/plddt_rcsb_chain_train.tsv").read_text().splitlines():
        filename, origin, length = line.split("\t")
        if filename in admitted and 64 <= int(length) <= 512:
            candidates.append(line)
    candidates.sort(key=lambda line: hashlib.sha256(f"20261018:{line}".encode()).digest())
    shortlist = args.out / "training-candidates.tsv"
    shortlist.write_text("\n".join(candidates[:1024]) + "\n")
    dataset = RCSBDataset(
        "data/rcsb_train", esm_dir="data/rcsb_esmc6b", chain_list=str(shortlist), max_length=256
    )
    examples, selected, sequences = [], [], set()
    from mambafold.data.constants import AA_3TO1, ID_TO_AA

    panel = json.loads((args.panel / "panel.json").read_text())
    dev_sequences = {t["sequence"] for t in panel["targets"]}
    for index, line in enumerate(candidates[:1024]):
        example = dataset[index]
        if example is None or example.esm is None:
            continue
        sequence = "".join(AA_3TO1.get(ID_TO_AA[int(x)], "X") for x in example.res_type)
        if sequence in sequences or sequence in dev_sequences:
            continue
        sequences.add(sequence)
        examples.append(example)
        selected.append(line)
        if len(examples) == 128:
            break
    if len(examples) != 128:
        raise RuntimeError(f"expected 128 usable training chains, got {len(examples)}")
    (args.out / "training.tsv").write_text("\n".join(selected) + "\n")
    return examples


def migrate(model, old, source_config, target_config):
    """Admit an identity-initialized correction, preserving every backbone tensor."""
    from mambafold.train.trainer import initialize_residue_refiner

    return initialize_residue_refiner(model, old, source_config, target_config)


def train(args):
    enable_cuda_perf_flags()
    seed_all(20261018)
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = dict(yaml.safe_load((ROOT / "configs/residue_latent_refiner.yaml").read_text()),
               latent_loop_sampling=False, atom_recurrence=False)
    state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    provenance = dict(
        checkpoint_sha256=digest(args.checkpoint),
        checkpoint_step=state["step"],
        steps=args.steps,
        config=cfg,
        training_seed=20261018,
        development_panel_sha256=digest(args.panel / "panel.json"),
        scope="development-only; known homologs; no generalization claim",
        weight_migration="weights only; fresh optimizer; no exact resume",
        device=torch.cuda.get_device_name(0),
    )
    write_json(args.out / "protocol.json", provenance)
    model = build_model(state["args"], "cuda")
    model.load_state_dict(state["ema"], strict=True)
    evaluate(model, args, "legacy")
    del model
    torch.cuda.empty_cache()
    model = build_model(cfg, "cuda")
    write_json(
        args.out / "migration.json", initialize_refiner_from_checkpoint(model, args.checkpoint, cfg)
    )
    del state
    evaluate(model, args, "initial")
    examples = training_examples(args)
    collator = ProteinCollator(
        augment=True,
        copies_per_protein=2,
        max_length=256,
        length_bin=64,
        t_schedule="logit_normal",
        t_uniform_weight=0.1,
    )
    optimizer = build_optimizer(model, cfg)
    target_lrs = [g["lr"] for g in optimizer.param_groups]
    kwargs = {k: cfg[k] for k in inspect.signature(allatom_forward_and_loss).parameters if k in cfg}
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, args.steps + 1):
        seed_all(20261018 + step)
        example = examples[(step - 1) % len(examples)]
        batch = collator([example]).to("cuda")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for group, lr in zip(optimizer.param_groups, target_lrs):
            group["lr"] = lr * min(1.0, step / 50)
        torch.cuda.synchronize()
        started = time.perf_counter()
        loss, metrics = allatom_forward_and_loss(model, batch, **kwargs)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        skip = not bool(torch.isfinite(grad)) or grad.item() > 1e4 or not bool(torch.isfinite(loss))
        if not skip:
            optimizer.step()
        torch.cuda.synchronize()
        row = dict(
            step=step,
            **metrics,
            grad_norm=grad.item(),
            skipped=skip,
            seconds=time.perf_counter() - started,
            peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
        )
        rows.append(row)
        if step % 20 == 0:
            write_json(args.out / "training.json", rows)
            print(json.dumps(dict(stage="training", **row)), flush=True)
    write_json(args.out / "training.json", rows)
    torch.save(
        dict(model=model.state_dict(), args=cfg, step=args.steps, pilot_provenance=provenance),
        args.out / "trained.pt",
    )
    evaluate(model, args, "trained")
    write_json(
        args.out / "training_complete.json",
        dict(
            completed_steps=args.steps,
            skipped_steps=sum(r["skipped"] for r in rows),
            mean_seconds=statistics.mean(r["seconds"] for r in rows[10:]),
            peak_allocated_GiB=max(r["peak_allocated_GiB"] for r in rows),
            trained_sha256=digest(args.out / "trained.pt"),
        ),
    )


def infer(args):
    from mambafold.sampling import prepare_inference_batch, sample
    from mambafold.structure_io import write_pdb
    from scripts.rollout import benchmark_seeds, example_from_sequence

    enable_cuda_perf_flags()
    state = torch.load(args.out / "trained.pt", map_location="cpu", mmap=True, weights_only=False)
    model = build_model(state["args"], "cuda").eval()
    model.load_state_dict(state["model"], strict=True)
    del state
    panel = json.loads((args.panel / "panel.json").read_text())
    selected = [next(t for t in panel["targets"] if t["bin_index"] == i) for i in range(4)]
    rows = []
    # Frozen target selection and protocol precede the first sampled score.
    write_json(
        args.out / "sampling_protocol.json",
        dict(
            targets=selected,
            n_steps=500,
            method="sde",
            sde_tau=0.01,
            seed=0,
            checkpoint_sha256=digest(args.out / "trained.pt"),
            parent_panel_sha256=digest(args.panel / "panel.json"),
            modes=["K1", "K2", "K3"],
        ),
    )
    for target in selected:
        with np.load(args.panel / "inputs" / f"{target['target']}.npz") as packed:
            example = example_from_sequence(
                target["sequence"], torch.from_numpy(packed["esm"].copy())
            )
        assert torch.count_nonzero(example.coords) == 0
        batch = prepare_inference_batch([example], "cuda", length_bin=64)
        seed = benchmark_seeds(0, [target["target"]])[0]
        for mode in ("K2", "K1", "K3"):
            with route(model, mode):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model(batch)
                    model(batch)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                result = sample(
                    model,
                    batch,
                    [seed],
                    n_steps=500,
                    method="sde",
                    sde_tau=0.01,
                    return_trunk_latent=False,
                )
                torch.cuda.synchronize()
                row = dict(
                    target=target["target"],
                    route=mode,
                    seconds=time.perf_counter() - started,
                    peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
                )
            pairs = args.out / mode / "pairs"
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
            row["prediction_sha256"] = digest(pairs / f"{name}_pred.pdb")
            rows.append(row)
            write_json(args.out / "sampling.json", rows)
            print(json.dumps(dict(stage="sampling", **row)), flush=True)
    for mode in ("K1", "K2", "K3"):
        selected_rows = [r for r in rows if r["route"] == mode]
        if len(selected_rows) != 4:
            raise RuntimeError("incomplete sampling coverage")
        write_json(args.out / mode / "pairs/manifest.json", dict(expected=4, rows=selected_rows))


def diagnose(args):
    """Current-source equivalence, causal interventions and trained gradients."""
    from mambafold.model.bimamba3 import mamba3_runtime_info
    from mambafold.model.fold.latent_mamba import LatentMambaResidual

    enable_cuda_perf_flags()
    seed_all(20261018)
    state = torch.load(args.out / "trained.pt", map_location="cpu", mmap=True, weights_only=False)
    model = build_model(state["args"], "cuda")
    model.load_state_dict(state["model"], strict=True)
    result = evaluate(model, args, "causal")
    frozen = json.loads((args.out / "trained.json").read_text())
    differences = {
        key: result["summary"]["K2"][key] - frozen["summary"]["K2"][key]
        for key in ("fm_atom", "lddt_atom")
    }
    panel = json.loads((args.panel / "panel.json").read_text())
    # Profile before any backward allocates full-model gradients. This process
    # has no pilot optimizer or retained training state_dict on the GPU.
    from mambafold.sampling import prepare_inference_batch, sample
    from scripts.rollout import benchmark_seeds, example_from_sequence

    target = max(panel["targets"], key=lambda item: item["length"])
    with np.load(args.panel / "inputs" / f"{target['target']}.npz") as packed:
        example = example_from_sequence(target["sequence"], torch.from_numpy(packed["esm"].copy()))
    sampling_batch = prepare_inference_batch([example], "cuda", length_bin=64)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(sampling_batch)
        model(sampling_batch)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    sampled = sample(
        model,
        sampling_batch,
        [benchmark_seeds(0, [target["target"]])[0]],
        n_steps=500,
        method="sde",
        sde_tau=0.01,
        return_trunk_latent=False,
    )
    torch.cuda.synchronize()
    write_json(
        args.out / "standalone_profile.json",
        dict(
            target=target["target"],
            length=target["length"],
            K=2,
            n_steps=500,
            seconds=time.perf_counter() - started,
            peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
            scope="fresh process; one model; no training optimizer; warmed SDE500",
        ),
    )
    del sampled, sampling_batch
    gradient_rows = []
    for target in panel["targets"][:2]:
        batch = load_batch(target, args.panel, 0.5)
        model.zero_grad(set_to_none=True)
        loss, metrics = allatom_forward_and_loss(model, batch)
        loss.backward()
        groups = {}
        for prefix in (
            "latent_residual.residue_stack.",
            "latent_residual.stack.",
            "latent_residual.compress_proj.",
            "latent_residual.output_proj.",
            "latent_residual.depth_embed.",
        ):
            parameters = [(n, p) for n, p in model.named_parameters() if n.startswith(prefix)]
            groups[prefix] = dict(
                tensors=len(parameters),
                connected=sum(p.grad is not None for _, p in parameters),
                finite=all(
                    p.grad is not None and bool(torch.isfinite(p.grad).all()) for _, p in parameters
                ),
                norm=sum(
                    p.grad.detach().double().square().sum().item()
                    for _, p in parameters
                    if p.grad is not None
                )
                ** 0.5,
            )
        gradient_rows.append(
            dict(target=target["target"], loss=float(loss.detach()), groups=groups)
        )
    write_json(args.out / "trained_gradients.json", gradient_rows)
    write_json(
        args.out / "current_source_equivalence.json",
        dict(
            summary_delta=differences,
            parameters=sum(p.numel() for p in model.parameters()),
            parameters_finite=all(bool(torch.isfinite(p).all()) for p in model.parameters()),
            runtime=mamba3_runtime_info(model),
            source_sha256={
                str(p.relative_to(ROOT)): digest(p) for p in (ROOT / "src/mambafold").rglob("*.py")
            },
        ),
    )
    del model, state
    torch.cuda.empty_cache()
    state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    model = build_model(state["args"], "cuda").eval()
    model.load_state_dict(state["ema"], strict=True)
    target = panel["targets"][0]
    batch = load_batch(target, args.panel, 0.5)
    traces = {}
    for label in ("legacy", "stabilized_same_weights"):
        if label != "legacy":
            cfg = state["args"]
            branch = (
                LatentMambaResidual(
                    cfg["d_res"],
                    cfg["latent_compression"],
                    cfg["d_latent"],
                    cfg["n_latent_layers"],
                    n_loops=2,
                    d_temb=cfg["d_temb"],
                    d_state=cfg["d_state"],
                    mimo_rank=cfg["mimo_rank"],
                    expand=cfg["expand"],
                    headdim=cfg["headdim"],
                    stable=True,
                )
                .cuda()
                .eval()
            )
            incompatible = branch.load_state_dict(model.latent_residual.state_dict(), strict=False)
            assert not incompatible.unexpected_keys
            assert all(k.endswith(".ssm.norm.weight") for k in incompatible.missing_keys)
            model.latent_residual = branch
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), scales(model) as traced:
            output = model(batch)
        _, metrics = allatom_loss_surface(output, batch)
        traces[label] = dict(scales=traced, metrics=metrics)
    write_json(args.out / "same_weight_stabilization.json", traces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "infer", "diagnose"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()
    if args.mode == "train" and (args.out / "protocol.json").exists():
        parser.error("refusing to overwrite an existing pilot; choose a new output directory")
    if not torch.cuda.is_available() or args.steps < 1:
        parser.error("positive steps and a CUDA allocation required")
    {"train": train, "infer": infer, "diagnose": diagnose}[args.mode](args)


if __name__ == "__main__":
    main()
