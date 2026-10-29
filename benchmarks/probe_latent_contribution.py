#!/usr/bin/env python3
"""Read-only checkpoint attribution; never steps an optimizer or saves model weights."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from benchmarks.compare_latent_loops import write_json  # noqa: E402


@contextmanager
def route(model, mode: str):
    """Temporarily remove one inference route without changing its weights."""
    branch = model.latent_residual
    original = branch.n_loops
    handles = []
    had_forward = "forward" in branch.__dict__
    forward = branch.forward
    try:
        branch.n_loops = int(mode[1:]) if mode.startswith("K") else 2
        if mode == "bypass":
            handles.append(branch.register_forward_hook(lambda module, args, out: args[0]))
        elif mode == "local_only":
            handles.append(
                branch.stack.register_forward_hook(lambda module, args, out: torch.zeros_like(out))
            )
        elif mode == "compression_readout":

            def prelude(*args, **kwargs):
                return forward(*args, **{**kwargs, "return_states": True})[0]

            branch.forward = prelude
        elif mode == "identity_stack":
            handles.append(branch.stack.register_forward_hook(lambda module, args, out: args[0]))
        elif mode == "reset_history":
            anchor = []
            handles.append(
                branch.compress_proj.register_forward_hook(
                    lambda module, inputs, output: anchor.append(output)
                )
            )

            def reset(module, inputs):
                z = anchor[-1]
                return (F.rms_norm(z, (z.shape[-1],), eps=1e-6) + z, *inputs[1:])

            handles.append(branch.stack.register_forward_pre_hook(reset))
        elif not mode.startswith("K"):
            raise ValueError(mode)
        yield
    finally:
        for handle in handles:
            handle.remove()
        branch.n_loops = original
        if had_forward:
            branch.forward = forward
        elif "forward" in branch.__dict__:
            del branch.forward


def rms(value, mask=None):
    value = value.detach().float()
    if mask is not None:
        value = value[mask.bool()]
    return float(value.square().mean().sqrt())


def relative(delta, reference, mask=None):
    return rms(delta, mask) / max(rms(reference, mask), 1e-12)


def gradients(model):
    groups = defaultdict(
        lambda: dict(
            parameters=0, grad_parameters=0, missing=0, zero=0, finite=True, w2=0.0, g2=0.0
        )
    )
    details = {}
    for name, parameter in model.named_parameters():
        key = (
            ".".join(name.split(".")[:2])
            if name.startswith("latent_residual.")
            else name.split(".")[0]
        )
        group = groups[key]
        group["parameters"] += parameter.numel()
        group["w2"] += float(parameter.detach().float().square().sum())
        if parameter.grad is None:
            group["missing"] += 1
            continue
        grad = parameter.grad.detach().float()
        group["grad_parameters"] += parameter.numel()
        norm = float(grad.square().sum())
        group["g2"] += norm
        group["zero"] += int(norm == 0)
        group["finite"] &= bool(torch.isfinite(grad).all())
        if name.startswith("latent_residual."):
            details[name] = dict(grad_rms=rms(grad), weight_rms=rms(parameter))
    for group in groups.values():
        group["grad_norm"] = group["g2"] ** 0.5
        group["grad_rms"] = (group["g2"] / max(group["grad_parameters"], 1)) ** 0.5
        group["grad_to_weight_norm"] = (group["g2"] / max(group["w2"], 1e-24)) ** 0.5
    return dict(groups=groups, latent_parameters=details)


def inspect_history(args):
    previous = {}
    rows = []
    for path in args.checkpoints:
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        row = dict(step=state["step"], weights={}, optimizer={})
        for kind in ("model", "ema"):
            current = {
                k: v.detach().float().clone()
                for k, v in state[kind].items()
                if k.startswith("latent_residual.")
            }
            row["weights"][kind] = {
                k: dict(
                    rms=rms(v),
                    nonzero=bool(v.count_nonzero()),
                    delta_previous_relative=(
                        relative(v - previous[kind][k], previous[kind][k])
                        if kind in previous
                        else None
                    ),
                )
                for k, v in current.items()
            }
            previous[kind] = current
        for kind, child in state["optimizer"]["children"].items():
            for group in child["param_groups"]:
                for name, index in zip(group["param_names"], group["params"], strict=True):
                    if name.startswith("latent_residual."):
                        tensors = child["state"].get(index, {})
                        row["optimizer"][name] = dict(
                            kind=kind,
                            state={
                                key: rms(value)
                                for key, value in tensors.items()
                                if torch.is_tensor(value)
                            },
                        )
        rows.append(row)
        print(f"checkpoint inspected: step={state['step']}", flush=True)
        del state
    write_json(args.out / "checkpoint_history.json", rows)


@lru_cache(maxsize=1)
def reference_dataset(directory):
    from mambafold.data.dataset import RCSBDataset

    return RCSBDataset(directory, max_length=1024)


def load_batch(target, panel_root, time):
    from mambafold.data.collate import ProteinCollator

    source = Path(target["source"])
    dataset = reference_dataset(str(source.parent))
    with np.load(source) as data:
        example = dataset._canonicalize(
            data, source, only_chain_origin=int(target["target"].rsplit("_ch", 1)[1])
        )
    assert example is not None and example.seq_len == target["length"]
    with np.load(panel_root / "inputs" / f"{target['target']}.npz") as data:
        example.esm = torch.from_numpy(data["esm"].copy())
    torch.manual_seed(20261018)
    batch = ProteinCollator(augment=False, length_bin=64)([example]).to("cuda")
    t = torch.full_like(batch.t, time)
    return replace(
        batch, t=t, x_t=(t * batch.x_clean + (1 - t) * batch.eps) * batch.atom_mask.unsqueeze(-1)
    )


def observe(model, batch, mode):
    saved = dict(states=[], stack_inputs=[], readouts=[])
    branch = model.latent_residual

    def store(name):
        def hook(module, inputs, output):
            if name in ("states", "readouts"):
                saved[name].append(output.detach())
            else:
                saved[name] = output.detach()

        return hook

    with route(model, mode):
        handles = [
            branch.compress_proj.register_forward_hook(store("initial")),
            branch.residue_proj.register_forward_hook(store("local")),
            branch.offset_embed.register_forward_hook(store("offset")),
            branch.stack.register_forward_hook(store("states")),
            branch.output_proj.register_forward_hook(store("readouts")),
            branch.register_forward_pre_hook(
                lambda module, inputs: saved.update(input=inputs[0].detach())
            ),
            branch.stack.register_forward_pre_hook(
                lambda module, inputs: saved["stack_inputs"].append(inputs[0].detach())
            ),
        ]
        try:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(batch)
        finally:
            for handle in handles:
                handle.remove()
    zmask = batch.res_mask.reshape(batch.batch_size, -1, branch.compression).any(-1)
    states = [saved["initial"], *saved["states"]]
    info = dict(
        branch_relative_update=relative(
            output["trunk_latent"] - saved["input"], saved["input"], batch.res_mask
        ),
        local_rms=rms(saved["local"], batch.res_mask),
        offset_rms=rms(saved["offset"]),
        initial_context_rms=rms(states[0], zmask),
        computed_final_context_rms=rms(states[-1], zmask),
        preadd_update_relative=relative(
            saved["readouts"][0] if mode == "compression_readout" else saved["readouts"][-1],
            saved["input"],
            batch.res_mask,
        ),
        residue_changed_fraction=float(
            (output["trunk_latent"] != saved["input"])[batch.res_mask].float().mean()
        ),
        between_loop_changes=[relative(b - a, a, zmask) for a, b in zip(states, states[1:])],
        stack_change=[
            relative(b - a, a, zmask) for a, b in zip(saved["stack_inputs"], saved["states"])
        ],
    )
    return output["v_atom"].detach().float(), info


def probe(args):
    from mambafold.train.engine import allatom_loss_surface
    from mambafold.train.trainer import build_model

    panel = json.loads((args.panel / "panel.json").read_text())
    model = None
    rows = []
    grad_rows = []
    for path in args.checkpoints[1:]:
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        if model is None:
            model = build_model(state["args"], "cuda")
        for kind in ("ema", "model"):
            model.load_state_dict(state[kind], strict=True)
            model.eval()
            targets = panel["targets"] if kind == "ema" else panel["targets"][:1]
            for target in targets:
                for time in (0.1, 0.5, 0.9):
                    batch = load_batch(target, args.panel, time)
                    outputs = {}
                    for mode in (
                        "bypass",
                        "local_only",
                        "compression_readout",
                        "identity_stack",
                        "reset_history",
                        "K1",
                        "K2",
                        "K3",
                        "K4",
                        "K8",
                    ):
                        velocity, info = observe(model, batch, mode)
                        outputs[mode] = velocity
                        row = dict(
                            step=state["step"],
                            weights=kind,
                            target=target["target"],
                            time=time,
                            mode=mode,
                            **info,
                        )
                        rows.append(row)
                    baseline = outputs["K2"]
                    for row in rows[-len(outputs) :]:
                        delta = outputs[row["mode"]] - baseline
                        row.update(
                            velocity_relative_to_K2=relative(delta, baseline, batch.valid_mask),
                            clean_prediction_delta_rms_A=(1 - time)
                            * 10
                            * rms(delta, batch.valid_mask),
                        )
                    del batch, outputs
            write_json(args.out / "attribution.json", rows)
            if kind == "model":
                for target in (panel["targets"][0], panel["targets"][3]):
                    for time in (0.1, 0.5, 0.9):
                        batch = load_batch(target, args.panel, time)
                        model.train()
                        model.zero_grad(set_to_none=True)
                        states = []

                        def retain(module, inputs, output):
                            output.retain_grad()
                            states.append(output)

                        handle = model.latent_residual.stack.register_forward_hook(retain)
                        torch.manual_seed(222)
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            output = model(batch)
                        handle.remove()
                        loss, metrics = allatom_loss_surface(output, batch)
                        loss.backward()
                        assert len(states) == 2
                        grad_rows.append(
                            dict(
                                step=state["step"],
                                target=target["target"],
                                time=time,
                                loss=float(loss.detach()),
                                metrics=metrics,
                                depths=output["latent_depths"].detach().cpu().tolist(),
                                loop_output_gradient_rms=[
                                    rms(v.grad) if v.grad is not None else None for v in states
                                ],
                                **gradients(model),
                            )
                        )
                        model.zero_grad(set_to_none=True)
                        del batch, output, loss, states
                write_json(args.out / "gradients.json", grad_rows)
        del state
        print(f"attribution and backward complete: step={path.name}", flush=True)


def rollout(args):
    from mambafold.sampling import prepare_inference_batch, sample
    from mambafold.structure_io import write_pdb
    from mambafold.train.trainer import build_model
    from scripts.rollout import benchmark_seeds, example_from_sequence

    panel = json.loads((args.panel / "panel.json").read_text())
    state = torch.load(args.panel / "ema.pt", map_location="cpu", mmap=True, weights_only=False)
    model = build_model(state["args"], "cuda")
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    del state
    selected = [next(t for t in panel["targets"] if t["bin_index"] == i) for i in range(4)]
    manifest = []
    for mode in ("bypass", "local_only", "compression_readout", "K2", "K8"):
        pairs = args.out / mode / "pairs"
        pairs.mkdir(parents=True, exist_ok=True)
        for target in selected:
            with np.load(args.panel / "inputs" / f"{target['target']}.npz") as data:
                example = example_from_sequence(
                    target["sequence"], torch.from_numpy(data["esm"].copy())
                )
            assert torch.count_nonzero(example.coords) == 0
            batch = prepare_inference_batch([example], "cuda", length_bin=64)
            seed = benchmark_seeds(0, [target["target"]])[0]
            with route(model, mode):
                result = sample(
                    model, batch, [seed], n_steps=500, method="sde", return_trunk_latent=False
                )
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
            manifest.append(dict(mode=mode, target=target["target"], seed=seed))
            write_json(args.out / "rollouts.json", manifest)
            print(f"rollout complete: {mode} {name}", flush=True)
        write_json(
            pairs / "manifest.json",
            dict(EMA_sha256=panel["ema_sha256"], mode=mode, targets=selected, seed=0, n_steps=500),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("history", "probe", "rollout"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    {"history": inspect_history, "probe": probe, "rollout": rollout}[args.mode](args)


if __name__ == "__main__":
    main()
