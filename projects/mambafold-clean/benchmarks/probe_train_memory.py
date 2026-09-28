#!/usr/bin/env python
"""Probe the training-step memory and speed of candidate model sizes.

The shared-memory probe (`probe_mamba3_dstate.py`) answers which
(d_state, mimo_rank) a GPU can launch at all. It says nothing about whether a
*model* of a given width and depth fits alongside its optimizer state and the
activations of a 1024-residue crop. That is the number the config decision
actually turns on, and it cannot be derived from parameter count: the trunk is
bidirectional, so every BiMamba3 layer stores two SSM passes, and the atom
levels carry an extra A=MAX_ATOMS_PER_RES axis.

So this runs the real thing — the real module, a real forward, the real
flow-matching + all-atom-lDDT objective, a real backward, and a real AdamW
step — on a synthetic batch of the right shape, and reports peak allocated
memory and milliseconds per step.

Batch sizes are tried in ascending order and the sweep for a (config, crop)
stops at the first failure, since larger batches cannot then succeed.

Usage:
    python benchmarks/probe_train_memory.py --out results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mambafold.data.constants import MAX_ATOMS_PER_RES, NUM_PAIR_TYPES  # noqa: E402
from mambafold.data.types import ProteinBatch  # noqa: E402
from mambafold.losses.lddt import soft_lddt_all_atom_loss  # noqa: E402
from mambafold.model.fold.all_atom import NUM_RES_TYPES  # noqa: E402
from mambafold.model.fold.atom_mamba import NUM_ATOM_TYPES  # noqa: E402
from mambafold.train.trainer import build_model  # noqa: E402


def device_report() -> dict:
    props = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    report = {
        "name": props.name,
        "capability": f"{capability[0]}.{capability[1]}",
        "total_memory_gib": round(props.total_memory / 1024**3, 2),
        "multi_processor_count": props.multi_processor_count,
    }
    for attr in ("shared_memory_per_block", "shared_memory_per_block_optin",
                 "shared_memory_per_multiprocessor"):
        if hasattr(props, attr):
            report[attr] = getattr(props, attr)
    return report


def synthetic_batch(batch: int, length: int, d_plm: int, device: str) -> ProteinBatch:
    """A full-density batch: every residue valid, every atom slot occupied.

    Deliberately the worst case. A real crop has padding and missing side-chain
    slots, so a config that fits here fits in training.
    """
    A = MAX_ATOMS_PER_RES
    shape_res = (batch, length)
    shape_atom = (batch, length, A)
    ones_res = torch.ones(shape_res, dtype=torch.bool, device=device)
    ones_atom = torch.ones(shape_atom, dtype=torch.bool, device=device)
    zeros_res = torch.zeros(shape_res, dtype=torch.long, device=device)
    arange = torch.arange(length, device=device).expand(batch, length).contiguous()
    x = torch.randn(batch, length, A, 3, device=device)
    return ProteinBatch(
        res_type=torch.randint(0, NUM_RES_TYPES, shape_res, device=device),
        res_seq_nums=arange,
        atom_type=torch.randint(0, NUM_ATOM_TYPES, shape_atom, device=device),
        pair_type=torch.randint(0, NUM_PAIR_TYPES, shape_atom, device=device),
        res_mask=ones_res,
        atom_mask=ones_atom,
        valid_mask=ones_atom,
        ca_mask=ones_res,
        chain_id=zeros_res,
        entity_id=zeros_res,
        sym_id=zeros_res,
        is_nterm=torch.zeros(shape_res, dtype=torch.bool, device=device),
        is_cterm=torch.zeros(shape_res, dtype=torch.bool, device=device),
        x_clean=x,
        x_t=torch.randn_like(x),
        eps=torch.randn_like(x),
        t=torch.rand(batch, 1, 1, 1, device=device),
        esm=torch.randn(batch, length, d_plm, device=device),
    )


def classify(error: str) -> str:
    lowered = error.lower()
    if "out of memory" in lowered:
        return "device_out_of_memory"
    if "shared memory" in lowered or "out of resource" in lowered:
        return "shared_memory_limit"
    if "chunk_size must be at least" in lowered:
        return "chunk_below_kernel_minimum"
    return "other"


def try_case(cfg: dict, crop: int, batch: int, d_plm: int, steps: int,
             loss_mode: str) -> dict:
    result = {"batch": batch, "loss_mode": loss_mode}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = optimizer = None
    try:
        # Built through `build_model` from the run config, not from a handful of
        # keyword arguments. The previous form named eight of them and inherited
        # the constructor defaults for the rest, so it silently measured
        # `atom_mixer="mamba"`, `d_temb=128` and `d_plm_proj=256` — none of
        # which is what the config trains. Every number this produced for the
        # current model was therefore wrong.
        model = build_model(cfg, "cuda")
        result["n_params"] = sum(parameter.numel() for parameter in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        data = synthetic_batch(batch, crop, d_plm, "cuda")
        # Flow-matching target for the straight interpolant.
        target = data.x_clean - data.eps

        elapsed = []
        for step in range(steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(data)
            v = out["v_atom"].float()
            loss = torch.nn.functional.mse_loss(v, target)
            if loss_mode != "fm":
                # x0 estimate along the straight path, so the lDDT term sees a
                # structure rather than a velocity.
                x0 = data.x_t + (1.0 - data.t) * v
                loss = loss + soft_lddt_all_atom_loss(
                    x0,
                    data.x_clean,
                    data.valid_mask,
                )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if step:  # first step pays allocator and autotune costs
                elapsed.append(time.perf_counter() - t0)

        result["status"] = "ok"
        result["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        result["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        result["ms_per_step"] = round(1000 * sum(elapsed) / len(elapsed), 1) if elapsed else None
        result["residues_per_s"] = (
            round(batch * crop / (sum(elapsed) / len(elapsed))) if elapsed else None
        )
    except Exception as exc:  # noqa: BLE001
        message = f"{type(exc).__name__}: {exc}"
        result["status"] = "fail"
        result["failure"] = classify(message)
        result["error"] = message.replace("\n", " ")[:300]
    finally:
        del model, optimizer
        torch.cuda.empty_cache()
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--d_plm", type=int, default=2560, help="ESMC-6B width")
    ap.add_argument("--crops", default="256,512,1024")
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument(
        "--loss_mode",
        choices=("fm", "lddt_exact"),
        default="lddt_exact",
        help="Isolate flow-matching MSE alone or include exact all-atom lDDT.",
    )
    ap.add_argument("--d_state", type=int, default=64)
    ap.add_argument("--mimo_rank", type=int, default=4)
    ap.add_argument("--expand", type=int, default=2)
    ap.add_argument("--headdim", type=int, default=64)
    ap.add_argument(
        "--config",
        default="configs/run_a_mamba.yaml",
        help="Run config to measure. The model is built from this exactly as "
             "training builds it, so the numbers apply to the model that trains.",
    )
    ap.add_argument(
        "--override",
        default="",
        help="Optional comma-separated key=value overrides applied to the "
             "config, for sweeping one setting (e.g. 'd_state=96,mimo_rank=4').",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 2

    crops = [int(c) for c in args.crops.split(",")]
    batches = [int(b) for b in args.batches.split(",")]
    with open(args.config) as fh:
        base = yaml.safe_load(fh)
    for item in (o for o in args.override.split(",") if o.strip()):
        key, _, raw = item.partition("=")
        base[key.strip()] = yaml.safe_load(raw)
    configs = [base]
    args.d_plm = int(base.get("d_plm", args.d_plm))

    report = {
        "device": device_report(),
        "settings": {
            "d_plm": args.d_plm, "dtype": "bfloat16", "optimizer": "adamw",
            "loss_mode": args.loss_mode, "config": args.config, "override": args.override,
            "atoms_per_residue": MAX_ATOMS_PER_RES, "steps": args.steps,
        },
        "cases": [],
    }
    print(json.dumps(report["device"], indent=2), flush=True)

    for cfg in configs:
        params = None
        for crop in crops:
            for batch in batches:
                case = try_case(cfg, crop, batch, args.d_plm, args.steps, args.loss_mode)
                case.update(cfg)
                case["crop"] = crop
                report["cases"].append(case)
                print(
                    f"d_res={cfg['d_res']} n_trunk={cfg['n_trunk']} "
                    f"d_atom={cfg['d_atom']} crop={crop} batch={batch} "
                    f"params={case.get('n_params', 'unknown')} -> "
                    f"{case['status']} {case.get('peak_alloc_gib', case.get('failure', ''))} "
                    f"{case.get('ms_per_step', '')}",
                    flush=True,
                )
                if case["status"] != "ok":
                    break  # larger batches cannot fit either
        del params

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
