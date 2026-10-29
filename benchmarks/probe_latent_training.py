#!/usr/bin/env python
"""Short, paired GPU training-update probe on one admitted chain with cached PLM.

Times the production loss, accumulation, backward, clipping, optimizer and EMA.
Data loading/collation/transfers and cold-kernel warmup are outside the timer.
Each invocation measures one objective in its own process/allocator context.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mambafold.data.collate import ProteinCollator  # noqa: E402
from mambafold.data.dataset import RCSBDataset  # noqa: E402
from mambafold.model.bimamba3 import mamba3_runtime_info  # noqa: E402
from mambafold.train.distributed import enable_cuda_perf_flags  # noqa: E402
from mambafold.train.ema import EMA  # noqa: E402
from mambafold.train.engine import allatom_forward_and_loss  # noqa: E402
from mambafold.train.optim import build_optimizer, optimizer_summary  # noqa: E402
from mambafold.train.trainer import build_model, seed_all  # noqa: E402


def source_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src/mambafold").rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def training_update(model, optimizer, ema, batches, cfg: dict) -> dict:
    """Use the same accumulated-gradient update as scripts/train.py, without I/O."""
    kwargs = {
        key: cfg[key] for key in inspect.signature(allatom_forward_and_loss).parameters
        if key in cfg and key not in ("model", "batch")
    }
    optimizer.zero_grad(set_to_none=True)
    metrics = {}
    for batch in batches:
        loss, metrics = allatom_forward_and_loss(model, batch, **kwargs)
        if not torch.isfinite(loss).item():
            raise RuntimeError("nonfinite probe loss")
        (loss / len(batches)).backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]).item()
    if not (0 <= grad_norm < 1e4):
        raise RuntimeError(f"probe gradient would skip the production update: {grad_norm}")
    optimizer.step()
    ema.update(model)
    return {"loss": metrics["loss"], "grad_norm": grad_norm}


def measure(cfg: dict, chain_list: str, length: int, warmup: int, steps: int) -> dict:
    print(json.dumps({"stage": "prepare", "length": length,
                      "supervision": cfg["latent_supervision"]}), flush=True)
    seed_all(1729)
    source = cfg["train_sources"][0]
    dataset = RCSBDataset(
        data_dir=source["data_dir"], esm_dir=source["esm_dir"],
        chain_list=chain_list, max_length=length,
    )
    example = dataset[0]
    if example is None or example.seq_len != length or example.esm is None:
        raise RuntimeError("probe needs a usable admitted chain at least as long as the crop")
    copies = cfg["copies_per_protein"]
    accum = cfg["grad_accum_steps"]
    collator = ProteinCollator(
        augment=True, copies_per_protein=copies, max_length=length,
        length_bin=cfg["length_bin"], t_schedule=cfg["t_schedule"],
        t_uniform_weight=cfg["t_uniform_weight"],
    )
    # Fixed, independently corrupted microbatches are identical between arms.
    batches = [collator([example]).to(torch.device("cuda")) for _ in range(accum)]
    seed_all(2718)
    model = build_model(cfg, "cuda")
    ema = EMA(model, decay=cfg["ema_decay"])
    optimizer = build_optimizer(model, cfg)
    row = {
        "length": length, "copies": copies, "grad_accum": accum,
        "supervision": cfg["latent_supervision"],
        "compression": cfg["latent_compression"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "optimizer_groups": optimizer_summary(optimizer),
        "canonical_atoms_per_copy": int(batches[0].atom_mask[0].sum()),
        "resolved_atoms_per_copy": int(batches[0].valid_mask[0].sum()),
        "plm_rows": batches[0].esm.shape[0],
        "chunk_sizes": {
            "atom": model.atom_encoder.cross.stack.layers[0].mamba_f.chunk_size,
            "residue": model.residue_trunk.layers[0].mamba_f.chunk_size,
            "latent": model.latent_residual.stack.layers[0].mamba_f.chunk_size,
        },
    }
    print(json.dumps({"stage": "warmup", **row}), flush=True)
    started = time.perf_counter()
    for _ in range(warmup):
        training_update(model, optimizer, ema, batches, cfg)
    torch.cuda.synchronize()
    row["warmup_seconds"] = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    durations = []
    for _ in range(steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        metrics = training_update(model, optimizer, ema, batches, cfg)
        torch.cuda.synchronize()
        durations.append(1000 * (time.perf_counter() - started))
    median = statistics.median(durations)
    parameters_finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
    if not parameters_finite:
        raise RuntimeError("nonfinite model parameter after optimizer updates")
    row.update(
        status="ok", update_ms=durations, median_update_ms=median,
        mean_update_ms=statistics.mean(durations),
        min_update_ms=min(durations), max_update_ms=max(durations),
        residues_per_second=1000 * copies * accum * length / median,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
        parameters_finite=parameters_finite,
        final_metrics=metrics,
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/run_a_mamba.yaml")
    parser.add_argument("--chain-list", required=True)
    parser.add_argument("--supervision", choices=("final", "lift"), required=True)
    parser.add_argument("--optimizer", choices=("adamw", "muon"), default=None)
    parser.add_argument("--compression", type=int, choices=(4, 8), default=4)
    parser.add_argument("--lengths", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.warmup < 2 or args.steps < 1 or any(length < 1 for length in args.lengths):
        parser.error("require warmup>=2, steps>=1 and positive crop lengths")
    if not torch.cuda.is_available():
        parser.error("a CUDA GPU allocation is required")
    out = Path(args.out)
    if out.exists():
        parser.error("refusing to overwrite an existing probe report")
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.optimizer is not None:
        cfg["optimizer"] = args.optimizer
    cfg.update(latent_compression=args.compression, latent_loops=2, latent_position="post",
               latent_loop_mode="conditioned", latent_supervision=args.supervision,
               d_latent=256, n_latent_layers=2)
    enable_cuda_perf_flags()
    props = torch.cuda.get_device_properties(0)
    report = {
        "source_sha256": source_digest(), "config": cfg,
        "mamba3_runtime": mamba3_runtime_info(),
        "chain_list_sha256": hashlib.sha256(Path(args.chain_list).read_bytes()).hexdigest(),
        "device": props.name, "device_memory_gib": props.total_memory / 1024**3,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "dtype": "bf16 autocast / fp32 parameters", "warmup_updates": args.warmup,
        "optimizer": cfg.get("optimizer", "adamw"),
        "measured_updates": args.steps, "includes": [
            "production FM/lDDT loss and metric syncs", "backward", "gradient accumulation",
            "gradient clipping", "optimizer update", "EMA",
        ],
        "excludes": ["data loading/collation/transfer", "DDP", "cold kernel compilation"],
        "initialization": "paired random weights, no checkpoint or accuracy measurement",
        "cases": [],
    }
    for length in args.lengths:
        try:
            row = measure(cfg, args.chain_list, length, args.warmup, args.steps)
        except Exception as exc:
            row = {"length": length, "status": "fail", "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()}
        report["cases"].append(row)
        report["source_unchanged"] = source_digest() == report["source_sha256"]
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    return 0 if report["source_unchanged"] and all(
        row["status"] == "ok" for row in report["cases"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
