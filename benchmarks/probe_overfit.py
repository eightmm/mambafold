#!/usr/bin/env python
"""Overfit a handful of real chains and check the model can actually learn.

This is the cheapest end-to-end test that exists: it drives real npz records
through the real dataset, collator, model, loss, backward and optimizer, and
asks one question — can the architecture memorise a few structures?

A model that cannot overfit four proteins in a few hundred steps has a defect
that no amount of training will fix, and the defect classes this catches are
exactly the ones that produce a plausible-looking loss curve instead of an
exception: a mask that leaks padding into valid positions, a conditioning
signal that never reaches a block, a zero-initialised path that stays zero, an
lDDT term whose gradient does not reach the coordinates.

It is not a quality measurement and says nothing about generalisation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mambafold.data.collate import ProteinCollator  # noqa: E402
from mambafold.data.dataset import RCSBDataset  # noqa: E402
from mambafold.train.engine import allatom_forward_and_loss  # noqa: E402
from mambafold.train.trainer import build_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run_a_mamba.yaml")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--esm_dir", default=None)
    ap.add_argument("--file_list", default=None,
                    help="Restrict the corpus so the chain index is built over a "
                         "handful of files instead of the whole 158k-entry set, "
                         "which takes about an hour and a half.")
    ap.add_argument("--n_proteins", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--copies", type=int, default=4)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 2

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    cfg["max_length"] = args.max_length
    cfg["use_plm"] = args.esm_dir is not None

    ds = RCSBDataset(
        data_dir=args.data_dir,
        max_length=args.max_length,
        esm_dir=args.esm_dir,
        file_list=args.file_list,
        extract_monomer_chains=True,
        chain_index_workers=4,
    )
    examples = []
    for i in range(len(ds)):
        ex = ds[i]
        if ex is None or ex.seq_len < 60:
            continue
        if cfg["use_plm"] and ex.esm is None:
            continue
        examples.append(ex)
        if len(examples) >= args.n_proteins:
            break
    if len(examples) < args.n_proteins:
        print(f"only {len(examples)} usable chains found", file=sys.stderr)
        if not examples:
            return 2

    lengths = [int(e.seq_len) for e in examples]
    print(f"[overfit] {len(examples)} chains, lengths {lengths}, "
          f"copies={args.copies}, plm={cfg['use_plm']}", flush=True)

    model = build_model(cfg, "cuda")
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    collator = ProteinCollator(
        augment=True,
        copies_per_protein=args.copies,
        t_schedule=cfg.get("t_schedule", "logit_normal"),
        t_uniform_weight=cfg.get("t_uniform_weight", 0.02),
        max_length=args.max_length,
        length_bin=cfg.get("length_bin", 64),
    )
    print(f"[overfit] {n_params:,} params", flush=True)

    history = []
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        ex = examples[step % len(examples)]
        batch = collator([ex])
        if batch is None:
            continue
        batch = batch.to(torch.device("cuda"))
        loss, metrics = allatom_forward_and_loss(
            model,
            batch,
            alpha_mode=cfg.get("alpha_mode", "const"),
            use_rigid_align=cfg.get("use_rigid_align", True),
            use_amp=True,
            w_fm=cfg.get("w_fm", 1.0),
            w_lddt_atom=cfg.get("w_lddt_atom", 1.0),
            lddt_cutoff_A=cfg.get("lddt_cutoff_A", 15.0),
            lddt_pair_chunk_size=cfg.get("lddt_pair_chunk_size", 512),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.get("grad_clip", 2.0)
        ).item()
        optimizer.step()
        row = {"step": step, "grad_norm": gnorm, **metrics}
        history.append(row)
        if step % args.log_every == 0 or step == 1:
            print(f"  step {step:>5} loss={metrics['loss']:.4f} "
                  f"fm={metrics['fm_atom']:.4f} lddt={metrics['lddt_atom']:.4f} "
                  f"t={metrics['t_mean']:.3f} gnorm={gnorm:.2f} "
                  f"vram={torch.cuda.max_memory_allocated()/1024**3:.2f}GB",
                  flush=True)

    def window(key: str, n: int = 25) -> float:
        vals = [h[key] for h in history[:n]] if n > 0 else []
        return sum(vals) / max(1, len(vals))

    first = {k: window(k) for k in ("loss", "fm_atom", "lddt_atom")}
    last = {k: sum(h[k] for h in history[-25:]) / max(1, len(history[-25:]))
            for k in ("loss", "fm_atom", "lddt_atom")}
    verdict = {
        "learned": last["loss"] < first["loss"] * 0.9,
        "lddt_improved": last["lddt_atom"] < first["lddt_atom"] - 0.02,
        "finite": all(h["loss"] == h["loss"] for h in history),
    }
    print(f"\n[overfit] first-25 mean {first}")
    print(f"[overfit] last-25  mean {last}")
    print(f"[overfit] verdict  {verdict}")
    print(f"[overfit] {args.steps} steps in {time.perf_counter()-t0:.0f}s "
          f"({(time.perf_counter()-t0)/max(1,args.steps)*1000:.0f} ms/step)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"lengths": lengths, "n_params": n_params, "copies": args.copies,
             "first25": first, "last25": last, "verdict": verdict,
             "history": history}, indent=2))
    return 0 if verdict["learned"] and verdict["finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
