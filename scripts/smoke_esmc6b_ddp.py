#!/usr/bin/env python
"""Synthetic worst-case DDP memory smoke for the ESMC-6B training path."""

import argparse
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mambafold.data.types import ProteinBatch  # noqa: E402
from mambafold.train.distributed import distributed_max_int  # noqa: E402
from mambafold.train.ema import EMA  # noqa: E402
from mambafold.train.engine import allatom_forward_and_loss  # noqa: E402
from mambafold.train.trainer import build_model  # noqa: E402


def make_batch(batch_size: int, length: int, d_plm: int, device: torch.device):
    atoms = 15
    res_mask = torch.ones(batch_size, length, dtype=torch.bool, device=device)
    atom_mask = torch.ones(
        batch_size, length, atoms, dtype=torch.bool, device=device
    )
    zeros_res = torch.zeros(batch_size, length, dtype=torch.long, device=device)
    return ProteinBatch(
        res_type=torch.randint(0, 20, (batch_size, length), device=device),
        res_seq_nums=torch.arange(length, device=device)
        .expand(batch_size, length)
        .contiguous(),
        atom_type=torch.zeros(
            batch_size, length, atoms, dtype=torch.long, device=device
        ),
        pair_type=torch.zeros(
            batch_size, length, atoms, dtype=torch.long, device=device
        ),
        res_mask=res_mask,
        atom_mask=atom_mask,
        valid_mask=atom_mask.clone(),
        ca_mask=res_mask.clone(),
        chain_id=zeros_res.clone(),
        entity_id=zeros_res.clone(),
        sym_id=zeros_res.clone(),
        is_nterm=torch.zeros_like(res_mask),
        is_cterm=torch.zeros_like(res_mask),
        x_clean=torch.randn(batch_size, length, atoms, 3, device=device),
        x_t=torch.randn(batch_size, length, atoms, 3, device=device),
        eps=torch.randn(batch_size, length, atoms, 3, device=device),
        t=torch.rand(batch_size, 1, 1, 1, device=device),
        esm=torch.randn(
            batch_size, length, d_plm, device=device, dtype=torch.bfloat16
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/direct_allatom_puremamba_attn6_geo_adaln_sf360_esmc6b.yaml",
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument(
        "--rank-length-step",
        type=int,
        default=0,
        help="Subtract this many residues per rank before global shape sync.",
    )
    parser.add_argument("--grad-accum", type=int, default=2)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    torch.manual_seed(int(cfg["seed"]) + rank)
    torch.cuda.reset_peak_memory_stats(device)

    model = build_model(cfg, str(device))
    model = DDP(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        find_unused_parameters=bool(cfg["find_unused_parameters"]),
        gradient_as_bucket_view=True,
    )
    ema = EMA(model.module, decay=float(cfg["ema_decay"]))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["lr"]),
        weight_decay=1e-2,
        fused=True,
    )
    input_length = args.length - rank * args.rank_length_step
    if input_length < 1:
        raise ValueError(f"rank {rank} received invalid length {input_length}")
    batch = make_batch(args.batch_size, input_length, int(cfg["d_plm"]), device)
    global_length = distributed_max_int(batch.max_len, device)
    batch = batch.pad_to_length(global_length)

    started = time.time()
    for micro_step in range(args.grad_accum):
        sync_context = (
            model.no_sync()
            if micro_step < args.grad_accum - 1
            else nullcontext()
        )
        with sync_context:
            loss, _ = allatom_forward_and_loss(
                model,
                batch,
                alpha_mode=cfg["alpha_mode"],
                use_amp=True,
                **{
                    key: cfg[key]
                    for key in (
                        "w_fm",
                        "w_lddt_atom",
                        "w_lddt_ca",
                        "w_bond",
                        "w_clash",
                        "w_ca_clash",
                        "w_distogram",
                        "w_drmsd",
                        "w_contact",
                        "w_pcb",
                        "w_conf",
                        "w_ca_angle",
                        "w_ca_self_clash",
                        "w_chirality",
                        "w_chirality_atom",
                        "max_lddt_atoms",
                        "max_clash_atoms",
                    )
                },
                self_condition_prob=0.0,
            )
            (loss / args.grad_accum).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
    optimizer.step()
    ema.update(model.module)
    torch.cuda.synchronize(device)

    free, total = torch.cuda.mem_get_info(device)
    print(
        f"rank={rank} B={args.batch_size} input_L={input_length} "
        f"global_L={batch.max_len} "
        f"grad_accum={args.grad_accum} "
        f"loss={loss.item():.6f} elapsed={time.time() - started:.1f}s "
        f"allocated_peak={torch.cuda.max_memory_allocated(device) / 2**30:.3f}GiB "
        f"reserved_peak={torch.cuda.max_memory_reserved(device) / 2**30:.3f}GiB "
        f"total={total / 2**30:.3f}GiB free_after={free / 2**30:.3f}GiB",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
