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

from mambafold.data.constants import AA_TO_ID, RESIDUE_ATOMS  # noqa: E402
from mambafold.data.types import ProteinBatch  # noqa: E402
from mambafold.train.distributed import distributed_max_int  # noqa: E402
from mambafold.train.ema import EMA  # noqa: E402
from mambafold.train.engine import allatom_forward_and_loss  # noqa: E402
from mambafold.train.trainer import build_model  # noqa: E402


def make_batch(
    batch_size: int,
    length: int,
    d_plm: int,
    device: torch.device,
    *,
    structured_coordinates: bool = False,
    fixed_t: float | None = None,
):
    atoms = 15
    res_mask = torch.ones(batch_size, length, dtype=torch.bool, device=device)
    atom_mask = torch.ones(batch_size, length, atoms, dtype=torch.bool, device=device)
    zeros_res = torch.zeros(batch_size, length, dtype=torch.long, device=device)
    res_type = torch.randint(0, 20, (batch_size, length), device=device)
    x_clean = torch.randn(batch_size, length, atoms, 3, device=device)
    eps = torch.randn_like(x_clean)
    t = torch.rand(batch_size, 1, 1, 1, device=device)
    if structured_coordinates:
        # A long, locally perturbed chain keeps the exact clash candidate set
        # realistic while retaining a near-worst-case 14-heavy-atom residue.
        res_type.fill_(AA_TO_ID["TRP"])
        atom_mask.zero_()
        atom_mask[..., : len(RESIDUE_ATOMS["TRP"])] = True
        ca_x = 0.38 * (
            torch.arange(length, device=device, dtype=x_clean.dtype) - 0.5 * (length - 1)
        )
        centers = torch.zeros(1, length, 1, 3, device=device)
        centers[..., 0] = ca_x.view(1, length, 1)
        x_clean = centers + 0.08 * torch.randn_like(x_clean)
        eps = torch.randn_like(x_clean)
    if fixed_t is not None:
        t.fill_(fixed_t)
    x_t = t * x_clean + (1.0 - t) * eps
    return ProteinBatch(
        res_type=res_type,
        res_seq_nums=torch.arange(length, device=device).expand(batch_size, length).contiguous(),
        atom_type=torch.zeros(batch_size, length, atoms, dtype=torch.long, device=device),
        pair_type=torch.zeros(batch_size, length, atoms, dtype=torch.long, device=device),
        res_mask=res_mask,
        atom_mask=atom_mask,
        valid_mask=atom_mask.clone(),
        ca_mask=res_mask.clone(),
        chain_id=zeros_res.clone(),
        entity_id=zeros_res.clone(),
        sym_id=zeros_res.clone(),
        is_nterm=torch.zeros_like(res_mask),
        is_cterm=torch.zeros_like(res_mask),
        x_clean=x_clean,
        x_t=x_t,
        eps=eps,
        t=t,
        esm=torch.randn(batch_size, length, d_plm, device=device, dtype=torch.bfloat16),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/direct_allatom_puremamba_attn6_geo_adaln_sf360_esmc6b.yaml",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Defaults to the production batch_size from the config.",
    )
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument(
        "--length-sequence",
        default=None,
        help=(
            "Optional comma-separated lengths. Warm-up accumulation is used "
            "for preceding shapes and production accumulation for the final shape."
        ),
    )
    parser.add_argument(
        "--rank-length-step",
        type=int,
        default=0,
        help="Subtract this many residues per rank before global shape sync.",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="Steady-state accumulation count; defaults to the config value.",
    )
    parser.add_argument(
        "--warmup-grad-accum",
        type=int,
        default=1,
        help="Accumulation count for the first step that initializes AdamW state.",
    )
    parser.add_argument(
        "--optimizer-steps",
        type=int,
        default=2,
        help=(
            "Run at least two steps so the final step is measured after AdamW "
            "optimizer state has been allocated."
        ),
    )
    parser.add_argument("--structured-coordinates", action="store_true")
    parser.add_argument("--fixed-t", type=float, default=None)
    parser.add_argument("--min-free-gib", type=float, default=0.0)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    batch_size = int(cfg["batch_size"]) if args.batch_size is None else args.batch_size
    grad_accum = int(cfg["grad_accum_steps"]) if args.grad_accum is None else args.grad_accum
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if grad_accum < 1:
        raise ValueError("--grad-accum must be >= 1")
    if args.warmup_grad_accum < 1:
        raise ValueError("--warmup-grad-accum must be >= 1")
    if args.fixed_t is not None and not (0.0 <= args.fixed_t <= 1.0):
        raise ValueError("--fixed-t must be in [0, 1]")
    if args.min_free_gib < 0.0:
        raise ValueError("--min-free-gib must be non-negative")
    if args.length_sequence:
        lengths = [int(value.strip()) for value in args.length_sequence.split(",")]
        if len(lengths) < 2 or any(length < 1 for length in lengths):
            raise ValueError("--length-sequence requires at least two positive lengths")
    else:
        if args.optimizer_steps < 2:
            raise ValueError("--optimizer-steps must be >= 2")
        lengths = [args.length] * args.optimizer_steps
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
    started = time.time()
    loss = None
    final_input_length = None
    final_global_length = None
    for optimizer_step, requested_length in enumerate(lengths):
        input_length = requested_length - rank * args.rank_length_step
        if input_length < 1:
            raise ValueError(f"rank {rank} received invalid length {input_length}")
        batch = make_batch(
            batch_size,
            input_length,
            int(cfg["d_plm"]),
            device,
            structured_coordinates=args.structured_coordinates,
            fixed_t=args.fixed_t,
        )
        global_length = distributed_max_int(batch.max_len, device)
        batch = batch.pad_to_length(global_length)
        final_input_length = input_length
        final_global_length = batch.max_len
        micro_steps = grad_accum if optimizer_step == len(lengths) - 1 else args.warmup_grad_accum
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(micro_steps):
            sync_context = model.no_sync() if micro_step < micro_steps - 1 else nullcontext()
            with sync_context:
                loss, metrics = allatom_forward_and_loss(
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
                    w_ost_clash=float(cfg.get("w_ost_clash", 0.0)),
                    ost_clash_mode=str(cfg.get("ost_clash_mode", "huber")),
                    ost_clash_margin_A=float(cfg.get("ost_clash_margin_A", 0.1)),
                    ost_clash_huber_A=float(cfg.get("ost_clash_huber_A", 0.25)),
                    ost_clash_softplus_tau_A=float(cfg.get("ost_clash_softplus_tau_A", 0.05)),
                    ost_clash_softplus_halo=float(cfg.get("ost_clash_softplus_halo", 6.0)),
                    ost_clash_pair_chunk_size=int(cfg.get("ost_clash_pair_chunk_size", 1024)),
                    w_covalent_guard=float(cfg.get("w_covalent_guard", 0.0)),
                    covalent_guard_tolerance_z=float(cfg.get("covalent_guard_tolerance_z", 3.0)),
                    w_peptide_planarity_guard=float(cfg.get("w_peptide_planarity_guard", 0.0)),
                    geo_t_start=float(cfg.get("geo_t_start", 0.55)),
                    geo_t_ramp_end=float(cfg.get("geo_t_ramp_end", 0.65)),
                    geo_t_taper_start=float(cfg.get("geo_t_taper_start", 0.95)),
                    geo_t_end=float(cfg.get("geo_t_end", 0.98)),
                    geo_jacobian_floor=float(cfg.get("geo_jacobian_floor", 0.1)),
                    geo_max_examples_per_batch=int(cfg.get("geo_max_examples_per_batch", 0)),
                    self_condition_prob=0.0,
                )
                (loss / micro_steps).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
        optimizer.step()
        ema.update(model.module)
        torch.cuda.synchronize(device)
        free, _ = torch.cuda.mem_get_info(device)
        print(
            f"rank={rank} optimizer_step={optimizer_step + 1}/"
            f"{len(lengths)} length={batch.max_len} micro_steps={micro_steps} "
            f"free_after={free / 2**30:.3f}GiB",
            flush=True,
        )

    free, total = torch.cuda.mem_get_info(device)
    if float(cfg.get("w_ost_clash", 0.0)) > 0.0:
        if metrics["geo_active_fraction"] <= 0.0 or metrics["ost_clash"] <= 0.0:
            raise RuntimeError(
                "geometry smoke did not exercise a non-zero clash objective: "
                f"active={metrics['geo_active_fraction']} "
                f"ost={metrics['ost_clash']}"
            )
    if free / 2**30 < args.min_free_gib:
        raise RuntimeError(
            f"free GPU memory {free / 2**30:.3f} GiB is below required {args.min_free_gib:.3f} GiB"
        )
    print(
        f"rank={rank} B={batch_size} input_L={final_input_length} "
        f"global_L={final_global_length} lengths={lengths} "
        f"warmup_grad_accum={args.warmup_grad_accum} "
        f"steady_grad_accum={grad_accum} "
        f"optimizer_steps={len(lengths)} "
        f"loss={loss.item():.6f} elapsed={time.time() - started:.1f}s "
        f"allocated_peak={torch.cuda.max_memory_allocated(device) / 2**30:.3f}GiB "
        f"reserved_peak={torch.cuda.max_memory_reserved(device) / 2**30:.3f}GiB "
        f"total={total / 2**30:.3f}GiB free_after={free / 2**30:.3f}GiB",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
