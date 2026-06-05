"""Training configuration: YAML loading + CLI argument parsing."""

import argparse
import os
import time

import yaml


def parse_args(argv=None):
    """Parse training config from YAML file + CLI overrides."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args(argv)

    cfg = {}
    if pre_args.config:
        with open(pre_args.config) as f:
            cfg = yaml.safe_load(f) or {}

    parser = argparse.ArgumentParser(description="MambaFold training")
    parser.add_argument("--config", default=None)
    # Data
    parser.add_argument("--data_dir", default="afdb_data/train")
    parser.add_argument("--val_data_dir", default=None)
    parser.add_argument("--file_list", default=None)
    parser.add_argument("--val_file_list", default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--copies_per_protein", type=int, default=1)
    parser.add_argument("--single_chain_only", action="store_true", default=False,
                        help="Use only entries with exactly one kept protein chain.")
    # Output
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--resume", default=None)
    # Training
    parser.add_argument("--total_steps", type=int, default=200_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=2_000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--ckpt_interval", type=int, default=5_000)
    parser.add_argument("--eval_interval", type=int, default=0)
    parser.add_argument("--t_schedule", default="uniform",
                        help="Time sampling schedule: 'uniform' (FM standard) or "
                             "'logit_normal' (SimpleFold-style oversampling near t→1).")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation: effective batch = "
                             "batch_size × world_size × grad_accum_steps. "
                             "DDP all-reduce is throttled to the last micro-step.")
    parser.add_argument("--crop_schedule", default=None,
                        help="Mixed-crop schedule (optional). YAML list of phases: "
                             "[{until: <step>, weights: {<L>: <prob>, ...}}, ...].")
    # Geometric / auxiliary loss weights
    parser.add_argument("--w_bond", type=float, default=0.0,
                        help="Weight for backbone + Cβ bond-length loss (0 disables).")
    parser.add_argument("--w_clash", type=float, default=0.0,
                        help="Weight for Cα-Cα steric clash loss.")
    parser.add_argument("--w_distogram", type=float, default=0.0,
                        help="Auxiliary distogram CE loss (binned Cα-Cα distance). "
                             "AF2/AF3-style aux supervision; helps fold geometry.")
    parser.add_argument("--alpha_mode", default="ramp",
                        help="lDDT weight mode: 'const' (α=1) or 'ramp' "
                             "(α = 1 + 8·ReLU(t-0.5).mean → strongest at clean end).")
    parser.add_argument("--reset_optimizer", action="store_true", default=False,
                        help="On --resume, keep model+ema weights but re-initialize "
                             "optimizer and scheduler with current args (lr/warmup/"
                             "total_steps). Use at a stage transition (e.g. PT L=512 "
                             "→ CT L=1024).")
    parser.add_argument("--start_step", type=int, default=0,
                        help="Override starting step counter (for fresh stage start).")
    # Model
    parser.add_argument("--d_atom", type=int, default=256)
    parser.add_argument("--d_res", type=int, default=256)
    parser.add_argument("--d_state", type=int, default=64)
    parser.add_argument("--mimo_rank", type=int, default=4)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--n_atom_enc", type=int, default=2)
    parser.add_argument("--n_trunk", type=int, default=6)
    parser.add_argument("--n_atom_dec", type=int, default=2)
    parser.add_argument("--d_res_pos", type=int, default=64)
    parser.add_argument("--d_atom_slot", type=int, default=32)
    parser.add_argument("--d_res_type", type=int, default=32,
                        help="Residue-type embedding dim fed to trunk (sequence signal)")
    # PLM
    parser.add_argument("--use_plm", action="store_true", default=False)
    parser.add_argument("--d_plm", type=int, default=1536)
    parser.add_argument("--esm_dir", default=None)
    # Which stage to train. 1 = CA-only, 2 = all-atom (frozen S1), joint = both.
    parser.add_argument("--stage", default=1)
    # Stage 1 pair-side knobs.
    parser.add_argument("--d_pair", type=int, default=192)
    parser.add_argument("--n_pair_blocks", type=int, default=4)
    # Pair-block composition toggles (one code path, multiple designs):
    #   full=mult+attn (default), pairmixer=mult only (arXiv:2510.18870), attn-only.
    parser.add_argument("--pair_use_mult_update", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pair_use_tri_attn", action=argparse.BooleanOptionalAction, default=True)
    # Nemotron-style hybrid: make some trunk layers self-attention (gated +
    # zero-init AttnResidual). `trunk_attn_layers` = explicit 0-based indices
    # (e.g. [10,11] = last two of 12); `trunk_attn_every` = every k-th layer.
    parser.add_argument("--trunk_attn_layers", type=int, nargs="*", default=None)
    parser.add_argument("--trunk_attn_every", type=int, default=None)
    parser.add_argument("--n_attn_heads", type=int, default=16)
    parser.add_argument("--n_pair_heads", type=int, default=4)
    parser.add_argument("--d_pair_head", type=int, default=48)
    parser.add_argument("--pair_mult_c", type=int, default=128)
    parser.add_argument("--d_plm_proj", type=int, default=256)
    parser.add_argument("--d_ca_emb", type=int, default=128)
    # Stage 1 aux loss weights.
    parser.add_argument("--w_lddt_ca", type=float, default=1.0)
    parser.add_argument("--w_bond_caca", type=float, default=0.1)
    # Stage 1 scaffold-quality aux: distance-map, long-range contact, pseudo-Cβ
    # orientation, confidence calibration, local Cα geometry.
    parser.add_argument("--w_drmsd", type=float, default=0.5)
    parser.add_argument("--w_contact", type=float, default=0.3)
    parser.add_argument("--w_pcb", type=float, default=0.2)
    parser.add_argument("--w_conf", type=float, default=0.05)
    parser.add_argument("--w_ca_angle", type=float, default=0.1)
    parser.add_argument("--w_ca_self_clash", type=float, default=0.1)
    parser.add_argument("--n_cycles_train", type=int, default=1,
                        help="Stage 1 recycling iterations (1 = no recycling). "
                             "Earlier cycles run under no_grad and feed their "
                             "predicted Cα distance map back into the pair init.")
    # Stage 2: Stage 1 ckpt path and aux weights.
    parser.add_argument("--stage1_ckpt", default=None,
                        help="Path to a Phase-1 ckpt whose weights initialise Stage 1.")
    parser.add_argument("--w_lddt_full", type=float, default=1.0)
    parser.add_argument("--w_ca_anchor", type=float, default=2.0,
                        help="Stage 2 CA residual anchor to Stage 1 CA scaffold.")
    parser.add_argument("--ca_condition_noise_std", type=float, default=0.0,
                        help="Stddev of optional noise added to Stage 2 CA condition (normalized units).")
    parser.add_argument("--ca_condition_noise_prob", type=float, default=0.0,
                        help="Probability of applying Stage 2 CA condition noise per forward.")
    # Joint finetune: Phase-2 ckpt to warm-start; falls back to
    # `stage1_ckpt` + fresh Stage 2 when omitted.
    parser.add_argument("--joint_init_ckpt", default=None,
                        help="TwoStage ckpt (Phase-2 output) for joint warm start.")
    parser.add_argument("--w_stage1", type=float, default=1.0,
                        help="Global multiplier on Stage 1 loss sum during joint.")
    # Length-balanced sampler — counters the PDB short-tail bias (90% < 500 aa)
    # by upweighting longer proteins. dataset audit identified this as the main
    # cause of mono lDDT degradation at L=512-1024 (0.80 → 0.72).
    parser.add_argument("--length_balanced_sampling", action="store_true", default=False)
    parser.add_argument("--metadata_path", default="data/splits/metadata.tsv")
    parser.add_argument("--length_balance_mode", default="power",
                        choices=["power", "linear_clip"])
    parser.add_argument("--length_balance_exponent", type=float, default=0.5,
                        help="Used when mode=power. w = (L/200)^exponent clipped.")
    parser.add_argument("--length_balance_clip_min", type=float, default=1.0)
    parser.add_argument("--length_balance_clip_max", type=float, default=1.5)
    # Stage 2 atom-side dims.
    parser.add_argument("--d_res_polish", type=int, default=512)
    parser.add_argument("--n_polish", type=int, default=4)
    parser.add_argument("--d_ca_anchor", type=int, default=64)
    parser.add_argument("--d_fourier", type=int, default=128)
    # W&B
    parser.add_argument("--wandb_project", default="mambafold")
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=[])
    parser.add_argument("--wandb_offline", action="store_true", default=False)
    parser.add_argument("--no_wandb", action="store_true", default=False)

    parser.set_defaults(**cfg)
    args = parser.parse_args(argv)

    if args.stage in ("1", "stage1"):
        args.stage = 1
    elif args.stage in ("2", "stage2"):
        args.stage = 2

    if args.out_dir is None:
        job_id = os.environ.get("SLURM_JOB_ID", None)
        tag = job_id if job_id else time.strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"outputs/train/{tag}"

    return args, cfg
