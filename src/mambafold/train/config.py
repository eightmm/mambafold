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
    parser.add_argument("--gamma_schedule", default="logit_normal")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=0)
    # Finetune-specific
    parser.add_argument("--alpha_mode", default="const",
                        help="lDDT weight mode: 'const' (pretrain) or 'ramp' (finetune).")
    parser.add_argument("--reset_optimizer", action="store_true", default=False,
                        help="On --resume, keep model+ema weights but re-initialize "
                             "optimizer and scheduler with current args (lr/warmup/"
                             "total_steps). Use when switching stages (e.g. 256→512).")
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
    # PLM
    parser.add_argument("--use_plm", action="store_true", default=False)
    parser.add_argument("--d_plm", type=int, default=1536)
    parser.add_argument("--esm_dir", default=None)
    # W&B
    parser.add_argument("--wandb_project", default="mambafold")
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=[])
    parser.add_argument("--wandb_offline", action="store_true", default=False)
    parser.add_argument("--no_wandb", action="store_true", default=False)

    parser.set_defaults(**cfg)
    args = parser.parse_args(argv)

    if args.out_dir is None:
        job_id = os.environ.get("SLURM_JOB_ID", None)
        tag = job_id if job_id else time.strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"outputs/train/{tag}"

    return args, cfg
