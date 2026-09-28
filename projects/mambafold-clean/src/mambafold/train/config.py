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
        if not isinstance(cfg, dict):
            pre.error("training config must be a YAML mapping")

    parser = argparse.ArgumentParser(description="MambaFold training")
    parser.add_argument("--config", default=None)
    # Data
    parser.add_argument("--data_dir", default="afdb_data/train")
    parser.add_argument("--file_list", default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--loader_timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for a DataLoader worker before failing the run.",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=1,
        help="Batches prefetched by each DataLoader worker.",
    )
    parser.add_argument("--copies_per_protein", type=int, default=1)
    parser.add_argument(
        "--accum_same_protein",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Accumulate gradient over the same protein across micro-steps, so "
        "copies_per_protein x grad_accum_steps equals that many noise copies in "
        "one pass. Requires length_bucketing.",
    )
    parser.add_argument(
        "--single_chain_only",
        action="store_true",
        default=False,
        help="Use only entries with exactly one kept protein chain.",
    )
    # Output
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--resume", default=None)
    # Training
    parser.add_argument("--total_steps", type=int, default=200_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=2_000)
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Warmup start LR. SimpleFold's LinearWarmup ramps from here to --lr.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="SimpleFold's AdamW uses 0.0; this was silently 1e-2 before.",
    )
    parser.add_argument("--grad_clip", type=float, default=2.0)
    parser.add_argument(
        "--lr_cooldown_steps",
        type=int,
        default=0,
        help="Cosine decay to --min_lr over the final N steps. 0 reproduces "
        "SimpleFold's flat rate exactly.",
    )
    parser.add_argument(
        "--prewarm_kernels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile every binned sequence shape before step 1, so no training "
        "step stalls on a first-use TileLang compile.",
    )
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--ckpt_interval", type=int, default=5_000)
    parser.add_argument(
        "--keep_last_checkpoints",
        type=int,
        default=3,
        help="Keep this many most recent numbered checkpoints in addition to milestones.",
    )
    parser.add_argument(
        "--keep_checkpoint_steps",
        type=int,
        nargs="*",
        default=[],
        help="Numbered milestone checkpoints that pruning must retain.",
    )
    parser.add_argument(
        "--t_schedule",
        choices=("uniform", "logit_normal"),
        default="logit_normal",
        help="Time sampling schedule. The SimpleFold baseline uses logit_normal.",
    )
    parser.add_argument(
        "--t_uniform_weight",
        type=float,
        default=0.02,
        help="Interpolation weight between a logit-normal draw and a uniform draw. "
        "This is not a mixture: raising it pulls samples toward 0.5 and narrows "
        "both tails. SimpleFold uses 0.02.",
    )
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation: effective batch = "
        "batch_size × world_size × grad_accum_steps. "
        "DDP all-reduce is throttled to the last micro-step.",
    )
    parser.add_argument(
        "--alpha_mode",
        choices=("const", "ramp"),
        default="const",
        help="lDDT weight mode: const for pretraining; ramp for fine-tuning.",
    )
    parser.add_argument(
        "--use_rigid_align",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align clean coordinates to the detached one-step prediction for FM MSE.",
    )
    parser.add_argument(
        "--reset_optimizer",
        action="store_true",
        default=False,
        help="On --resume, keep model+ema weights but re-initialize "
        "optimizer and scheduler with current args (lr/warmup/"
        "total_steps).",
    )
    parser.add_argument(
        "--start_step",
        type=int,
        default=0,
        help="Override starting step counter when resetting optimizer/scheduler.",
    )
    parser.add_argument(
        "--initialize_model_from_ema",
        action="store_true",
        default=False,
        help="With --reset_optimizer, initialize both train model and new EMA from checkpoint EMA.",
    )
    parser.add_argument(
        "--strict_resume",
        action="store_true",
        default=False,
        help="Fail on missing or unexpected model/EMA checkpoint keys.",
    )
    parser.add_argument(
        "--expected_missing_resume_keys",
        nargs="*",
        default=[],
        help="With --strict_resume, permit exactly these missing model and EMA keys. "
        "This is intended for an explicit zero-initialized architecture addition, "
        "such as enabling self-conditioning on an older checkpoint.",
    )
    parser.add_argument(
        "--expected_resume_step",
        type=int,
        default=None,
        help="Fail unless the loaded checkpoint has this exact optimizer step.",
    )
    # Model
    parser.add_argument("--d_res", type=int, default=256)
    parser.add_argument("--d_state", type=int, default=64)
    parser.add_argument("--mimo_rank", type=int, default=4)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--n_trunk", type=int, default=6)
    parser.add_argument("--d_res_pos", type=int, default=64)
    parser.add_argument(
        "--d_res_type",
        type=int,
        default=32,
        help="Residue-type embedding dim fed to trunk (sequence signal)",
    )
    # PLM
    parser.add_argument("--use_plm", action="store_true", default=False)
    parser.add_argument("--d_plm", type=int, default=1536)
    parser.add_argument("--esm_dir", default=None)
    parser.add_argument(
        "--self_conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Feed a detached x0 estimate back into the coordinate encoder.",
    )
    parser.add_argument(
        "--self_condition_prob",
        type=float,
        default=0.0,
        help="Training probability of computing and using self-conditioning.",
    )
    # Weight-tie BiMamba fwd/bwd (one shared SSM both directions) — halves trunk SSM params.
    parser.add_argument("--bimamba_share", action=argparse.BooleanOptionalAction, default=False)
    # Atom-level BiMamba encoder/decoder (intra-residue SSM; atom→token→atom).
    parser.add_argument("--d_atom", type=int, default=128)
    parser.add_argument("--n_atom_layers", type=int, default=4)
    # The atom levels are a fixed 14-slot tuple per residue, not a sequence, so
    # they carry their own settings rather than the trunk's.
    parser.add_argument(
        "--atom_mixer",
        choices=("mamba", "mlp"),
        default="mamba",
        help="'mlp' swaps the atom-slot scan for an MLP-Mixer over slots.",
    )
    parser.add_argument("--atom_d_state", type=int, default=64)
    parser.add_argument(
        "--n_atom_cross_layers",
        type=int,
        default=1,
        help="BiMamba layers along the residue axis over the backbone atom "
        "streams. 0 leaves the atom levels strictly intra-residue.",
    )
    parser.add_argument(
        "--n_backbone_streams",
        type=int,
        default=5,
        help="Atom slots carried across residues: N, CA, C, O, CB.",
    )
    parser.add_argument(
        "--atom_mimo_rank",
        type=int,
        default=2,
        help="Sets the SSM chunk size: rank 1 gives chunk 64 and pads A=14 to "
        "64; rank 2 gives chunk 16 and pads to 16.",
    )
    parser.add_argument("--d_plm_proj", type=int, default=256)
    parser.add_argument("--d_ca_emb", type=int, default=128)
    parser.add_argument(
        "--d_temb",
        type=int,
        default=128,
        help="Width the FM time embedding is carried at. AdaLN-Zero projects "
        "d_temb -> 6*d_res in every trunk block, so this multiplies through "
        "the budget; SimpleFold's DiT uses the full trunk width.",
    )
    # Folding objective. Geometry weights stay zero for pretraining and are
    # enabled only by the declared fine-tune invocation.
    parser.add_argument("--w_fm", type=float, default=1.0)
    parser.add_argument("--w_lddt_atom", type=float, default=1.0)
    parser.add_argument("--w_bond", type=float, default=0.0)
    parser.add_argument("--w_angle", type=float, default=0.0)
    parser.add_argument("--w_clash", type=float, default=0.0)
    parser.add_argument("--lddt_cutoff_A", type=float, default=15.0)
    parser.add_argument(
        "--lddt_pair_chunk_size",
        type=int,
        default=512,
        help="Rows per exact lDDT ground-truth distance chunk.",
    )
    parser.add_argument(
        "--clash_overlap_tolerance_A",
        type=float,
        default=1.5,
        help="OpenStructure-style allowed overlap between summed atom VDW radii. "
        "The 1.5-A default is intentionally more tolerant than MolProbity's "
        "hydrogen-aware 0.4-A clashscore threshold; S-S keeps OpenStructure's "
        "separate 2.03-1.00-A floor.",
    )
    parser.add_argument(
        "--clash_margin_A",
        type=float,
        default=0.1,
        help="Start the differentiable clash barrier this far outside the hard threshold.",
    )
    parser.add_argument(
        "--clash_huber_delta_A",
        type=float,
        default=0.25,
        help="Transition width of the clash Huber penalty in Angstroms.",
    )
    parser.add_argument(
        "--clash_soft_count_tau_A",
        type=float,
        default=0.05,
        help="Temperature of the smooth clash-count diagnostic in Angstroms.",
    )
    parser.add_argument(
        "--clash_pair_chunk_size",
        type=int,
        default=256,
        help="Candidate residue pairs per differentiable clash chunk.",
    )
    # Length-balanced sampler — counters the PDB short-tail bias (90% < 500 aa)
    # by upweighting longer proteins. dataset audit identified this as the main
    # cause of mono lDDT degradation at L=512-1024 (0.80 → 0.72).
    # Padding-waste controls. length_bin>0: collator pads each batch to the next
    # multiple of length_bin above batch_max (dynamic padding, bounded shapes).
    # length_bucketing: group near-equal-length proteins per batch (needs
    # batch_size>1) so batch_max ≈ each sequence length. Together they cut
    # padding and shape-specialization waste for every model stage.
    parser.add_argument("--length_bin", type=int, default=0)
    parser.add_argument("--length_bucketing", action="store_true", default=False)
    # Workers for the one-time per-file length cache (built on first bucketing run,
    # then reused). Pre-build with scripts/precompute_lengths.py to avoid the
    # startup scan.
    parser.add_argument("--length_cache_workers", type=int, default=8)
    # Monomer extraction: index every protein chain of every entry as its own
    # single-chain example (supersedes single_chain_only). Turns multimers into
    # extra monomer training data. Builds a cached chain index on first run.
    parser.add_argument("--extract_monomer_chains", action="store_true", default=False)
    # Fraction of a crop's canonical atoms that must be resolved for the chain to
    # be usable. SimpleFold has no such filter; 0.0 reproduces that.
    parser.add_argument("--min_obs_ratio", type=float, default=0.0)
    # Homomer copies of one sequence inside one entry are the same target once
    # the FM loss rigid-aligns, so they multiply sampling weight without adding
    # supervision. Collapse them to the best-resolved copy.
    parser.add_argument(
        "--dedup_homomer_chains",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="With --extract_monomer_chains, keep one chain per distinct "
        "sequence per entry (the one resolving the most atoms).",
    )
    # W&B
    parser.add_argument("--wandb_project", default="mambafold")
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=[])
    parser.add_argument("--wandb_offline", action="store_true", default=False)
    parser.add_argument("--no_wandb", action="store_true", default=False)

    valid_config_keys = {action.dest for action in parser._actions} | {"train_sources"}
    unknown_config_keys = sorted(set(cfg) - valid_config_keys)
    if unknown_config_keys:
        parser.error("unknown config key(s): " + ", ".join(unknown_config_keys))

    parser.set_defaults(**cfg)
    args = parser.parse_args(argv)

    if args.lddt_pair_chunk_size <= 0:
        parser.error("--lddt_pair_chunk_size must be positive")
    if args.lddt_cutoff_A <= 0:
        parser.error("--lddt_cutoff_A must be positive")
    if any(
        weight < 0
        for weight in (args.w_fm, args.w_lddt_atom, args.w_bond, args.w_angle, args.w_clash)
    ):
        parser.error("folding loss weights must be non-negative")
    if args.clash_pair_chunk_size <= 0:
        parser.error("--clash_pair_chunk_size must be positive")
    if args.clash_overlap_tolerance_A < 0:
        parser.error("--clash_overlap_tolerance_A must be non-negative")
    if args.clash_margin_A < 0:
        parser.error("--clash_margin_A must be non-negative")
    if args.clash_huber_delta_A <= 0:
        parser.error("--clash_huber_delta_A must be positive")
    if args.clash_soft_count_tau_A <= 0:
        parser.error("--clash_soft_count_tau_A must be positive")
    if args.copies_per_protein < 1:
        parser.error("--copies_per_protein must be positive")
    if not (0.0 <= args.self_condition_prob <= 1.0):
        parser.error("--self_condition_prob must be in [0, 1]")
    if args.initialize_model_from_ema and not (args.resume and args.reset_optimizer):
        parser.error("--initialize_model_from_ema requires --resume and --reset_optimizer")
    if args.expected_resume_step is not None and not args.resume:
        parser.error("--expected_resume_step requires --resume")
    if args.expected_missing_resume_keys and not (args.resume and args.strict_resume):
        parser.error("--expected_missing_resume_keys requires --resume and --strict_resume")
    if len(args.expected_missing_resume_keys) != len(set(args.expected_missing_resume_keys)):
        parser.error("--expected_missing_resume_keys must not contain duplicates")

    if args.out_dir is None:
        job_id = os.environ.get("SLURM_JOB_ID", None)
        tag = job_id if job_id else time.strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"outputs/train/{tag}"

    return args, cfg
