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
    parser.add_argument("--grad_clip", type=float, default=1.0)
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
        "--find_unused_parameters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable DDP unused-parameter detection for ablations with disabled heads.",
    )
    parser.add_argument(
        "--crop_schedule",
        default=None,
        help="Mixed-crop schedule (optional). YAML list of phases: "
        "[{until: <step>, weights: {<L>: <prob>, ...}}, ...].",
    )
    # Geometric / auxiliary loss weights
    parser.add_argument(
        "--w_bond",
        type=float,
        default=0.0,
        help="Weight for backbone + Cβ bond-length loss (0 disables).",
    )
    parser.add_argument(
        "--w_clash", type=float, default=0.0, help="Weight for sampled all-atom steric clash loss."
    )
    parser.add_argument(
        "--w_ca_clash", type=float, default=0.0, help="Weight for C-alpha steric clash loss."
    )
    parser.add_argument(
        "--w_ost_clash",
        type=float,
        default=0.0,
        help="Weight for the atom-normalized OpenStructure-aligned clash surrogate.",
    )
    parser.add_argument(
        "--ost_clash_mode",
        choices=("huber", "softplus"),
        default="huber",
    )
    parser.add_argument("--ost_clash_margin_A", type=float, default=0.1)
    parser.add_argument("--ost_clash_huber_A", type=float, default=0.25)
    parser.add_argument("--ost_clash_softplus_tau_A", type=float, default=0.05)
    parser.add_argument("--ost_clash_softplus_halo", type=float, default=6.0)
    parser.add_argument("--ost_clash_pair_chunk_size", type=int, default=1024)
    parser.add_argument("--w_covalent_guard", type=float, default=0.0)
    parser.add_argument("--covalent_guard_tolerance_z", type=float, default=3.0)
    parser.add_argument("--w_peptide_planarity_guard", type=float, default=0.0)
    parser.add_argument("--geo_t_start", type=float, default=0.55)
    parser.add_argument("--geo_t_ramp_end", type=float, default=0.65)
    parser.add_argument("--geo_t_taper_start", type=float, default=0.95)
    parser.add_argument("--geo_t_end", type=float, default=0.98)
    parser.add_argument("--geo_jacobian_floor", type=float, default=0.1)
    parser.add_argument(
        "--geo_max_examples_per_batch",
        type=int,
        default=0,
        help="Bound costly geometry examples per rank/microbatch (0 means all).",
    )
    parser.add_argument(
        "--w_distogram",
        type=float,
        default=0.0,
        help="Auxiliary distogram CE loss (binned Cα-Cα distance). "
        "AF2/AF3-style aux supervision; helps fold geometry.",
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
    # Pair-free aux heads. There is no pair stack and no trunk attention to
    # configure: both were removed from the model rather than defaulted off.
    parser.add_argument(
        "--pairfree_aux_heads",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Build O(L^2) distogram/contact heads from residue features. "
        "Output-only — nothing is fed back into the trunk.",
    )
    parser.add_argument(
        "--d_aux_pair",
        type=int,
        default=192,
        help="Width of the pairfree aux-head outer sum. Unused unless "
        "--pairfree_aux_heads.",
    )
    parser.add_argument(
        "--trunk_time_film",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Inject FM time embedding before every residue trunk block.",
    )
    parser.add_argument(
        "--trunk_adaln_zero",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use AdaLN-Zero time-conditioned scale/shift/gates inside "
        "each residue trunk residual branch.",
    )
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
    parser.add_argument("--d_plm_proj", type=int, default=256)
    parser.add_argument("--d_ca_emb", type=int, default=128)
    # Auxiliary loss weights.
    # Weight on the core flow-matching velocity loss (the generative objective).
    # 1.0 = current behaviour; lower the aux weights or raise this if the ~5-6x
    # nominal aux mass is found to drown the learned vector field.
    parser.add_argument("--w_fm", type=float, default=1.0)
    parser.add_argument("--w_lddt_atom", type=float, default=1.0)
    parser.add_argument("--w_lddt_ca", type=float, default=0.0)
    parser.add_argument("--w_drmsd", type=float, default=0.0)
    parser.add_argument("--w_contact", type=float, default=0.0)
    parser.add_argument("--w_pcb", type=float, default=0.0)
    parser.add_argument("--w_conf", type=float, default=0.0)
    parser.add_argument("--w_ca_angle", type=float, default=0.0)
    parser.add_argument("--w_ca_self_clash", type=float, default=0.0)
    # Chirality (handedness) loss: penalizes mirror-image predictions — the
    # dominant holdout failure (~35% are internally-correct mirror images that
    # every reflection-invariant aux misses). Key term; tune if it fights local
    # geometry early.
    parser.add_argument("--w_chirality", type=float, default=0.0)
    # All-atom (Cα N/C/CB tetrahedron) chirality — per-residue side-chain
    # stereochemistry, complements the CA-CA-CA global-handedness term above.
    parser.add_argument("--w_chirality_atom", type=float, default=0.0)
    parser.add_argument("--lddt_cutoff_A", type=float, default=15.0)
    parser.add_argument(
        "--max_lddt_atoms",
        type=int,
        default=0,
        help="All-atom lDDT atom cap; 0 uses every resolved atom (baseline).",
    )
    parser.add_argument(
        "--lddt_pair_chunk_size",
        type=int,
        default=512,
        help="Rows per exact lDDT ground-truth distance chunk.",
    )
    parser.add_argument("--max_clash_atoms", type=int, default=2048)
    # Length-balanced sampler — counters the PDB short-tail bias (90% < 500 aa)
    # by upweighting longer proteins. dataset audit identified this as the main
    # cause of mono lDDT degradation at L=512-1024 (0.80 → 0.72).
    parser.add_argument("--length_balanced_sampling", action="store_true", default=False)
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
    parser.add_argument("--metadata_path", default="data/splits/metadata.tsv")
    parser.add_argument("--length_balance_mode", default="power", choices=["power", "linear_clip"])
    parser.add_argument(
        "--length_balance_exponent",
        type=float,
        default=0.5,
        help="Used when mode=power. w = (L/200)^exponent clipped.",
    )
    parser.add_argument("--length_balance_clip_min", type=float, default=1.0)
    parser.add_argument("--length_balance_clip_max", type=float, default=1.5)
    # W&B
    parser.add_argument("--wandb_project", default="mambafold")
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=[])
    parser.add_argument("--wandb_offline", action="store_true", default=False)
    parser.add_argument("--no_wandb", action="store_true", default=False)

    parser.set_defaults(**cfg)
    args = parser.parse_args(argv)

    if args.max_lddt_atoms < 0:
        parser.error("--max_lddt_atoms must be non-negative (0 means all atoms)")
    if args.lddt_pair_chunk_size <= 0:
        parser.error("--lddt_pair_chunk_size must be positive")
    if args.lddt_cutoff_A <= 0:
        parser.error("--lddt_cutoff_A must be positive")
    if args.geo_max_examples_per_batch < 0:
        parser.error("--geo_max_examples_per_batch must be non-negative")
    if not (
        0.0
        <= args.geo_t_start
        < args.geo_t_ramp_end
        <= args.geo_t_taper_start
        < args.geo_t_end
        <= 1.0
    ):
        parser.error(
            "geometry time bounds must satisfy 0 <= start < ramp_end <= taper_start < end <= 1"
        )
    if not (0.0 < args.geo_jacobian_floor <= 1.0):
        parser.error("--geo_jacobian_floor must be in (0, 1]")
    if args.initialize_model_from_ema and not (args.resume and args.reset_optimizer):
        parser.error("--initialize_model_from_ema requires --resume and --reset_optimizer")
    if args.expected_resume_step is not None and not args.resume:
        parser.error("--expected_resume_step requires --resume")

    if args.out_dir is None:
        job_id = os.environ.get("SLURM_JOB_ID", None)
        tag = job_id if job_id else time.strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"outputs/train/{tag}"

    return args, cfg
