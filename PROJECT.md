# PROJECT.md

## Status

- State: confirmed
- Last confirmed by: sjm0775@snu.ac.kr
- Last updated: 2026-07-21

## Project

- Name: MambaFold
- Type: ml
- Goal: thesis/paper-grade single-chain all-atom protein structure generation with direct all-atom flow matching.
- Core claim: a sufficiently scaled Bi-Mamba trunk with sparse gated attention and
  timestep-adaptive residuals can learn global fold and atomistic detail in one path
  without an explicit pair stack.
- Users/workflow: solo local/HPC training; the retained ESM3 run was trained on
  4x NVIDIA B200, while the new ESMC-6B mainline targets 4x RTX 6000 Ada. W&B
  is used for tracking and Boltz-style processed RCSB data for comparability.
- Scope: single-chain proteins, standard amino acids, L <= 1024 active training. MSA-free / PLM-conditioned path using pinned sequence-only ESMC-6B embeddings; legacy ESM3 caches/checkpoints remain reproducible but are not the next training path.
- Non-goals: multimer/interface prediction, ligands, nucleic acids, metals, cofactors, water, non-standard residues/PTMs, EqM.

## Architecture

- Active path: direct all-atom flow matching,
  `configs/direct_allatom_puremamba_attn6_geo_adaln_sf360_esmc6b.yaml`.
  - The Ada run uses `d_state=64`, MIMO rank 4, head dim 64, and the official
    minimum Mamba-3 chunk size 8. `d_state=128` requires 123,216 bytes of
    dynamic shared memory in the backward kernel, above RTX 6000 Ada's 101,376
    byte per-block limit. The `d_state=64` production path passed a BF16
    2-rank DDP batch-10/GPU, L=1024, gradient-accumulation optimizer+EMA smoke
    at 44.50 GiB peak allocation and 46.22 GiB peak reservation on each 47.37
    GiB GPU. The 4-GPU run uses accumulation 12 for effective batch 480.
  - Input: sequence/residue features, residue index, pinned ESMC-6B embeddings, FM time/noise, noised atom-slot coordinates.
  - Output: atom-slot velocity for the atom14+OXT layout plus CA/topology aux heads.
  - Active mainline losses: all-atom FM, sampled all-atom lDDT, soft C-alpha
    lDDT, dRMSD, small all-atom/CA geometry terms, C-alpha virtual-angle/
    self-clash, chirality (CA global handedness + per-residue Cα N/C/CB
    stereochemistry). Distogram/contact/pseudo-Cβ/confidence heads are
    implemented but weight-0 in the base mainline unless an ablation enables
    `pairfree_aux_heads` / related loss weights.
  - FM time/noise level enters via a sinusoidal TimeEmbedding + FiLM on the trunk
    input and inside the atom encoder/decoder (not a single scalar feature).
    Predicted pLDDT is written to inference PDB B-factors.
  - No recycling/cycle loop and no frozen conditioning path.
  - Self-conditioning support is implemented as an optional detached `x_hat`
    coordinate branch (`--self_conditioning`, `--self_condition_prob`), but it
    is not enabled in the active long baseline run.
  - Three SSM levels, atom → token → atom (`model/fold/atom_mamba.py`): an
    AtomEncoder (BiMamba over each residue's 15 atom slots → gated pool to a
    residue token) feeds a pair-free BiMamba residue trunk with sparse gated
    RoPE self-attention every 6 layers, and an AtomDecoder (residue latent +
    atom skip → BiMamba over atom slots) reads out per-atom velocity. Trunk
    blocks use timestep-conditioned AdaLN-Zero residual scale/shift/gates.
    No attention in the atom path — intra-residue (side-chain) geometry is SSM,
    inter-residue global communication is the residue trunk. SimpleFold-360M
    comparable trunk/atom sizing is used while keeping `max_length=1024`:
    `d_res=1024`, `n_trunk=18`, `n_attn_heads=16`, `d_atom=256`,
    `n_atom_layers=2`.

## Rationale

- Direct all-atom lets CA/backbone/side-chain errors co-adapt from step 1.
- The topology auxiliaries are kept so the model still receives strong global fold
  supervision while optimizing atom coordinates directly.

## Data

- Source/schema: official Boltz `rcsb_processed_targets.tar` preprocessing -> `.npz` records with
  `residues`, `atoms`, `chains` structured arrays. Do not rebuild these records
  from raw mmCIF for the active training corpus.
- Active data: `data/rcsb_boltz_official_full/`, generated only after the
  64,948,531,200-byte official archive passes size and record/structure pairing
  checks. The previous 67,657-record local directory was incomplete and is
  quarantined as `data/rcsb_boltz_partial_67k/`.
- PLM cache: the next training path uses pinned `biohub/ESMC-6B` revision
  `45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a` with 2560-dimensional embeddings
  under `data/rcsb_esmc6b_official_full/` and `data/afdb_swissprot_esmc6b/`.
  Active caches are content-addressed by SHA-256 of the full canonical sequence:
  `by_sequence/<hash-prefix>/<hash>.npy`. Structure/chain occurrences compute
  the same hash at load time, so identical sequences share one ESM inference and
  one stored array while retaining distinct coordinate examples. Legacy
  `<pdb>_ch<origin>.npy` caches remain readable as a fallback. The prior
  `data/rcsb_esm_official_full/` ESM3 cache covers only 127,946/211,742 RCSB
  entries and must not be described as complete.
- Splits: `data/splits/{train,val,val_casp,holdout_ids}.txt` are frozen. Active
  Boltz intersections are `train_boltz_official_full.txt`,
  `val_boltz_official_full.txt`, and `val_casp_boltz_official_full.txt`; regenerate them only with
  `scripts/setup_boltz_rcsb.py` from the frozen source splits.
- Active filtering: `single_chain_only: true` and `extract_monomer_chains: true`
  in `configs/direct_allatom_puremamba_attn6_geo_adaln_sf360_esmc6b.yaml`.

## Training Plan

- Phase A: direct all-atom from-scratch folding head, pair-free
  attn6+geometry+AdaLN-Zero, L=1024, conditioned on frozen sequence-only
  ESMC-6B embeddings via
  `configs/direct_allatom_puremamba_attn6_geo_adaln_sf360_esmc6b.yaml`.
  DDP length-bucket batches are drawn globally and then sharded across ranks;
  the final padded sequence length is max-synchronized before CUDA transfer.
  This keeps TileLang JIT shapes identical across ranks and prevents a faster
  rank from entering gradient all-reduce while another rank compiles a new
  sequence-length kernel.
  DataLoader workers are a per-rank setting and must be capped automatically
  from the CPU allocation visible to the process. The Slurm script intentionally
  does not request a CPU count; the code must fit the cluster default instead of
  multiplying a fixed worker count by four ranks.
- Phase B: CASP14 eval at 50k/100k/150k/200k checkpoints with
  SimpleFold-style metrics.
- Ablation queue: `scripts/run_selfcond_ablation_queue.sh` runs self-conditioning
  first, then self-conditioning + pair-free distogram/contact auxiliary heads.

## Commands

- Setup: `MAMBA_SKIP_CUDA_BUILD=TRUE uv sync --extra dev`
- Train default: `CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG=configs/direct_allatom_puremamba_attn6_geo_adaln_sf360_esmc6b.yaml bash scripts/train.sh`
- Slurm train: `sbatch scripts/slurm_train_esmc6b_ada.sh` (do not set a CPU count;
  the cluster default is used). The current account rejected `verylong` as an
  invalid QOS, so the active ESMC-6B Ada queue entry uses `long` QOS with a
  three-day wall time; do not point it at the retained ESM3 checkpoint.
- Slurm data gate: run `slurm_migrate_esmc6b_by_sequence.sh`, then
  `slurm_prepare_esmc6b_train_data.sh`; the latter builds monomer chain indexes
  off-GPU and requires a real mixed-source batch with 2560-dimensional ESMC.
- Resume: `CUDA_VISIBLE_DEVICES=0,1,2,3 RESUME=outputs/train/<run>/ckpt_latest.pt bash scripts/train.sh`
- Inference: `PYTHONPATH=src uv run python benchmarks/run_inference.py --ckpt <ckpt> --ids <ids.txt> --out <out_dir>`
- Score: `tools/scoring_venv/bin/python benchmarks/score_simplefold_metrics.py --in_dir <out_dir> --out <out_dir>/scores.json`
- Syntax smoke: `PYTHONPATH=src uv run --no-sync python -m py_compile ...`

## Verification

- Required quick checks after code edits:
  - `py_compile` on touched Python files.
  - config parse for `configs/direct_allatom_360m.yaml`.
  - single-chain dataset smoke: multichain example filtered, single-chain example kept.
- Before long training:
  - focused tests or full `uv run pytest` when feasible.
  - a 4-rank real-data loader preflight completes several synchronized batches
    within the Slurm default CPU/RAM allocation.
  - a 2-rank asymmetric-length model smoke completes one optimizer/EMA update.
  - a 4-rank production-shape model smoke completes before submitting the long
    run; GPU memory remains below the physical 48 GB limit on every rank.
  - config diff reviewed.
  - W&B run name/tags set.
  - no long run writes into an existing output directory; failed attempts are
    archived and the retained ESM3 checkpoint is never used as an ESMC resume.
- Metrics:
  - Train: all-atom FM, sampled all-atom lDDT, C-alpha lDDT, distogram, bond/clash metrics.
  - Later drug-discovery-oriented metrics: pocket heavy-atom RMSD, side-chain chi accuracy, clashscore.

## Paths

- Source: `src/mambafold/`
- Active config: `configs/direct_allatom_puremamba_attn6_geo_adaln_sf360_esmc6b.yaml`; legacy ESM3 config: `configs/direct_allatom_puremamba_attn6_geo_adaln_sf360.yaml`
- Train scripts: `scripts/train.py`, `scripts/train.sh`
- Benchmarks: `benchmarks/run_inference.py`, `benchmarks/score.py`, `benchmarks/score_simplefold_metrics.py`
- Outputs/checkpoints: `outputs/train/<run>/`

## References (borrowed concepts)

Papers whose ideas are used, and where in the code.

- **SimpleFold** (arXiv:2509.18480) — flow-matching folding with general blocks (no
  triangle/pair required); logit-normal time sampling. → FM objective + sampler
  (`data/transforms.py` `_sample_t`, `sampling/samplers.py`). We replace its Transformer
  with Bi-Mamba.
- **Flow Matching** (Lipman et al. 2023, arXiv:2210.02747) — `x_t=t·x_clean+(1-t)·ε`,
  velocity target, Euler ODE. → `losses`/`engine`/`samplers`.
- **Mamba-3** (arXiv:2603.15569, ICLR 2026) — SSM sequence model; used bidirectionally.
  → `model/bimamba3.py` (Bi-Mamba trunk, atom enc/dec).
- **AlphaFold2** (Jumper et al. 2021, Nature) — triangle multiplicative update, distogram
  aux, gated (output) self-attention, atom14 layout, FAPE/lDDT-style supervision.
  → `model/fold/multiplicative_update.py`, `pair_blocks.py`, `losses/`, `data/constants.py`.
- **AlphaFold3** (Abramson et al. 2024, Nature) — Pairformer lineage, relpos / chain-entity-sym
  encodings, confidence head framing, atom-level local distance/geometry supervision.
  → `model/embeddings.py`, `losses/lddt.py`, `losses/geometry.py`,
  `losses/ca_only.py`.
- **SeedFold** (arXiv:2512.24354v1) — Linear Triangle Attention (ReLU feature map +
  associative reorder, gated; O(L³)→O(L²)). → `model/fold/linear_tri_attn.py`
  (toggle `pair_use_tri_attn`; currently OFF under the Pairmixer preset).
- **"Triangle Multiplication is All You Need" / Pairmixer** (arXiv:2510.18870) — drop
  triangle attention, keep triangle multiplication (+FFN). → active pair-stack preset
  (`pair_use_tri_attn: false` in all configs).
- **NVIDIA Nemotron-H** (hybrid Mamba-Transformer) — interleave a few self-attention
  layers among Mamba layers. → `MambaStack` `attn_layers` / `AttnBlock`
  (`trunk_attn_layers: [10,11]`).
- **Attention Residuals** (Kimi Team, arXiv:2603.15031) — replace unit-weight residual
  accumulation with depth-wise softmax aggregation over preceding layer outputs
  (learned per-layer pseudo-query + RMSNorm keys), mitigating PreNorm dilution.
  → `MambaStack` `use_attn_residual` (`trunk_attn_residual: true`).
- **GAU** (Hua et al. 2022, "Transformer Quality in Linear Time") / **Qwen Gated Attention**
  (2025) — sigmoid gate on the attention output. → `GatedSelfAttention`.
- **LayerScale** (Touvron et al. 2021) / **Flamingo** tanh-gating (Alayrac et al. 2022) —
  zero-init per-channel residual gate so a new sublayer starts as identity. → `AttnBlock`
  `attn_scale` ("AttnResidual").
- **T5** (Raffel et al. 2020) — bucketed relative-position bias. → `RelativePositionBias`.
- **lDDT** (Mariani et al. 2013) — soft differentiable lDDT loss. → `losses/lddt.py`,
  `losses/ca_only.py`.
- **Engh & Huber 2001** — ideal covalent bond lengths. → `losses/geometry.py`.
- **ESMC** (Candido et al. 2026) — sequence-only PLM embeddings as conditioning;
  pinned 6B revision, no structure-token or coordinate supervision in the base
  PLM. → `data/esm.py`, model PLM projection. ESM3 remains a legacy/upper-bound
  conditioning path only.
- **Boltz-1** (Wohlwend et al. 2024) — Boltz-style processed RCSB `.npz` records. → `data/dataset.py` `RCSBDataset`.

## Do Not Touch Without Explicit Confirmation

- `data/rcsb/`
- `data/rcsb_esm/`, `data/rcsb_esmc/`
- `data/splits/`
- live `outputs/train/<active-run>/`
- destructive git operations

## Open Decisions

- Resolved (2026-07-15): PLM cache identity is the full canonical amino-acid
  sequence, not the PDB/chain occurrence. Storage uses deterministic SHA-256
  paths and global unique-sequence sharding. Structure occurrences remain
  separate training examples; cache deduplication does not itself change
  structure sampling weights.

- Resolved (2026-07-14): next mainline conditioning = pinned sequence-only
  ESMC-6B (`biohub/ESMC-6B`, revision
  `45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a`, d=2560). Do not reuse ESM3
  checkpoints because the PLM projection shape changes. CASP14 reporting must
  disclose that ESMC sequence pretraining uses post-CASP14 database snapshots.

- Resolved (2026-06-24): per-atom path = AF3-style atom→token→atom but all SSM —
  BiMamba AtomEncoder/Decoder over each residue's atom slots (intra-residue),
  not attention, to keep the MambaFold identity at every level.
  `model/fold/atom_mamba.py`. Mainline now uses SimpleFold-360M-comparable
  d_atom=256, n_atom_layers=2 while keeping max_length=1024.
  (Superseded the earlier interim lightweight MLP decoder.)
- Resolved (2026-07-04): explicit pair stack is not the mainline. 20k ablations
  showed pair was slower without a clear CASP14 gain. Mainline is pair-free
  BiMamba + sparse gated attention every 6 trunk layers + small geometry losses
  + AdaLN-Zero, with SimpleFold-360M-comparable sizing except max_length=1024.
  Note: a config plumbing bug made `trunk_attn_layers: []` override
  `trunk_attn_every`; fixed so empty lists now allow every-k attention.
- Resolved (2026-06-24): FM time sampling = logit_normal (oversample clean end).
- Resolved (2026-06-24): added per-residue Cα N/C/CB chirality (`allatom_chirality_loss`,
  `w_chirality_atom=0.5`), all-atom-lDDT confidence target (pLDDT) + B-factor
  surfacing, and FM time conditioning via TimeEmbedding+FiLM.
- Resolved (2026-06-24, multi-agent review): applied per-step CoM re-centering in
  the Euler sampler (was only initial-noise centered) and added a `w_fm` lever
  (default 1.0) for the core flow-matching loss. Confirmed not-bugs: GT losses use
  `valid_mask` while geometry priors use `atom_mask` (intentional — gives
  unobserved/modeled atoms ideal-geometry signal); SO(3) augmentation is
  per-example; val split is MMseqs2 cluster-based (`scripts/make_val_split.py`,
  homolog-leakage guarded).
- Open (ablate, do not apply blind): the `(1-t)` reconstruction Jacobian makes aux
  gradients vanish at t→1 while `alpha_mode: ramp` + `logit_normal` concentrate
  weight/sampling there — consider aux ~1/(1-t) weighting or a mid-t window; log
  per-term grad norms first.
- Open: explicit residue-frame / side-chain torsion heads.
- Implemented, pending ablation: self-conditioning (feed previous detached
  `x_hat`) in train and sampler.
- Implemented, pending ablation: pair-free O(L²) distogram/contact aux heads
  from residue features, without pair-stack feedback.
- Open: sampler 2nd-order/churn.
- Resolved for the Ada ESMC mainline: use `d_state=64`; the official Mamba-3
  MIMO backward kernel passed BF16 forward/backward at L=1024. Continue to
  monitor grad-clip hit-rate versus the non-smooth chirality hinges.
- Open: sampled all-atom lDDT/clash budgets 1024 vs 2048 vs 4096 atoms.
