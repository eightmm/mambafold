# PROJECT.md

## Status

- State: confirmed
- Last confirmed by: sjm0775@snu.ac.kr
- Last updated: 2026-06-24

## Project

- Name: MambaFold
- Type: ml
- Goal: thesis/paper-grade single-chain all-atom protein structure generation with direct all-atom flow matching.
- Core claim: a sufficiently scaled Bi-Mamba + pair reasoning model can learn global fold and atomistic detail in one path.
- Users/workflow: solo local/HPC training on 4x NVIDIA B200; W&B for tracking; Boltz-style processed RCSB data for comparability.
- Scope: single-chain proteins, standard amino acids, L <= 1024 active training. MSA-free / PLM-conditioned path using ESM3 cache.
- Non-goals: multimer/interface prediction, ligands, nucleic acids, metals, cofactors, water, non-standard residues/PTMs, EqM.

## Architecture

- Active path: direct all-atom flow matching, `configs/direct_allatom_360m.yaml`.
  - Input: sequence/residue features, residue index, ESM3, FM time/noise, noised atom-slot coordinates.
  - Output: atom-slot velocity for the atom14+OXT layout plus CA/topology aux heads.
  - Losses: all-atom FM, sampled all-atom lDDT, soft C-alpha lDDT,
    all-atom bond/clash geometry, CA clash, dRMSD, distogram,
    long-range contact, pseudo-Cβ direction, all-atom-lDDT confidence
    (pLDDT) calibration, C-alpha virtual-angle/self-clash, chirality
    (CA global handedness + per-residue Cα N/C/CB stereochemistry).
  - FM time/noise level enters via a sinusoidal TimeEmbedding + FiLM on the trunk
    input and inside the atom encoder/decoder (not a single scalar feature).
    Predicted pLDDT is written to inference PDB B-factors.
  - No recycling/cycle loop and no frozen conditioning path.
  - Three SSM levels, atom → token → atom (`model/fold/atom_mamba.py`): an
    AtomEncoder (BiMamba over each residue's 15 atom slots → gated pool to a
    residue token) feeds the Mamba + triangle-mult pair trunk (global,
    inter-residue), and an AtomDecoder (residue latent + atom skip → BiMamba over
    atom slots) reads out per-atom velocity. No attention in the atom path —
    intra-residue (side-chain) geometry is SSM, inter-residue is the trunk.
    `d_atom=128`, `n_atom_layers=4` (enc & dec).

## Rationale

- Direct all-atom lets CA/backbone/side-chain errors co-adapt from step 1.
- The topology auxiliaries are kept so the model still receives strong global fold
  supervision while optimizing atom coordinates directly.

## Data

- Source/schema: RCSB -> Boltz preprocessing -> `.npz` records with `residues`, `atoms`, `chains` structured arrays.
- Active data: `data/rcsb/` (~212k Boltz-style `.npz`).
- ESM cache: `data/rcsb_esm/` canonical ESM3 embeddings. `data/rcsb_esmc/` may remain for ablation but is not active.
- Splits: `data/splits/{train,val,val_casp,holdout_ids}.txt` are frozen. Do not regenerate unless all reported metrics are invalidated and rerun.
- Active filtering: `single_chain_only: true` and `extract_monomer_chains: true`
  in `configs/direct_allatom_360m.yaml`.

## Training Plan

- Phase A: direct all-atom scratch, ~390M actual params
  (d_state=128 for long-range), L=1024, `configs/direct_allatom_360m.yaml`.
- Phase B: CASP14 eval at 50k/100k checkpoints with SimpleFold-style metrics.

## Commands

- Setup: `uv sync`
- Train default: `CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG=configs/direct_allatom_360m.yaml bash scripts/train.sh`
- Resume: `CUDA_VISIBLE_DEVICES=0,1,2,3 RESUME=outputs/train/<run>/ckpt_latest.pt bash scripts/train.sh`
- Inference: `PYTHONPATH=src uv run python benchmarks/run_inference.py --ckpt <ckpt> --ids <ids.txt> --out <out_dir>`
- Score: `tools/scoring_venv/bin/python benchmarks/score_simplefold_metrics.py --in_dir <out_dir> --out <out_dir>/scores.json`
- Syntax smoke: `PYTHONPATH=src uv run python -m py_compile ...`

## Verification

- Required quick checks after code edits:
  - `py_compile` on touched Python files.
  - config parse for `configs/direct_allatom_360m.yaml`.
  - single-chain dataset smoke: multichain example filtered, single-chain example kept.
- Before long training:
  - focused tests or full `uv run pytest` when feasible.
  - GPU 0-3 free.
  - config diff reviewed.
  - W&B run name/tags set.
- Metrics:
  - Train: all-atom FM, sampled all-atom lDDT, C-alpha lDDT, distogram, bond/clash metrics.
  - Later drug-discovery-oriented metrics: pocket heavy-atom RMSD, side-chain chi accuracy, clashscore.

## Paths

- Source: `src/mambafold/`
- Active configs: `configs/direct_allatom_360m.yaml`
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
- **ESM3** (Hayes et al. 2024) — PLM embeddings as conditioning. → `data/esm.py`, model PLM proj.
- **Boltz-1** (Wohlwend et al. 2024) — Boltz-style processed RCSB `.npz` records. → `data/dataset.py` `RCSBDataset`.

## Do Not Touch Without Explicit Confirmation

- `data/rcsb/`
- `data/rcsb_esm/`, `data/rcsb_esmc/`
- `data/splits/`
- live `outputs/train/<active-run>/`
- destructive git operations

## Open Decisions

- Resolved (2026-06-24): per-atom path = AF3-style atom→token→atom but all SSM —
  BiMamba AtomEncoder/Decoder over each residue's atom slots (intra-residue),
  not attention, to keep the MambaFold identity at every level.
  `model/fold/atom_mamba.py`. d_atom=128, n_atom_layers=4.
  (Superseded the earlier interim lightweight MLP decoder.)
- Resolved (2026-06-24): pair stack kept at current budget (~2.9M / 389.6M) for the
  baseline; revisit after first run if global-fold signal is weak.
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
- Open: self-conditioning (feed previous x_hat); sampler 2nd-order/churn.
- Open: bf16 SSM at d_state=128 — confirm selective-scan state update runs fp32;
  watch grad-clip hit-rate vs the non-smooth chirality hinges.
- Open: sampled all-atom lDDT/clash budgets 1024 vs 2048 vs 4096 atoms.
