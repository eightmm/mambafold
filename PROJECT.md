# PROJECT.md

## Status

- State: confirmed
- Last confirmed by: sjm0775@snu.ac.kr
- Last updated: 2026-06-05

## Project

- Name: MambaFold
- Type: ml
- Goal: thesis/paper-grade single-chain all-atom protein structure generation with a Seed3D-style coarse-to-fine decomposition: residue-level C-alpha scaffold first, atom-level conditional refinement second.
- Core claim: decouple global fold generation from atomistic detail recovery. Stage 1 learns C-alpha backbone topology with Bi-Mamba/Mamba-3 sequence modeling; Stage 2 uses the predicted scaffold to recover chemically plausible all-atom structure.
- Users/workflow: solo local/HPC training on 4x NVIDIA B200; W&B for tracking; Boltz-style processed RCSB data for comparability.
- Scope: single-chain proteins, standard amino acids, L <= 1024 active training. MSA-free / PLM-conditioned path using ESM3 cache.
- Non-goals: multimer/interface prediction, ligands, nucleic acids, metals, cofactors, water, non-standard residues/PTMs, EqM.

## Architecture

- Stage 1: topology-preserving C-alpha scaffold generator (flow-matching Bi-Mamba).
  - Input: sequence/residue features, residue index, optional ESM3, FM time/noise.
  - Output: C-alpha velocity, Stage 1 residue latent, pseudo-Cβ unit direction
    (side-chain orientation cue), per-residue confidence (predicted Cα-lDDT).
  - Losses: C-alpha FM, soft C-alpha lDDT, dRMSD (distance-map), C-alpha bond +
    virtual-angle floor + self-clash geometry, distogram + long-range contact aux,
    pseudo-Cβ cosine, confidence calibration.
  - Recycling: training-time 2-cycle — cycle-1 Cα distance map fed back into the pair
    init for cycle-2 (earlier cycle under no_grad; `n_cycles_train`).
  - Purpose: global topology, long-range fold, domain/backbone arrangement; emit a
    scaffold (Cα + orientation + calibrated confidence) Stage 2 can refine.
- Stage 2: conditional all-atom refiner.
  - Input: sequence/residue/atom features, Stage 1 C-alpha scaffold, pseudo-Cβ direction,
    confidence, Stage 1 latent, FM time/noise.
  - Output: atom-slot velocity for atom14-style coordinates.
  - Losses: non-CA atom FM, all-atom/CA lDDT, bond geometry, clash, confidence-weighted
    C-alpha anchor (high-confidence cores anchored hard, uncertain loops free to move).
  - Purpose: backbone atoms, side-chain placement, local stereochemistry, clash reduction.
- C-alpha policy: Stage 2 may move C-alpha locally, but is anchored to Stage 1 by `w_ca_anchor`. This avoids fixed-CA error lock-in while limiting fold drift.
- Robust conditioning: Stage 2 training can inject small noise into the Stage 1 C-alpha condition (`ca_condition_noise_std`, `ca_condition_noise_prob`) so it learns to handle predicted/noisy scaffolds, not only perfect anchors.

## Rationale

- Protein structure naturally factorizes as `p(all_atom | sequence) ~= p(CA | sequence) * p(all_atom | CA, sequence)`.
- C-alpha prediction has much lower output dimension than all-atom prediction, giving cleaner global fold supervision.
- Bi-Mamba is best used for sequence-scale/global context in Stage 1; Stage 2 gets a 3D scaffold before learning local atomistic detail.
- This mirrors coarse-to-fine 3D generation: coarse shape first, high-frequency chemical detail second.
- Main risks: Stage 1 error lock-in, C-alpha orientation ambiguity, Stage 2 distribution shift if trained only on perfect C-alpha.
- Mitigations: distogram auxiliary in Stage 1, C-alpha residual refinement with anchor in Stage 2, noisy/predicted C-alpha conditioning.

## Data

- Source/schema: RCSB -> Boltz preprocessing -> `.npz` records with `residues`, `atoms`, `chains` structured arrays.
- Active data: `data/rcsb/` (~212k Boltz-style `.npz`).
- ESM cache: `data/rcsb_esm/` canonical ESM3 embeddings. `data/rcsb_esmc/` may remain for ablation but is not active.
- Splits: `data/splits/{train,val,val_casp,holdout_ids}.txt` are frozen. Do not regenerate unless all reported metrics are invalidated and rerun.
- Active filtering: `single_chain_only: true` in stage configs.

## Training Plan

- Phase 1a: Stage 1 PT, L=512, `configs/stage1.yaml`.
- Phase 1b: Stage 1 CT, L=1024, `configs/stage1_ct.yaml`.
- Phase 2: Stage 2 all-atom refiner, frozen Stage 1, `configs/stage2.yaml`.
- Phase 3: optional joint finetune, `configs/joint.yaml`.
- Stage 2 should not be trained only on GT C-alpha. It should condition on Stage 1 predicted C-alpha plus optional injected noise.

## Commands

- Setup: `uv sync`
- Train default: `CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG=configs/stage1.yaml bash scripts/train.sh`
- Resume: `CUDA_VISIBLE_DEVICES=0,1,2,3 RESUME=outputs/train/<run>/ckpt_latest.pt bash scripts/train.sh`
- Stage 2: `CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG=configs/stage2.yaml STAGE1_CKPT=outputs/train/<stage1>/ckpt_latest.pt bash scripts/train.sh`
- Inference: `PYTHONPATH=src uv run python benchmarks/run_inference.py --ckpt <ckpt> --ids <ids.txt> --out <out_dir>`
- Score: `tools/scoring_venv/bin/python benchmarks/score.py --in_dir <out_dir> --out <out_dir>/scores.json`
- Syntax smoke: `PYTHONPATH=src uv run python -m py_compile ...`

## Verification

- Required quick checks after code edits:
  - `py_compile` on touched Python files.
  - config parse for stage1/stage1_ct/stage2/joint.
  - single-chain dataset smoke: multichain example filtered, single-chain example kept.
- Before long training:
  - focused tests or full `uv run pytest` when feasible.
  - GPU 0-3 free.
  - config diff reviewed.
  - W&B run name/tags set.
- Metrics:
  - Stage 1: C-alpha lDDT, C-alpha RMSD, distogram/bond losses.
  - Stage 2: C-alpha lDDT/TM/RMSD, all-atom RMSD, bond/clash metrics.
  - Later drug-discovery-oriented metrics: pocket heavy-atom RMSD, side-chain chi accuracy, clashscore.

## Paths

- Source: `src/mambafold/`
- Active configs: `configs/stage1.yaml`, `configs/stage1_ct.yaml`, `configs/stage2.yaml`, `configs/joint.yaml`
- Train scripts: `scripts/train.py`, `scripts/train.sh`
- Benchmarks: `benchmarks/run_inference.py`, `benchmarks/run_stage1_ca_eval.py`, `benchmarks/score.py`
- Outputs/checkpoints: `outputs/train/<run>/`

## References (borrowed concepts)

Papers whose ideas are used, and where in the code.

- **Seed3D 2.0** (ByteDance Seed, tech report) — coarse-to-fine 2-stage decomposition
  (global shape → high-frequency detail). → our Stage 1 (Cα scaffold) → Stage 2 (all-atom).
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
  encodings, confidence head framing. → `model/embeddings.py` `SequenceFourierEmbedder`,
  Stage 1 `conf_head`.
- **SeedFold** (arXiv:2512.24354v1) — Linear Triangle Attention (ReLU feature map +
  associative reorder, gated; O(L³)→O(L²)). → `model/fold/linear_tri_attn.py`
  (toggle `pair_use_tri_attn`; currently OFF under the Pairmixer preset).
- **"Triangle Multiplication is All You Need" / Pairmixer** (arXiv:2510.18870) — drop
  triangle attention, keep triangle multiplication (+FFN). → active pair-stack preset
  (`pair_use_tri_attn: false` in all configs).
- **NVIDIA Nemotron-H** (hybrid Mamba-Transformer) — interleave a few self-attention
  layers among Mamba layers. → `MambaStack` `attn_layers` / `AttnBlock`
  (`trunk_attn_layers: [10,11]`).
- **GAU** (Hua et al. 2022, "Transformer Quality in Linear Time") / **Qwen Gated Attention**
  (2025) — sigmoid gate on the attention output. → `GatedSelfAttention`.
- **LayerScale** (Touvron et al. 2021) / **Flamingo** tanh-gating (Alayrac et al. 2022) —
  zero-init per-channel residual gate so a new sublayer starts as identity. → `AttnBlock`
  `attn_scale` ("AttnResidual").
- **T5** (Raffel et al. 2020) — bucketed relative-position bias. → `RelativePositionBias`.
- **lDDT** (Mariani et al. 2013) — soft differentiable lDDT loss. → `losses/lddt.py`,
  `losses/ca_only.py`.
- **Engh & Huber 2001** — ideal covalent bond lengths. → `losses/geometry.py`.
- **ESM3** (Hayes et al. 2024) — PLM embeddings as conditioning. → `data/esm.py`, Stage 1 PLM proj.
- **Boltz-1** (Wohlwend et al. 2024) — Boltz-style processed RCSB `.npz` records. → `data/dataset.py` `RCSBDataset`.

## Do Not Touch Without Explicit Confirmation

- `data/rcsb/`
- `data/rcsb_esm/`, `data/rcsb_esmc/`
- `data/splits/`
- live `outputs/train/<active-run>/`
- destructive git operations

## Open Decisions

- Whether Stage 1 should explicitly predict residue frames / pseudo-CB directions in addition to C-alpha.
- Whether Stage 2 should add an explicit C-alpha kNN geometry block beyond current CA anchor Fourier conditioning.
- Whether to run Phase 3 joint finetune after Stage 2 plateau.
