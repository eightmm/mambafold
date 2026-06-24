# HANDOFF — direct all-atom migration + first training run

Snapshot for the next agent. Authoritative spec is `PROJECT.md`; this file is
the *current operational state* and what's in flight. Date: 2026-06-24.

## Branch & commits

Branch: `feat/direct-allatom-folding` (off `main`). New commits this session:

- `c6f1d23` feat: direct all-atom flow-matching fold model (the migration)
- `26f34fb` fix: drop DDP `static_graph` (breaks grad-accum `no_sync`)
- `aff313e` fix: log all aux losses (`lddt_ca`/`ca_clash` were dropped from W&B)
- `3fbe5d4` feat: add SimpleFold-lean loss ablation config

Not committed (intentionally untracked, not for git): `.tmp/`, `tea_debug.log`,
`data/casp_official/`.

## What the model is now

2-stage (CA→atom) pipeline was deleted; replaced by ONE direct all-atom
flow-matching model. atom → token → atom, all-SSM:

- `src/mambafold/model/fold/atom_mamba.py` — `AtomEncoder`/`AtomDecoder`:
  BiMamba over each residue's 15 atom slots (intra-residue), `FiLM` time
  conditioning. `d_atom=128`, `n_atom_layers=4`.
- `src/mambafold/model/fold/all_atom.py` — `MambaFoldAllAtom`: AtomEncoder →
  pool to residue token → Mamba trunk (17 layers, `d_res=1024`, `d_state=128`,
  hybrid RoPE attn at layers 15-16) + triangle-mult pair stack (4 blocks) →
  AtomDecoder. `TimeEmbedding` (sinusoidal) + FiLM injects FM noise level into
  trunk input and atom enc/dec. Aux heads: distogram, contact, pseudo-CB, conf
  (pLDDT). ~389.6M params.
- Losses (`src/mambafold/train/engine.py` + `losses/`): all-atom FM velocity,
  soft lDDT (CA + sampled all-atom), bond, clash (CA + all-atom), CA distogram,
  dRMSD, long-range contact, pseudo-CB, all-atom-lDDT confidence (pLDDT),
  chirality (CA-CA-CA global handedness + per-residue N/C/CB stereochemistry).
- FM: `x_t = t*x_clean + (1-t)*eps`, target `v = x_clean - eps`. `t_schedule:
  logit_normal` (SimpleFold p(t)=0.98*LN(0.8,1.7)+0.02*U). Sampler =
  `sampling/samplers.py` Euler, now with per-step CoM re-centering.

This matches SimpleFold's recipe (FM + alpha(t)*LDDT, logit-normal, no
self-conditioning) but adds ~12 extra aux losses and swaps Transformer→Mamba.

## Training run IN FLIGHT

- Config `configs/direct_allatom_360m.yaml` (full-aux), GPUs 0-3.
- Out dir `outputs/train/direct_allatom_v1/`, log
  `outputs/train/direct_allatom_v1.log`, W&B run `direct_allatom_v1` (`l3f0z8nt`).
- Status: healthy, ~step 1700/100k as of writing, loss 14.8→4.7, FM 8.7→1.3,
  no crashes, ~47 samp/s, GPU 0-3 ~90-100%. First ckpt+val at step 5000.
- A background watcher (`b23lblei5`) polls every 25 min and reports when step
  5000 (val metrics / `ckpt_0005000.pt`) lands.
- Launch (if relaunch needed): `CUDA_VISIBLE_DEVICES=0,1,2,3 TMPDIR=$PWD/.cache/tmp
  CONFIG=configs/direct_allatom_360m.yaml OUT_DIR=outputs/train/direct_allatom_v1
  bash scripts/train.sh --wandb_name direct_allatom_v1`
- Stop safely (shared account — PID-targeted, NEVER `pkill -f`): find torchrun
  PIDs via `nvidia-smi --query-compute-apps=pid --format=csv,noheader` (GPU 0-3),
  then `kill <PID>`.

## Hard constraints / gotchas

- GPUs 0-3 ONLY (4-7 belong to another user). Cannot run two 4-GPU jobs at once.
- `TMPDIR=$PWD/.cache/tmp` is REQUIRED — TileLang mamba kernels fail to map in
  `/tmp` on this box ("failed to map segment").
- DDP: `static_graph=True` is INCOMPATIBLE with the grad-accum `model.no_sync()`
  path (triggers `reducer.cpp expect_autograd_hooks_` assert → NVLink-looking
  NCCL crash). Do not re-add it. `gradient_as_bucket_view=True` is fine.
- `logging.py` fix (`aff313e`) only applies to the NEXT run — a running process
  already imported the old module.
- First training step is slow: TileLang JIT-compiles mamba kernels per new
  length bucket. Errors surface within the first 1-2 steps after compile.

## Open / next steps (not done, with rationale)

- `configs/direct_allatom_lean.yaml` (committed) = SimpleFold-lean ablation
  (FM + alpha(t)*all-atom LDDT only, 13 aux zeroed, identical model). NOT YET
  RUN — blocked on GPU (full-aux run is using 0-3). Run it to test whether our
  12 extra aux losses actually beat the lean baseline on val lDDT + mirror-rate.
- FM-timing concern (from multi-agent review, verified, NOT applied — needs
  ablation): `x_hat = x_t+(1-t)v` makes aux gradients vanish at t→1, while
  `alpha_mode: ramp` + `logit_normal` concentrate weight/sampling there. Added a
  `w_fm` lever (config, default 1.0) to rebalance; consider aux ~1/(1-t) or a
  mid-t window, and log per-term grad norms first.
- Self-conditioning: absent (matches SimpleFold). Highest-ROI architecture add if
  pursued (feed previous x_hat, 50% train / always sample).
- `max_lddt_atoms`/`max_clash_atoms` = 2048 (subsample bounds the O((L*A)^2)
  all-atom lDDT/clash; we train full L=1024 instead of AF-style cropping). B200
  can afford 4096; PROJECT lists this as an open ablation.
- CA-loss trims (analysis done, not applied): `pcb` likely redundant now that the
  atom decoder predicts CB directly; `ca_clash`/`ca_self_clash` mutually overlap
  (keep one); `drmsd` weakest, ablate. Keep distogram/contact (pair-rep
  supervision), `lddt_ca` (dense global, all-atom lDDT is subsampled), and
  ca-chirality (complements CB-chirality — backbone helix handedness vs
  stereocenter).
- Confirmed NOT bugs: GT losses use `valid_mask` while geometry priors use
  `atom_mask` (intentional — gives unobserved modeled atoms an ideal-geometry
  signal); SO(3) augmentation is per-example; val split is MMseqs2 cluster-based
  (`scripts/make_val_split.py`, homolog-leakage guarded).
- Watch: bf16 SSM at d_state=128 — confirm selective-scan state update is fp32;
  gnorm runs ~30 (grad_clip=1.0 clamps) partly from non-smooth chirality hinges.

## Quick verification (pre-run sanity)

- `PYTHONPATH=src TMPDIR=$PWD/.cache/tmp .venv/bin/python scripts/smoke_all_atom.py`
- `PYTHONPATH=src TMPDIR=$PWD/.cache/tmp .venv/bin/python -m pytest tests/test_mamba3.py -q`
- `PYTHONPATH=src .venv/bin/python -c "from mambafold.train.config import parse_args; parse_args(['--config','configs/direct_allatom_360m.yaml'])"`

Note: Codex was editing this repo concurrently this session; the committed state
on this branch is the source of truth.
