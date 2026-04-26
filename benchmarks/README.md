# MambaFold benchmarks

End-to-end evaluation pipeline: predict structures for a held-out set of PDB
IDs and score them with monomer + multimer metrics. Holdout is the
date-cutoff split (`data/splits/holdout_ids.txt`) so all targets were
deposited after the training cutoff.

## Quickstart

```bash
# 1. one-time: select tiered eval sets (deterministic, seed=0)
.venv/bin/python benchmarks/select_eval_set.py
# writes benchmarks/sets/{t0_smoke, t1_quick, t2_full}.txt + manifest.tsv

# 2. one-time: scoring tools (isolated venv, numpy<2 + DockQ + tmtools + biotite)
#    already created at tools/scoring_venv/

# 3. predict + score for one ckpt × one tier
bash benchmarks/run_eval.sh outputs/train/<run>/ckpt_latest.pt t1_quick 0

# 4. compare two runs side-by-side
.venv/bin/python benchmarks/compare.py \
    benchmarks/results/<phaseA>_t1_quick/scores.json \
    benchmarks/results/<phaseB>_t1_quick/scores.json \
    --out benchmarks/results/compare_A_B.md
```

## Tiers

| tier | max L | mono | multi | use case |
|---|---|---|---|---|
| `t0_smoke` |  512 |  10 |  10 | minute-scale sanity (CI / new ckpt arrived?) |
| `t1_quick` | 1024 |  50 |  50 | hour-scale standard eval (Phase 2 / Phase 3 comparable) |
| `t2_full`  | 2048 | 150 | 150 | half-day full eval (Phase 3 only — needs crop=2048 model) |

Each tier is fixed (seed=0 sample) so different ckpts evaluate on identical
targets — DockQ / lDDT differences reflect model quality only.

## Metrics

For every target the scorer writes:

- **`ca_lddt`** — Cα-LDDT, cutoff 15 Å, thresholds 0.5 / 1 / 2 / 4 Å.
  Standard for fold quality, monomer + multimer.
- **`tm_score`** — TM-score on Cα via TM-align (tmtools).
  - Monomer: whole-chain TM-score (canonical).
  - Multimer: **per-chain TM-score, averaged** — measures intra-chain fold quality.
    Bulk-concatenated TM mixes fold quality with relative pose, which DockQ
    captures separately.
- **`interface_lddt`** (multimer only) — LDDT restricted to inter-chain
  residue pairs. Low-cost interface signal that complements DockQ.
- **`dockq`** (multimer only) — DockQ 2.x via `run_on_all_native_interfaces`,
  reported as mean and best across interfaces. Captures relative pose, fnat,
  iRMS, LRMS.
- **`ca_rmsd`** — naive RMSD on Cα (after the inference step's Kabsch align).

## Layout

```
benchmarks/
├── select_eval_set.py    # holdout → t0/t1/t2 deterministic selection
├── run_inference.py      # ckpt + ids → multichain PDBs (+ GT pair)
├── score.py              # PDB pairs → scores.json (uses scoring_venv)
├── compare.py            # two scores.json → markdown report
├── run_eval.sh           # orchestration: 1 ckpt × 1 tier
├── README.md
├── sets/                 # tier id lists + manifest.tsv
└── results/<run_label>/
    ├── inference.log
    ├── score.log
    ├── manifest.json     # {ckpt, ids, n_predicted, ...}
    ├── summary.txt
    ├── scores.json       # consumed by compare.py
    ├── <pid>_pred.pdb        # canonical prediction (= seed-0 of multi-seed)
    ├── <pid>_pred_seed{i}.pdb
    └── <pid>_gt.pdb          # ground-truth, same chain ordering as pred
```

## Two venvs, why

- `.venv/` (project, uv-managed): torch, numpy 2.x, training stack. Used for
  inference (`run_inference.py`).
- `tools/scoring_venv/` (separate, uv venv): numpy<2 + DockQ + tmtools +
  biotite. DockQ pins numpy<2; mixing it into the training env would force a
  global downgrade and risk breaking torch CUDA builds.

Both are uv-managed but point at different lockfiles — they don't interfere.

## Notes / caveats

- **Multi-chain PDB output.** `save_pdb_multichain` maps integer
  `chain_id` (0..n-1) → letters `A..Z, a..z, 0..9`. ProteinExample residues
  are already chain-sorted by RCSBDataset, so a single linear pass with TER
  records is enough. Pred and GT use identical chain order, so DockQ's
  default mapping works.
- **Multi-seed.** `run_inference.py --n_seeds N` writes `<pid>_pred_seed{i}.pdb`;
  `score.py` currently reads only `<pid>_pred.pdb` (= seed 0). Consume more
  seeds explicitly if you want best-of-N.
- **No-ESM ckpts.** Pass `--esm_dir` only for ckpts trained with PLM features.
  Phase 1/2/3 are no-ESM, so leave it unset.
- **OOM.** Inference catches CUDA OOM per-target and skips. Scoring is CPU
  only; DockQ on very large complexes can take minutes (no timeout enforced).

## Future work (not in this setup)

- **CASP15-multimer official targets** — needed for direct comparison to
  AF-Multimer / Boltz / Chai-1 published numbers. Holdout split is a fine
  internal benchmark but not a published baseline.
- **CAMEO weekly automation** — recurring eval on freshly-deposited PDB
  entries. Good for regression tracking once a model is "released".
- **W&B logging** — push the per-run summary to the training W&B run as a
  table for ckpt-vs-ckpt history.
