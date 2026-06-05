# MambaFold

Single-chain protein structure generation with a coarse-to-fine Mamba backbone.

The active project is no longer a family of model versions. There is one current path:

1. Stage 1 predicts a C-alpha scaffold with flow matching.
2. Stage 2 refines that scaffold into all-atom coordinates.
3. Optional joint finetuning updates both stages together.

Scope is single-chain standard proteins only. Multimer/interface prediction, ligands, nucleic acids, metals, cofactors, water, PTMs, and EqM are out of scope.

## Active Files

| Area | Path |
|---|---|
| Project spec | `PROJECT.md` |
| Model | `src/mambafold/model/fold/` |
| Training engine | `src/mambafold/train/engine.py` |
| Trainer/checkpoints | `src/mambafold/train/trainer.py` |
| Sampler | `src/mambafold/sampling/samplers.py` |
| Configs | `configs/stage1.yaml`, `configs/stage1_ct.yaml`, `configs/stage2.yaml`, `configs/joint.yaml` |
| Inference/scoring | `benchmarks/run_inference.py`, `benchmarks/score.py` |

## Data

Active data is Boltz-style RCSB `.npz` records:

```text
data/rcsb/       structures
data/rcsb_esm/   ESM3 embeddings
data/splits/     frozen train/val/holdout splits
```

Training configs set `single_chain_only: true`.

## Train

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG=configs/stage1.yaml bash scripts/train.sh
```

Continue Stage 1 at longer crops:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
CONFIG=configs/stage1_ct.yaml \
RESUME=outputs/train/<stage1_run>/ckpt_latest.pt \
bash scripts/train.sh
```

Train Stage 2:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
CONFIG=configs/stage2.yaml \
STAGE1_CKPT=outputs/train/<stage1_run>/ckpt_latest.pt \
bash scripts/train.sh
```

## Infer / Score

```bash
PYTHONPATH=src uv run python benchmarks/run_inference.py \
  --ckpt outputs/train/<run>/ckpt_latest.pt \
  --ids benchmarks/sets/t1_quick.txt \
  --out benchmarks/results/<run>_t1

tools/scoring_venv/bin/python benchmarks/score.py \
  --in_dir benchmarks/results/<run>_t1 \
  --out benchmarks/results/<run>_t1/scores.json
```

## Verify

```bash
PYTHONPATH=src uv run python -m py_compile scripts/train.py src/mambafold/train/*.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src uv run python scripts/smoke_stage1.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src uv run python scripts/smoke_stage2.py
```
