# MambaFold

Single-chain direct all-atom protein structure generation with flow matching,
ESM3 conditioning, Bi-Mamba sequence modeling, and pair reasoning.

## Active Path

- Model: `src/mambafold/model/fold/all_atom.py::MambaFoldAllAtom`
- Config: `configs/direct_allatom_360m.yaml`
- Training entrypoint: `scripts/train.sh`
- Inference entrypoint: `benchmarks/run_inference.py`

## Train

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG=configs/direct_allatom_360m.yaml bash scripts/train.sh
```

Resume:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
RESUME=outputs/train/<run>/ckpt_latest.pt \
CONFIG=configs/direct_allatom_360m.yaml \
bash scripts/train.sh
```

## Inference

```bash
PYTHONPATH=src uv run python benchmarks/run_inference.py \
  --ckpt outputs/train/<run>/ckpt_latest.pt \
  --ids benchmarks/sets/t1_quick.txt \
  --out benchmarks/results/<run>_t1 \
  --esm_dir data/rcsb_esm \
  --n_steps 50
```

Score:

```bash
tools/scoring_venv/bin/python benchmarks/score_simplefold_metrics.py \
  --in_dir benchmarks/results/<run>_t1 \
  --out benchmarks/results/<run>_t1/scores.json
```

## Smoke

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src uv run python scripts/smoke_all_atom.py
PYTHONPATH=src uv run python -m py_compile src/mambafold/train/engine.py scripts/train.py
```
