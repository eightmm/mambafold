# Inference

Inference uses two-stage Euler sampling in `src/mambafold/sampling/samplers.py`.

1. Stage 1 integrates CA coordinates from noise to a scaffold.
2. Stage 2 initializes atom slots with noise, sets CA from Stage 1, then integrates all atoms with CA residual refinement.
3. Optional recycling re-noises from an intermediate `t` and denoises again.

## Command

```bash
PYTHONPATH=src uv run python benchmarks/run_inference.py \
  --ckpt outputs/train/<run>/ckpt_latest.pt \
  --ids benchmarks/sets/t1_quick.txt \
  --out benchmarks/results/<run>_t1 \
  --n_steps 50
```

Useful knobs:

| Option | Meaning |
|---|---|
| `--n_steps` | Stage 1 Euler steps |
| `--n_steps_s2` | Stage 2 Euler steps, defaults to `--n_steps` |
| `--n_recycle` | both-stage recycle count |
| `--n_recycle_s1`, `--n_recycle_s2` | per-stage recycle override |
| `--recycle_t_start` | re-noise time for recycling |

## Scoring

```bash
tools/scoring_venv/bin/python benchmarks/score.py \
  --in_dir benchmarks/results/<run>_t1 \
  --out benchmarks/results/<run>_t1/scores.json
```

Metrics: CA lDDT, TM-score, CA RMSD, all-atom RMSD where paired GT exists.
