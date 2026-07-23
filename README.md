# MambaFold

Single-chain direct all-atom protein structure generation with flow matching,
sequence-only ESMC-6B conditioning, and a pair-free Bi-Mamba sequence trunk
with sparse gated attention.

## Model tracks

MambaFold keeps model tracks separate because the conditioning embeddings have
different widths and their checkpoints are not interchangeable.

| Track | Status | Conditioning | Config | Published evaluation |
| --- | --- | --- | --- | --- |
| **ESM3** | **frozen inference/evaluation project** | ESM3-open, 1536-d embeddings | `projects/esm3/` | [CASP14 70-target report](projects/esm3/README.md) |
| **ESMC-6B** | active next-generation training path | pinned sequence-only ESMC-6B, 2560-d embeddings | `configs/direct_allatom_puremamba_attn6_geo_adaln_sf360_esmc6b.yaml` | no completed training checkpoint yet |

The ESM3 result is a frozen historical project: no additional ESM3 training,
fine-tuning, architecture changes, or result replacement is permitted. Its
release manifest pins the exact checkpoint hash, saved training configuration,
and CASP14 protocol. It must not be resumed with the ESMC configuration: the
PLM projection is shape-incompatible. Model weights and processed datasets are
intentionally not committed to Git; see the project package for provenance and
reproduction requirements.

## Frozen ESM3 project

The ESM3 project is inference/evaluation only. Start with its artifact check:

```bash
python projects/esm3/verify_artifact.py \
  --checkpoint /path/to/ckpt_0120000.pt
```

Then use [`projects/esm3/run_casp14.sh`](projects/esm3/run_casp14.sh) with
explicit checkpoint, CASP14 `.npz`, and ESM3 embedding locations. The current
immutable interface release is identified by Git tag `esm3-v1.1.0`.

## Active Path

- Model: `src/mambafold/model/fold/all_atom.py::MambaFoldAllAtom`
- Config: `configs/direct_allatom_puremamba_attn6_geo_adaln_sf360_esmc6b.yaml`
- Training entrypoint: `scripts/train.sh`
- Inference entrypoint: `benchmarks/run_inference.py`

## Train

Set up the pinned CUDA 13 Mamba-3/TileLang environment:

```bash
MAMBA_SKIP_CUDA_BUILD=TRUE uv sync --extra dev
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CONFIG=configs/direct_allatom_puremamba_attn6_geo_adaln_sf360_esmc6b.yaml bash scripts/train.sh
```

The ESMC config starts a new run at
`outputs/train/direct_puremamba_attn6_geo_adaln_sf360_esmc6b_ada_dstate64_v1`.
It does not resume or overwrite the retained ESM3 checkpoint. On Slurm:

```bash
sbatch scripts/slurm_train_esmc6b_ada.sh
```

Resume:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
RESUME=outputs/train/<run>/ckpt_latest.pt \
CONFIG=configs/direct_allatom_puremamba_attn6_geo_adaln_sf360_esmc6b.yaml \
bash scripts/train.sh
```

## Inference

```bash
PYTHONPATH=src uv run python benchmarks/run_inference.py \
  --ckpt outputs/train/<run>/ckpt_latest.pt \
  --ids benchmarks/sets/t1_quick.txt \
  --out benchmarks/results/<run>_t1 \
  --esm_dir data/rcsb_esmc6b_official_full \
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
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src uv run torchrun --standalone --nproc_per_node=2 scripts/smoke_esmc6b_ddp.py --batch-size 10 --length 1024 --grad-accum 2
PYTHONPATH=src uv run python -m py_compile src/mambafold/train/engine.py scripts/train.py
```
