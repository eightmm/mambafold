#!/usr/bin/env bash
#SBATCH --job-name=mf-lddt
#SBATCH --partition=6000ada
#SBATCH --qos=veryshort
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export PYTHONPATH=src
.venv/bin/python benchmarks/probe_exact_lddt.py \
  --atom-counts 2048 4096 8192 \
  --batch-size 8 \
  --chunk-size 512 \
  --density 0.05 \
  --repeats 2 \
  --device cuda
