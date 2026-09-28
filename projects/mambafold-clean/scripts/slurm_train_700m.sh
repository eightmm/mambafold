#!/usr/bin/env bash
#SBATCH --job-name=mf-train-700m
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --qos=verylong
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/slurm/mf-train-700m-%j.out
#SBATCH --error=logs/slurm/mf-train-700m-%j.out

set -euo pipefail

# Compound-scaled successor to the 370.48M Run A model.  The wider PLM
# projection preserves the full-trunk conditioning contract.  State size,
# MIMO rank, atom path, data, objective, and optimizer stay fixed so the first
# scale-up isolates residue-trunk capacity.  This configuration has exactly
# 670,119,418 parameters.
export MF_CONFIG="${MF_CONFIG:-configs/run_a_mamba.yaml}"
export MF_OUT_DIR="${MF_OUT_DIR:-out/run-a-mamba-700m}"

exec scripts/slurm_train.sh \
  --d_res 1152 \
  --n_trunk 22 \
  --d_plm_proj 1152 \
  "$@"
