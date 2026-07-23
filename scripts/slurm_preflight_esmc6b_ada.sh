#!/usr/bin/env bash
#SBATCH --job-name=mf-esmc6b-preflight
#SBATCH --partition=6000ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
bash scripts/preflight_esmc6b_ada.sh
