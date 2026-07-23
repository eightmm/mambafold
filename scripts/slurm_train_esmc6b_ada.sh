#!/usr/bin/env bash
#SBATCH --job-name=mf-esmc6b-train
#SBATCH --partition=6000ada
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

config="configs/direct_allatom_puremamba_attn6_geo_adaln_sf360_esmc6b.yaml"
out_dir="outputs/train/direct_puremamba_attn6_geo_adaln_sf360_esmc6b_ada_dstate64_gpu4_v1"

# This is an intentionally new ESMC-6B run. Refuse to overwrite or silently
# resume any existing run, especially the retained ESM3 checkpoint series.
if [[ -e "$out_dir/config.json" || -e "$out_dir/ckpt_latest.pt" ]]; then
    echo "Refusing to overwrite existing run: $out_dir" >&2
    exit 2
fi
unset RESUME

bash scripts/preflight_esmc6b_ada.sh
CONFIG="$config" OUT_DIR="$out_dir" bash scripts/train.sh
