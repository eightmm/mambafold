#!/usr/bin/env bash
#SBATCH --job-name=mf-esmc6b-seqcache
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export PYTHONPATH="${PYTHONPATH:-src}"

uv run --no-sync python -u scripts/migrate_esm_cache_by_sequence.py \
    --data_dir data/rcsb_boltz_official_full \
    --esm_dir data/rcsb_esmc6b_official_full \
    --max_length 1024 \
    --embedding_dim 2560 \
    --fail_on_error

uv run --no-sync python -u scripts/migrate_esm_cache_by_sequence.py \
    --data_dir data/afdb_swissprot/npz \
    --esm_dir data/afdb_swissprot_esmc6b \
    --single_chain_fasta data/afdb_swissprot/sequences.fasta \
    --max_length 1024 \
    --embedding_dim 2560 \
    --fail_on_error
