#!/usr/bin/env bash
#SBATCH --job-name=boltz-rcsb-download
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

url="https://boltz1.s3.us-east-2.amazonaws.com/rcsb_processed_targets.tar"
expected_bytes=64948531200
snapshot="2024-12-20"
base_dir="${BOLTZ_RCSB_BASE_DIR:-/home/jaemin/DB/RCSB/processed/structures}"
archive_dir="$base_dir/archives"
archive="$archive_dir/rcsb_processed_targets_${snapshot}.tar"
partial="$archive.part"
staging="$base_dir/.rcsb_processed_targets_full_${snapshot}.staging"
destination="$base_dir/rcsb_processed_targets_full_${snapshot}"

mkdir -p "$archive_dir"

if [[ -e "$destination" ]]; then
    echo "Refusing to replace existing destination: $destination" >&2
    exit 1
fi
if [[ -e "$staging" ]]; then
    echo "Refusing to reuse partial extraction staging: $staging" >&2
    exit 1
fi

if [[ ! -e "$archive" ]]; then
    wget -c --progress=dot:giga -O "$partial" "$url"
    actual_bytes="$(stat -c '%s' "$partial")"
    if [[ "$actual_bytes" != "$expected_bytes" ]]; then
        echo "Archive size mismatch: expected=$expected_bytes actual=$actual_bytes" >&2
        exit 1
    fi
    mv "$partial" "$archive"
fi

actual_bytes="$(stat -c '%s' "$archive")"
if [[ "$actual_bytes" != "$expected_bytes" ]]; then
    echo "Archive size mismatch: expected=$expected_bytes actual=$actual_bytes" >&2
    exit 1
fi

mkdir -p "$staging"
tar -xf "$archive" -C "$staging"
extracted="$staging/rcsb_processed_targets"
if [[ ! -f "$extracted/manifest.json" || ! -d "$extracted/structures" ]]; then
    echo "Unexpected archive layout under: $staging" >&2
    exit 1
fi

.venv/bin/python scripts/validate_boltz_rcsb.py \
    --root "$extracted" \
    --source-url "$url" \
    --snapshot "$snapshot" \
    --archive-bytes "$expected_bytes"

mv "$extracted" "$destination"
rmdir "$staging"
echo "destination=$destination"

.venv/bin/python scripts/setup_boltz_rcsb.py \
    --source-dir "$destination/structures" \
    --view-dir data/rcsb_boltz_official_full \
    --alias-dir data/rcsb \
    --tag boltz_official_full
