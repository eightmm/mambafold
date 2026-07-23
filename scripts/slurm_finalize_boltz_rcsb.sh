#!/usr/bin/env bash
#SBATCH --job-name=boltz-rcsb-finalize
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

url="https://boltz1.s3.us-east-2.amazonaws.com/rcsb_processed_targets.tar"
expected_bytes=64948531200
snapshot="2024-12-20"
base_dir="${BOLTZ_RCSB_BASE_DIR:-/home/jaemin/DB/RCSB/processed/structures}"
destination="$base_dir/rcsb_processed_targets_full_${snapshot}"
staging="$base_dir/.rcsb_processed_targets_full_${snapshot}.staging"
extracted="$staging/rcsb_processed_targets"

if [[ ! -d "$destination" ]]; then
    if [[ ! -d "$extracted" ]]; then
        echo "Missing extracted staging directory: $extracted" >&2
        exit 1
    fi
    .venv/bin/python scripts/validate_boltz_rcsb.py \
        --root "$extracted" \
        --source-url "$url" \
        --snapshot "$snapshot" \
        --archive-bytes "$expected_bytes"
    mv "$extracted" "$destination"
    rmdir "$staging"
fi

source_dir="$destination/structures"
manifest="$destination/download_manifest.json"

if [[ ! -f "$manifest" ]]; then
    echo "Missing verified download manifest: $manifest" >&2
    exit 1
fi

python3 - "$manifest" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
count = int(manifest["paired_id_count"])
if count < 200_000:
    raise SystemExit(f"Official archive unexpectedly contains only {count} paired records")
print(f"verified_paired_records={count}", flush=True)
PY

.venv/bin/python scripts/setup_boltz_rcsb.py \
    --source-dir "$source_dir" \
    --view-dir data/rcsb_boltz_official_full \
    --alias-dir data/rcsb \
    --tag boltz_official_full
