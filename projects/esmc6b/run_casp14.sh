#!/usr/bin/env bash
# Provisional ESMC-6B 170k baseline inference/evaluation. Never trains or resumes.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
manifest="$project_root/projects/esmc6b/manifest.json"
python_bin="${PYTHON_BIN:-$project_root/.venv/bin/python}"

: "${ESMC6B_CHECKPOINT:?Set ESMC6B_CHECKPOINT to mambafold-esmc6b-170k-ema.pt}"
: "${ESMC6B_DATA_DIR:?Set ESMC6B_DATA_DIR to the CASP14 .npz directory}"
: "${ESMC6B_EMBEDDINGS:?Set ESMC6B_EMBEDDINGS to the pinned ESMC-6B cache directory}"
: "${ESMC6B_IDS:?Set ESMC6B_IDS to the 70-target whole-chain ID list}"
: "${ESMC6B_OUT:?Set ESMC6B_OUT to a new output directory}"

test ! -e "$ESMC6B_OUT"
test -x "$python_bin"
"$python_bin" "$project_root/projects/esmc6b/verify_artifact.py" \
  --checkpoint "$ESMC6B_CHECKPOINT"

"$python_bin" - "$manifest" "$ESMC6B_IDS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
ids = Path(sys.argv[2]).read_text().split()
expected = manifest["evaluation"]["target_count"]
if len(ids) != expected:
    raise SystemExit(f"unexpected target count: expected={expected} actual={len(ids)}")
actual_sha256 = hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
expected_sha256 = manifest["evaluation"]["target_list_sha256"]
if actual_sha256 != expected_sha256:
    raise SystemExit(
        f"target-list SHA-256 mismatch: expected={expected_sha256} "
        f"actual={actual_sha256}"
    )
print(
    f"provisional project={manifest['project_id']} targets={len(ids)} "
    f"status={manifest['evaluation']['status']}"
)
PY

cd "$project_root"
PYTHONPATH=src "$python_bin" benchmarks/run_inference.py \
  --ckpt "$ESMC6B_CHECKPOINT" \
  --ids "$ESMC6B_IDS" \
  --out "$ESMC6B_OUT" \
  --data_dir "$ESMC6B_DATA_DIR" \
  --esm_dir "$ESMC6B_EMBEDDINGS" \
  --max_length 1024 \
  --sampler sde \
  --n_steps 500 \
  --sde_tau 0.01 \
  --sde_eps 0.01 \
  --sde_w_cutoff 0.99 \
  --sde_log_timesteps \
  --geometry_guidance_scale 0 \
  --n_seeds 1 \
  --seed_offset 0

"$python_bin" benchmarks/score_simplefold_metrics.py \
  --in_dir "$ESMC6B_OUT" \
  --out "$ESMC6B_OUT/scores.json"
