#!/usr/bin/env bash
# Frozen ESM3 v1.0.0 inference/evaluation entrypoint. Never trains or resumes.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
manifest="$project_root/projects/esm3/manifest.json"

: "${ESM3_CHECKPOINT:?Set ESM3_CHECKPOINT to the verified ckpt_0120000.pt}"
: "${ESM3_DATA_DIR:?Set ESM3_DATA_DIR to the CASP14 .npz directory}"
: "${ESM3_EMBEDDINGS:?Set ESM3_EMBEDDINGS to the matching ESM3 cache directory}"
: "${ESM3_IDS:?Set ESM3_IDS to the frozen 70-target ID list}"
: "${ESM3_OUT:?Set ESM3_OUT to a new output directory}"

test ! -e "$ESM3_OUT"
python "$project_root/projects/esm3/verify_artifact.py" --checkpoint "$ESM3_CHECKPOINT"
test "$(wc -l < "$ESM3_IDS")" -eq 70

python - "$manifest" "$ESM3_IDS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
ids = Path(sys.argv[2]).read_text().split()
if len(ids) != manifest["evaluation"]["target_count"]:
    raise SystemExit(f"unexpected target count: {len(ids)}")
print(f"frozen project={manifest['project_id']} targets={len(ids)}")
PY

cd "$project_root"
PYTHONPATH=src python benchmarks/run_inference.py \
  --ckpt "$ESM3_CHECKPOINT" \
  --ids "$ESM3_IDS" \
  --out "$ESM3_OUT" \
  --data_dir "$ESM3_DATA_DIR" \
  --esm_dir "$ESM3_EMBEDDINGS" \
  --max_length 1024 \
  --sampler sde \
  --n_steps 500 \
  --sde_tau 0.01 \
  --sde_eps 0.01 \
  --sde_w_cutoff 0.99 \
  --sde_log_timesteps \
  --n_seeds 1 \
  --seed_offset 0

python benchmarks/score_simplefold_metrics.py \
  --in_dir "$ESM3_OUT" \
  --out "$ESM3_OUT/scores.json"
