#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

repo="biohub/ESMC-6B"
revision="45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a"
cache_dir="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"

snapshot="$(uv run --no-sync hf download "$repo" \
    --revision "$revision" \
    --cache-dir "$cache_dir" \
    --max-workers 6 \
    --quiet)"

uv run --no-sync python - "$snapshot" "$revision" <<'PY'
import json
import sys
from pathlib import Path

snapshot = Path(sys.argv[1])
revision = sys.argv[2]
if snapshot.name != revision:
    raise SystemExit(f"Unexpected snapshot revision: {snapshot}")

config = json.loads((snapshot / "config.json").read_text())
expected = {"d_model": 2560, "n_heads": 40, "n_layers": 80}
actual = {key: config.get(key) for key in expected}
if actual != expected:
    raise SystemExit(f"Unexpected ESMC-6B config: {actual}")

index = json.loads((snapshot / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))
missing = [name for name in shards if not (snapshot / name).is_file()]
if missing:
    raise SystemExit(f"Incomplete ESMC-6B snapshot: {missing}")

actual_bytes = sum((snapshot / name).stat().st_size for name in shards)
tensor_bytes = int(index["metadata"]["total_size"])
if actual_bytes < tensor_bytes:
    raise SystemExit(
        f"ESMC-6B weight files are truncated: tensor_bytes={tensor_bytes} "
        f"file_bytes={actual_bytes}"
    )
print(f"snapshot={snapshot}")
print(f"revision={revision}")
print(
    f"weight_shards={len(shards)} tensor_bytes={tensor_bytes} "
    f"file_bytes={actual_bytes}"
)
PY
