#!/usr/bin/env bash
# Wait for the current pure-Mamba ablation queue to finish, then precompute AFDB ESM.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="outputs/esm_afdb_swissprot"
mkdir -p "$LOG_DIR"
WAIT_LOG="$LOG_DIR/wait_then_run.log"

echo "[$(date -Is)] waiting for mfold_pure_ablation queue" | tee -a "$WAIT_LOG"
while pgrep -f "bash scripts/run_puremamba_arch_ablation_queue.sh" >/dev/null; do
  echo "[$(date -Is)] queue still active" | tee -a "$WAIT_LOG"
  sleep "${WAIT_SECONDS:-300}"
done

echo "[$(date -Is)] queue finished; starting AFDB SwissProt ESM" | tee -a "$WAIT_LOG"
exec bash scripts/precompute_afdb_swissprot_esm_4gpu.sh
