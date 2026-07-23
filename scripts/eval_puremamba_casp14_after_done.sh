#!/usr/bin/env bash
set -euo pipefail

cd /NHNHOME/WORKSPACE/0526040024_A/jaemin/mambafold

RUN_DIR="outputs/train/direct_puremamba_global20k_v1"
CKPT="${RUN_DIR}/ckpt_0020000.pt"
OUT_DIR="outputs/eval/casp14_70_direct_puremamba_global20k_sde500_20260630"
LOG_DIR="outputs/eval/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}" .cache/tmp

echo "[wait] checkpoint: ${CKPT}"
while [[ ! -s "${CKPT}" ]]; do
  date '+[wait] %Y-%m-%d %H:%M:%S'
  tail -n 5 "${RUN_DIR}.log" || true
  sleep 120
done

echo "[wait] checkpoint exists; waiting for pure-Mamba train process to exit"
while pgrep -f "scripts/train.py --config configs/direct_allatom_puremamba_global20k.yaml" >/dev/null; do
  sleep 60
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export TMPDIR="$PWD/.cache/tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=src

echo "[infer] ckpt=${CKPT}"
echo "[infer] out=${OUT_DIR}"
uv run python benchmarks/run_inference.py \
  --ckpt "${CKPT}" \
  --ids data/casp_official/casp14_70_whole_ids_exact.txt \
  --out "${OUT_DIR}" \
  --data_dir data/casp_official/npz \
  --esm_dir data/casp_official/esm \
  --max_length 2048 \
  --n_steps 500 \
  --sampler sde \
  --sde_tau 0.01 \
  --sde_eps 0.01 \
  --sde_w_cutoff 0.99 \
  --sde_log_timesteps \
  --n_seeds 1

echo "[score] SimpleFold-style metrics"
tools/scoring_venv/bin/python benchmarks/score_simplefold_metrics.py \
  --in_dir "${OUT_DIR}" \
  --out "${OUT_DIR}/scores.json"

echo "[score] local geometry"
uv run python benchmarks/score_local_geometry.py \
  --in_dir "${OUT_DIR}" \
  --out "${OUT_DIR}/local_geometry.json"

python - <<'PY'
import json
from pathlib import Path

out = Path("outputs/eval/casp14_70_direct_puremamba_global20k_sde500_20260630")
scores = json.loads((out / "scores.json").read_text())
geom = json.loads((out / "local_geometry.json").read_text())

def pair(metric):
    v = scores[metric]
    return f"{v['mean']:.3f}/{v['median']:.3f}"

summary = {
    "out_dir": str(out),
    "n": scores["n"],
    "tm_score": pair("tm_score"),
    "gdt_ts": pair("gdt_ts"),
    "lddt": pair("lddt"),
    "lddt_ca": pair("lddt_ca"),
    "rmsd": pair("rmsd"),
    "aa_rmsd": pair("aa_rmsd"),
    "pred_bond_mae_A_mean": round(geom["pred_bond_mae_A_mean"], 4),
    "pred_clashes_per_1k_atoms_mean": round(geom["pred_clashes_per_1k_atoms_mean"], 3),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "[done] ${OUT_DIR}"
