#!/usr/bin/env bash
set -euo pipefail

cd /NHNHOME/WORKSPACE/0526040024_A/jaemin/mambafold

TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
EVAL_GPU="${EVAL_GPU:-2}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29620}"
mkdir -p outputs/train outputs/eval/logs .cache/tmp

export TMPDIR="$PWD/.cache/tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=src

run_train() {
  local name="$1"
  local config="$2"
  local out_dir="outputs/train/${name}"
  local port="$3"

  if [[ -s "${out_dir}/ckpt_0020000.pt" ]]; then
    echo "[skip-train] ${name}: ckpt_0020000.pt exists"
    return
  fi

  mkdir -p "${out_dir}"
  echo "[train-start] $(date -Is) name=${name} config=${config} gpus=${TRAIN_GPUS}"
  if [[ -s "${out_dir}/ckpt_latest.pt" ]]; then
    CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
    CONFIG="${config}" \
    OUT_DIR="${out_dir}" \
    MASTER_PORT="${port}" \
    RESUME="${out_dir}/ckpt_latest.pt" \
      bash scripts/train.sh --wandb_name "${name}" \
        >> "${out_dir}.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
    CONFIG="${config}" \
    OUT_DIR="${out_dir}" \
    MASTER_PORT="${port}" \
      bash scripts/train.sh --wandb_name "${name}" \
        > "${out_dir}.log" 2>&1
  fi
  echo "[train-done] $(date -Is) name=${name}"
}

run_eval() {
  local name="$1"
  local ckpt="$2"
  local out="outputs/eval/casp14_70_${name}_sde500_20260630"

  if [[ ! -s "${ckpt}" ]]; then
    echo "[skip-eval] ${name}: missing ${ckpt}"
    return
  fi
  if [[ -s "${out}/summary.json" ]]; then
    echo "[skip-eval] ${name}: summary exists"
    return
  fi

  mkdir -p "${out}"
  echo "[eval-start] $(date -Is) name=${name} ckpt=${ckpt} gpu=${EVAL_GPU}"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" uv run python benchmarks/run_inference.py \
    --ckpt "${ckpt}" \
    --ids data/casp_official/casp14_70_whole_ids_exact.txt \
    --out "${out}" \
    --data_dir data/casp_official/npz \
    --esm_dir data/casp_official/esm \
    --max_length 2048 \
    --n_steps 500 \
    --sampler sde \
    --sde_tau 0.01 \
    --sde_eps 0.01 \
    --sde_w_cutoff 0.99 \
    --sde_log_timesteps \
    --n_seeds 1 \
      > "outputs/eval/logs/${name}_infer.log" 2>&1

  tools/scoring_venv/bin/python benchmarks/score_simplefold_metrics.py \
    --in_dir "${out}" \
    --out "${out}/scores.json" \
      > "outputs/eval/logs/${name}_score.log" 2>&1

  uv run python benchmarks/score_local_geometry.py \
    --in_dir "${out}" \
    --out "${out}/local_geometry.json" \
      > "outputs/eval/logs/${name}_geometry.log" 2>&1

  OUT="${out}" CKPT_NAME="$(basename "${ckpt}")" python - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUT"])
scores = json.loads((out / "scores.json").read_text())
geom = json.loads((out / "local_geometry.json").read_text())

def pair(metric):
    v = scores[metric]
    return f"{v['mean']:.3f}/{v['median']:.3f}"

summary = {
    "out_dir": str(out),
    "ckpt": os.environ["CKPT_NAME"],
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
  echo "[eval-done] $(date -Is) name=${name}"
}

run_pair_final_eval_if_ready() {
  local name="direct_pair_global20k"
  local ckpt="outputs/train/direct_pair_global20k_v1/ckpt_0020000.pt"
  local out="outputs/eval/casp14_70_direct_pair_global20k_sde500_20260630"

  if [[ -s "${out}/summary.json" ]]; then
    echo "[skip-pair-eval] summary exists"
    return
  fi
  if [[ ! -s "${ckpt}" ]]; then
    echo "[skip-pair-eval] final checkpoint not ready"
    return
  fi
  if pgrep -f "scripts/train.py --config configs/direct_allatom_pair_global20k.yaml" >/dev/null; then
    echo "[skip-pair-eval] pair train still running"
    return
  fi
  run_eval "${name}" "${ckpt}"
}

echo "[queue-start] $(date -Is) train_gpus=${TRAIN_GPUS} eval_gpu=${EVAL_GPU}"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits || true

run_train "direct_puremamba_lwfilm20k_v1" \
  "configs/direct_allatom_puremamba_lwfilm20k.yaml" \
  "${MASTER_PORT_BASE}"
run_pair_final_eval_if_ready
run_eval "direct_puremamba_lwfilm20k" \
  "outputs/train/direct_puremamba_lwfilm20k_v1/ckpt_0020000.pt"

run_train "direct_puremamba_lwfilm_geom20k_v1" \
  "configs/direct_allatom_puremamba_lwfilm_geom20k.yaml" \
  "$((MASTER_PORT_BASE + 1))"
run_pair_final_eval_if_ready
run_eval "direct_puremamba_lwfilm_geom20k" \
  "outputs/train/direct_puremamba_lwfilm_geom20k_v1/ckpt_0020000.pt"

run_train "direct_puremamba_lwfilm_attn6_20k_v1" \
  "configs/direct_allatom_puremamba_lwfilm_attn6_20k.yaml" \
  "$((MASTER_PORT_BASE + 2))"
run_pair_final_eval_if_ready
run_eval "direct_puremamba_lwfilm_attn6_20k" \
  "outputs/train/direct_puremamba_lwfilm_attn6_20k_v1/ckpt_0020000.pt"

run_pair_final_eval_if_ready
echo "[queue-done] $(date -Is)"
