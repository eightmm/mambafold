#!/bin/bash
# MambaFold overfit 제출 스크립트
#
# 사용법:
#   ./scripts/slurm/submit_overfit.sh [GPU_TYPE]
#
# GPU_TYPE 옵션 (기본: h100):
#   h100    → heavy    partition, gpu:h100:1       (SM_90,  30일)
#   a5000   → test     partition, gpu:a5000:1      (SM_86,  2시간)
#   2080ti  → test     partition, gpu:2080ti:1     (SM_75,  2시간)
#   6000ada → 6000ada  partition, gpu:1            (SM_89,  30일)
#
# NOTE: 6000pro_maxq (Blackwell SM_120a) 제외 — tilelang 미지원

GPU_TYPE=${1:-h100}

case "$GPU_TYPE" in
    h100)
        PARTITION=heavy
        GRES=gpu:h100:1
        ;;
    a5000)
        PARTITION=test
        GRES=gpu:a5000:1
        ;;
    2080ti)
        PARTITION=test
        GRES=gpu:2080ti:1
        ;;
    6000ada)
        PARTITION=6000ada
        GRES=gpu:1
        ;;
    *)
        echo "Unknown GPU_TYPE: $GPU_TYPE"
        echo "Usage: $0 [h100|a5000|2080ti|6000ada]"
        exit 1
        ;;
esac

echo "Submitting: partition=$PARTITION, gres=$GRES"
sbatch --partition="$PARTITION" --gres="$GRES" \
    scripts/slurm/overfit_test.sh
