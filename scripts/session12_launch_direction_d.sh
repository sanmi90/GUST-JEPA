#!/usr/bin/env bash
# Session 12 Direction D launcher: higher-dimensional wake observable target.
#
# Usage:
#   bash scripts/session12_launch_direction_d.sh <variant> <gpu>
#
# variant: one of {coarse288, coarse512}.
#   coarse288: wake_coarse_pool mode (24x12 average-pooled wake ROI = 288D).
#   coarse512: wake_coarse_pool_32x16 mode (32x16 = 512D).
# gpu: 0 or 1 (selector into the RTX 6000 subset; D40)
#
# Retrains a JEPA encoder from scratch with a higher-D wake observable head.
# Same lambda_wake = 1.00 as the Session 11 W0_C_lam100 to isolate the target
# dimensionality effect. After encoder training, run
# session11_launch_decoder.sh to retrain the E1 LapFiLM decoder.
#
# Reference: SESSION12_CRISP_WAKE.md "Direction D".

set -euo pipefail

VARIANT=${1:?variant required (coarse288|coarse512)}
GPU=${2:?gpu required (0|1)}

case "$VARIANT" in
    coarse288) WAKE_TYPE=wake_coarse_pool;        TAG=coarse288 ;;
    coarse512) WAKE_TYPE=wake_coarse_pool_32x16;  TAG=coarse512 ;;
    *) echo "error: unknown variant $VARIANT" >&2; exit 2 ;;
esac

OUT_DIR="outputs/runs/session12/S12_D_${TAG}/encoder"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/../launch.log"

cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

echo "[s12-d] variant=$VARIANT wake_type=$WAKE_TYPE gpu=$GPU out=$OUT_DIR" | tee "$LOG"

nohup python -u -m src.training.train_jepa \
    --all-train \
    --max-iters 20000 \
    --seed 42 \
    --d 32 \
    --B 16 --T 32 --H-roll 8 \
    --lambda-sigreg 0.01 \
    --lr-encoder 1.5e-4 --lr-predictor 5e-4 \
    --weight-decay 0.05 --warmup-frac 0.05 \
    --num-workers 4 \
    --gpu "$GPU" \
    --output-dir "$OUT_DIR" \
    --log-every 50 --diagnostic-every 500 --checkpoint-every 2000 \
    --projection-norm batchnorm --anticollapse sigreg \
    --tag-suffix "session12_S12_D_${TAG}" \
    --observable-head cl_future --observable-head-weight 0.01 \
    --observable-head-deltas 0 \
    --wake-observable-type "$WAKE_TYPE" --lambda-wake 1.00 \
    --wake-loss smooth_l1 --wake-loss-beta 0.5 --wake-head-hidden 128 \
    --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \
    --wandb-mode offline \
    >> "$LOG" 2>&1 &

echo "[s12-d] PID=$!" | tee -a "$LOG"
