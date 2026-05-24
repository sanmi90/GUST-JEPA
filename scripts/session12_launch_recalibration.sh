#!/usr/bin/env bash
# Session 12 W0_C_lam100_v1.4 recalibration: retrain the Session 11 W0_C_lam100
# recipe on the post-D89 65-case split. Isolates lambda_wake from data-shift
# effects in the Session 12 Direction C ladder.
#
# Usage:
#   bash scripts/session12_launch_recalibration.sh <gpu>

set -euo pipefail

GPU=${1:?gpu required (0|1)}

OUT_DIR="outputs/runs/session12/W0_C_lam100_v1p4/encoder"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/../launch.log"

cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

echo "[recal] W0_C_lam100_v1.4 (lambda_wake=1.0 on 65-case split) gpu=$GPU out=$OUT_DIR" | tee "$LOG"

nohup python -u -m src.training.train_jepa \
    --all-train --max-iters 20000 --seed 42 \
    --d 32 --B 16 --T 32 --H-roll 8 \
    --lambda-sigreg 0.01 \
    --lr-encoder 1.5e-4 --lr-predictor 5e-4 \
    --weight-decay 0.05 --warmup-frac 0.05 \
    --num-workers 4 \
    --gpu "$GPU" \
    --output-dir "$OUT_DIR" \
    --log-every 50 --diagnostic-every 500 --checkpoint-every 2000 \
    --projection-norm batchnorm --anticollapse sigreg \
    --tag-suffix "session12_W0_C_lam100_v1p4" \
    --observable-head cl_future --observable-head-weight 0.01 \
    --observable-head-deltas 0 \
    --wake-observable-type patch_signed_spectrum --lambda-wake 1.00 \
    --wake-loss smooth_l1 --wake-loss-beta 0.5 --wake-head-hidden 128 \
    --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \
    --wandb-mode offline \
    >> "$LOG" 2>&1 &

echo "[recal] PID=$!" | tee -a "$LOG"
