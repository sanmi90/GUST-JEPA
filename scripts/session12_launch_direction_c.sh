#!/usr/bin/env bash
# Session 12 Direction C launcher: extended lambda_wake ladder.
#
# Usage:
#   bash scripts/session12_launch_direction_c.sh <lambda_wake> <gpu>
#
# lambda_wake: float, the wake observable loss weight (2.0, 3.0, 5.0 in Direction C).
# gpu: 0 or 1 (selector into the RTX 6000 subset; D40)
#
# Trains a JEPA encoder + predictor + observable heads from scratch with the
# Session 11 W0_C_lam100 recipe but at a higher lambda_wake than the Session 11
# max (1.00). After this finishes, run session11_launch_decoder.sh on the
# resulting encoder run dir to retrain the E1 LapFiLM decoder.
#
# Reference: SESSION12_CRISP_WAKE.md "Direction C: extended lambda_wake ladder".

set -euo pipefail

LAM=${1:?lambda_wake required (2.0, 3.0, 5.0 typical)}
GPU=${2:?gpu required (0|1)}

# Make a readable tag: lam200, lam300, lam500.
LAM_TAG="lam$(printf '%.0f' "$(echo "$LAM * 100" | bc)")"
OUT_DIR="outputs/runs/session12/S12_C_${LAM_TAG}/encoder"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/../launch.log"

cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

echo "[s12-c] lambda_wake=$LAM gpu=$GPU tag=$LAM_TAG out=$OUT_DIR" | tee "$LOG"

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
    --tag-suffix "session12_S12_C_${LAM_TAG}" \
    --observable-head cl_future --observable-head-weight 0.01 \
    --observable-head-deltas 0 \
    --wake-observable-type patch_signed_spectrum --lambda-wake "$LAM" \
    --wake-loss smooth_l1 --wake-loss-beta 0.5 --wake-head-hidden 128 \
    --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \
    --wandb-mode offline \
    >> "$LOG" 2>&1 &

echo "[s12-c] PID=$!" | tee -a "$LOG"
