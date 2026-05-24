#!/usr/bin/env bash
# Session 12 Direction F launcher: total-correlation penalty.
#
# Usage:
#   bash scripts/session12_launch_direction_f.sh <lambda_tc> <gpu>
#
# lambda_tc: float, the total-correlation loss weight (0.01, 0.03, 0.10 in
#            Direction F).
# gpu: 0 or 1 (selector into the RTX 6000 subset; D40)
#
# Retrains a JEPA encoder from scratch with the W0_C_lam100 recipe PLUS the
# off-diagonal-covariance TC penalty on the SIGReg-projected latent z.
# Tests whether decorrelation pressure broadens the effective dimensionality
# toward d=32 (Session 11 PR(z) was 11.66; TC may push it higher).
#
# Reference: SESSION12_CRISP_WAKE.md "Direction F".

set -euo pipefail

LTC=${1:?lambda_tc required (0.01, 0.03, 0.10 typical)}
GPU=${2:?gpu required (0|1)}

# Tag: TC0p01, TC0p03, TC0p10.
LTC_TAG="TC$(printf '%.2f' "$LTC" | tr '.' 'p')"
OUT_DIR="outputs/runs/session12/S12_F_${LTC_TAG}/encoder"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/../launch.log"

cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

echo "[s12-f] lambda_tc=$LTC gpu=$GPU tag=$LTC_TAG out=$OUT_DIR" | tee "$LOG"

nohup python -u -m src.training.train_jepa \
    --all-train \
    --max-iters 20000 \
    --seed 42 \
    --d 32 \
    --B 16 --T 32 --H-roll 8 \
    --lambda-sigreg 0.01 \
    --total-correlation-weight "$LTC" \
    --lr-encoder 1.5e-4 --lr-predictor 5e-4 \
    --weight-decay 0.05 --warmup-frac 0.05 \
    --num-workers 4 \
    --gpu "$GPU" \
    --output-dir "$OUT_DIR" \
    --log-every 50 --diagnostic-every 500 --checkpoint-every 2000 \
    --projection-norm batchnorm --anticollapse sigreg \
    --tag-suffix "session12_S12_F_${LTC_TAG}" \
    --observable-head cl_future --observable-head-weight 0.01 \
    --observable-head-deltas 0 \
    --wake-observable-type patch_signed_spectrum --lambda-wake 1.00 \
    --wake-loss smooth_l1 --wake-loss-beta 0.5 --wake-head-hidden 128 \
    --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \
    --wandb-mode offline \
    >> "$LOG" 2>&1 &

echo "[s12-f] PID=$!" | tee -a "$LOG"
