#!/usr/bin/env bash
# Session 11 Track 2 launcher -- wake_coarse_pool target (Mode D, 288D).
#
# Fires if Track 1 (Modes A/B/C scalar + patch + spectrum heads) does not
# clear the success criteria. Tests an alternative wake observable: predict
# the full wake-ROI vorticity field downsampled to 24x12 (288 scalars).
#
# Usage:
#   bash scripts/session11_launch_track2.sh <config_name> <gpu>
#
# config_name: one of {W2_coarse_lam03, W2_coarse_lam10, W2_coarse_lam30}
# gpu: 0 or 1
#
# Shares all hyperparameters with Track 1 except --wake-observable-type
# (now wake_coarse_pool) and --lambda-wake.

set -euo pipefail

CONFIG=${1:?config_name required (W2_coarse_lam03 | W2_coarse_lam10 | W2_coarse_lam30)}
GPU=${2:?gpu required (0 or 1)}

case "$CONFIG" in
    W2_coarse_lam03) LAM_WAKE=0.03 ;;
    W2_coarse_lam10) LAM_WAKE=0.10 ;;
    W2_coarse_lam30) LAM_WAKE=0.30 ;;
    *) echo "error: unknown config $CONFIG" >&2; exit 2 ;;
esac

OUT_DIR="outputs/runs/session11/${CONFIG}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/launch.log"

cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

echo "[launch-t2] config=$CONFIG gpu=$GPU lam_wake=$LAM_WAKE out=$OUT_DIR" | tee "$LOG"

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
    --tag-suffix "session11_${CONFIG}" \
    --observable-head cl_future --observable-head-weight 0.01 \
    --observable-head-deltas 0 \
    --wake-observable-type wake_coarse_pool --lambda-wake "$LAM_WAKE" \
    --wake-loss smooth_l1 --wake-loss-beta 0.5 --wake-head-hidden 128 \
    --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \
    --wandb-mode offline \
    >> "$LOG" 2>&1 &

echo "[launch-t2] PID=$!" | tee -a "$LOG"
