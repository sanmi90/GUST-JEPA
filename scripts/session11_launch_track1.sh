#!/usr/bin/env bash
# Session 11 Track 1 launcher.
#
# Usage:
#   bash scripts/session11_launch_track1.sh <config_name> <gpu>
#
# config_name: one of {W0_A_lam03, W0_B_lam03, W0_B_lam10, W0_C_lam03, W0_C_lam10, W0_C_lam30}
# gpu: 0 or 1 (selector into the RTX 6000 subset; see src.utils.device.require_rtx6000)
#
# All runs share:
#   d=32, B=16, T=32, H_roll=8, lambda_sigreg=0.01, 20k iters, seed=42
#   omega pipeline v1, observable_head=cl_future @ delta=[0] (cl_present per
#     user feedback that Fukami doesn't understand future-delta CL choice;
#     L_pred already pressures the predictor).
#   wake_loss=smooth_l1, wake_loss_beta=0.5, wake_head_hidden=128
#
# Per-config overrides are wake_observable_type and lambda_wake.
#
# Reference: SESSION11_WAKE_RESULTS_FIRST.md "Track 1: GPT collaborator's wake
# observable head sweep".

set -euo pipefail

CONFIG=${1:?config_name required (W0_A_lam03 | W0_B_lam03 | W0_B_lam10 | W0_C_lam03 | W0_C_lam10 | W0_C_lam30)}
GPU=${2:?gpu required (0 or 1)}

case "$CONFIG" in
    W0_A_lam03)
        WAKE_TYPE=enstrophy_scalar; LAM_WAKE=0.03 ;;
    W0_B_lam03)
        WAKE_TYPE=patch_signed; LAM_WAKE=0.03 ;;
    W0_B_lam10)
        WAKE_TYPE=patch_signed; LAM_WAKE=0.10 ;;
    W0_C_lam03)
        WAKE_TYPE=patch_signed_spectrum; LAM_WAKE=0.03 ;;
    W0_C_lam10)
        WAKE_TYPE=patch_signed_spectrum; LAM_WAKE=0.10 ;;
    W0_C_lam30)
        WAKE_TYPE=patch_signed_spectrum; LAM_WAKE=0.30 ;;
    W0_C_lam50)
        WAKE_TYPE=patch_signed_spectrum; LAM_WAKE=0.50 ;;
    W0_C_lam100)
        WAKE_TYPE=patch_signed_spectrum; LAM_WAKE=1.00 ;;
    W0_C_lam300)
        WAKE_TYPE=patch_signed_spectrum; LAM_WAKE=3.00 ;;
    *)
        echo "error: unknown config $CONFIG" >&2
        exit 2 ;;
esac

OUT_DIR="outputs/runs/session11/${CONFIG}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/launch.log"

cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

echo "[launch-t1] config=$CONFIG gpu=$GPU wake_type=$WAKE_TYPE lam_wake=$LAM_WAKE out=$OUT_DIR" | tee "$LOG"

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
    --wake-observable-type "$WAKE_TYPE" --lambda-wake "$LAM_WAKE" \
    --wake-loss smooth_l1 --wake-loss-beta 0.5 --wake-head-hidden 128 \
    --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \
    --wandb-mode offline \
    >> "$LOG" 2>&1 &

echo "[launch-t1] PID=$!" | tee -a "$LOG"
