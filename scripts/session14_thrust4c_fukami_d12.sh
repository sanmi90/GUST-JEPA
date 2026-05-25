#!/usr/bin/env bash
# Session 14 Thrust 4c launcher: a single Fukami AE at d=12 for the
# intrinsic-dim agreement test.
#
# Usage:
#   bash scripts/session14_thrust4c_fukami_d12.sh <gpu>
#
# gpu: 0 or 1 (selector into the RTX 6000 subset; D40)
#
# Standalone helper if you want to run the d=12 case outside the Thrust 6
# Fukami queue. Same recipe as Thrust 6a Fukami d=32 except --d 12, seed 0.
# About 2.8 h on an RTX 6000 Blackwell.
#
# Reference: SESSION14_JFM_NATCOMM_PUSH.md "Thrust 4c".

set -euo pipefail

GPU=${1:?gpu required (0|1)}

cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

ROOT="outputs/runs/session14/thrust6"
OUT_DIR="${ROOT}/fukami_d12_seed0"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/train.log"

echo "[s14-t4c] fukami d=12 gpu=$GPU start at $(date -Iseconds)" | tee -a "$LOG"

nohup python -u scripts/session9_train_fukami.py \
    --partition v1 \
    --all-train \
    --max-iters 20000 \
    --seed 0 \
    --d 12 \
    --B 16 --T 32 \
    --gpu "$GPU" \
    --output-dir "$OUT_DIR" \
    --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \
    --tag-suffix "fukami_d12_seed0" \
    --observable-head cl_future --observable-head-weight 1.0 \
    --observable-head-deltas 8 16 24 \
    --wake-observable-type patch_signed_spectrum --lambda-wake 1.00 \
    --wake-loss smooth_l1 --wake-loss-beta 0.5 --wake-head-hidden 128 \
    --log-every 50 --diagnostic-every 500 --checkpoint-every 2000 \
    --num-workers 4 \
    --wandb-mode offline \
    >> "$LOG" 2>&1 &

PID=$!
echo "[s14-t4c] PID=$PID" | tee -a "$LOG"
