#!/usr/bin/env bash
# Session 14 Thrust 6b launcher: three JEPA E d=64 seeds (serial on one GPU).
#
# Usage:
#   bash scripts/session14_thrust6_jepa_seeds.sh <gpu>
#
# gpu: 0 or 1 (selector into the RTX 6000 subset; D40)
#
# Three JEPA E d=64 retrains at seeds {0, 1, 2}, sharing the Session 12
# Direction E (D99 production winner) recipe: d=64, lambda_wake=1.0,
# patch_signed_spectrum wake head, observable cl_future @ delta=0. The
# SL part is a decoder stage trained AFTER each encoder lands; this
# script only trains the encoders. Each ~1.6 h on an RTX 6000 Blackwell;
# total queue ~5 h.
#
# Pattern: scripts/session12_launch_direction_e.sh.
#
# Reference: SESSION14_JFM_NATCOMM_PUSH.md "Thrust 6b".

set -euo pipefail

GPU=${1:?gpu required (0|1)}

cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

ROOT="outputs/runs/session14/thrust6"
mkdir -p "$ROOT"
QUEUE_LOG="${ROOT}/jepa_queue_gpu${GPU}.log"

echo "[s14-t6-jepa] queue start gpu=$GPU at $(date -Iseconds)" | tee -a "$QUEUE_LOG"

run_jepa() {
    local seed=$1
    local tag=$2
    local out_dir="${ROOT}/${tag}/encoder"
    mkdir -p "$out_dir"
    local log="${out_dir}/../train.log"

    echo "[s14-t6-jepa] starting ${tag} seed=${seed} gpu=${GPU} at $(date -Iseconds)" \
        | tee -a "$QUEUE_LOG"

    python -u -m src.training.train_jepa \
        --all-train \
        --max-iters 20000 \
        --seed "$seed" \
        --d 64 \
        --B 16 --T 32 --H-roll 8 \
        --lambda-sigreg 0.01 \
        --lr-encoder 1.5e-4 --lr-predictor 5e-4 \
        --weight-decay 0.05 --warmup-frac 0.05 \
        --num-workers 4 \
        --gpu "$GPU" \
        --output-dir "$out_dir" \
        --log-every 50 --diagnostic-every 500 --checkpoint-every 2000 \
        --projection-norm batchnorm --anticollapse sigreg \
        --tag-suffix "$tag" \
        --observable-head cl_future --observable-head-weight 0.01 \
        --observable-head-deltas 0 \
        --wake-observable-type patch_signed_spectrum --lambda-wake 1.00 \
        --wake-loss smooth_l1 --wake-loss-beta 0.5 --wake-head-hidden 128 \
        --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \
        --wandb-mode offline \
        >> "$log" 2>&1

    local rc=$?
    echo "[s14-t6-jepa] finished ${tag} rc=${rc} at $(date -Iseconds)" | tee -a "$QUEUE_LOG"
    return $rc
}

for seed in 0 1 2; do
    tag="jepa_d64_seed${seed}"
    run_jepa "$seed" "$tag" || true
done

echo "[s14-t6-jepa] queue end gpu=$GPU at $(date -Iseconds)" | tee -a "$QUEUE_LOG"
