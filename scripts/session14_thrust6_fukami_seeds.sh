#!/usr/bin/env bash
# Session 14 Thrust 6a launcher: three Fukami AE d=32 seeds (serial on one GPU).
#
# Usage:
#   bash scripts/session14_thrust6_fukami_seeds.sh <gpu>
#
# gpu: 0 or 1 (selector into the RTX 6000 subset; D40)
#
# Three Fukami AE d=32 retrains at seeds {0, 1, 2}, sharing the Session 11 D6
# (W0_C_lam100) Fukami + wake observable head recipe. Each ~2.8 h on an RTX
# 6000 Blackwell; total queue ~9 h. After the three seeds finish, the
# fourth job in the queue trains a Fukami AE at d=12 for the Thrust 4c
# intrinsic-dimensionality agreement test.
#
# Pattern: scripts/session11_launch_track1.sh + scripts/session12_launch_direction_e.sh,
# adapted to scripts/session9_train_fukami.py.
#
# Reference: SESSION14_JFM_NATCOMM_PUSH.md "Thrust 6", "Thrust 4c".

set -euo pipefail

GPU=${1:?gpu required (0|1)}

cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

ROOT="outputs/runs/session14/thrust6"
mkdir -p "$ROOT"
QUEUE_LOG="${ROOT}/fukami_queue_gpu${GPU}.log"
META="${ROOT}/launch_metadata.json"

echo "[s14-t6-fuk] queue start gpu=$GPU at $(date -Iseconds)" | tee -a "$QUEUE_LOG"

run_fukami() {
    local seed=$1
    local d=$2
    local tag=$3
    local out_dir="${ROOT}/${tag}"
    mkdir -p "$out_dir"
    local log="${out_dir}/train.log"

    echo "[s14-t6-fuk] starting ${tag} seed=${seed} d=${d} gpu=${GPU} at $(date -Iseconds)" \
        | tee -a "$QUEUE_LOG"

    python -u scripts/session9_train_fukami.py \
        --partition v1 \
        --all-train \
        --max-iters 20000 \
        --seed "$seed" \
        --d "$d" \
        --B 16 --T 32 \
        --gpu "$GPU" \
        --output-dir "$out_dir" \
        --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \
        --tag-suffix "$tag" \
        --observable-head cl_future --observable-head-weight 1.0 \
        --observable-head-deltas 8 16 24 \
        --wake-observable-type patch_signed_spectrum --lambda-wake 1.00 \
        --wake-loss smooth_l1 --wake-loss-beta 0.5 --wake-head-hidden 128 \
        --log-every 50 --diagnostic-every 500 --checkpoint-every 2000 \
        --num-workers 4 \
        --wandb-mode offline \
        >> "$log" 2>&1

    local rc=$?
    echo "[s14-t6-fuk] finished ${tag} rc=${rc} at $(date -Iseconds)" | tee -a "$QUEUE_LOG"
    return $rc
}

# Three Fukami AE d=32 seeds for the head-to-head with three JEPA d=64 seeds.
for seed in 0 1 2; do
    tag="fukami_d32_seed${seed}"
    run_fukami "$seed" 32 "$tag" || true
done

# Fourth job: Fukami AE d=12 for the Thrust 4c intrinsic-dim agreement test.
# Seed 0 fixed (single matched-dim run; CIs come from the d=32 triple).
run_fukami 0 12 "fukami_d12_seed0" || true

echo "[s14-t6-fuk] queue end gpu=$GPU at $(date -Iseconds)" | tee -a "$QUEUE_LOG"
