#!/usr/bin/env bash
# Session 14: parallelize the SL-decoder queue across both RTX 6000 cards.
#
# Strategy (avoids killing seed 0 mid-training):
#   1. Wait for the current seed-0 python (PID set from nvidia-smi) to exit.
#   2. Kill the supervisor (PID 3160203) immediately, so it cannot start
#      seed 1 on GPU 0 redundantly.
#   3. Drain GPU 0 for 60 s.
#   4. Launch seed 1 on GPU 0 and seed 2 on GPU 1 in parallel with nohup.
#   5. Done. Net wall-clock savings ~2 h vs serial 0 -> 1 -> 2.

set -euo pipefail

cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"

ROOT="outputs/runs/session14/thrust6"
SL_ROOT="${ROOT}/sl_decoders"
LAUNCH_LOG="${SL_ROOT}/parallelize.log"
echo "[parallelize] start at $(date -Iseconds)" | tee -a "$LAUNCH_LOG"

SUPERVISOR_PID=3160203
SEED0_PID=3681042

echo "[parallelize] waiting for seed 0 python (PID $SEED0_PID) to exit..." \
    | tee -a "$LAUNCH_LOG"
while [ -d "/proc/${SEED0_PID}" ]; do
    sleep 60
done
echo "[parallelize] seed 0 python exited at $(date -Iseconds)" | tee -a "$LAUNCH_LOG"

# Kill the supervisor so it cannot launch seed 1 on GPU 0 redundantly.
if [ -d "/proc/${SUPERVISOR_PID}" ]; then
    echo "[parallelize] killing supervisor PID $SUPERVISOR_PID at $(date -Iseconds)" \
        | tee -a "$LAUNCH_LOG"
    kill -TERM "$SUPERVISOR_PID" || true
    sleep 5
    kill -KILL "$SUPERVISOR_PID" 2>/dev/null || true
fi

# Drain GPU 0 cuda context.
sleep 60

source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

launch_seed() {
    local seed=$1
    local gpu=$2
    local enc_run="${ROOT}/jepa_d64_seed${seed}/encoder"
    local dec_dir="${enc_run}/decoder_specloss_recipe"
    local log="${enc_run}/../decoder_specloss_parallel_launch.log"
    mkdir -p "$dec_dir"
    echo "[parallelize] launching seed $seed on GPU $gpu at $(date -Iseconds)" \
        | tee -a "$LAUNCH_LOG"
    nohup python -u scripts/session9_train_decoder.py \
        --encoder-run "$enc_run" \
        --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \
        --decoder-type lapfilm \
        --decoder-upsample pixelshuffle \
        --decoder-loss region_pyr_specloss \
        --lambda-region 1.0 --lambda-pyramid 0.4 \
        --lambda-gradient 1.0 --lambda-spectral-amp 1.0 \
        --lambda-enstrophy 0.02 --lambda-circulation 0.01 \
        --spectral-window hann --spectral-wake-only \
        --max-iters 12000 \
        --B 16 --T 32 --seed 42 \
        --gpu "$gpu" \
        --output-dir "$dec_dir" \
        --eval-every 2000 --checkpoint-every 2000 --log-every 200 \
        > "$log" 2>&1 &
    echo "[parallelize] seed $seed PID=$!" | tee -a "$LAUNCH_LOG"
}

launch_seed 1 0
launch_seed 2 1

echo "[parallelize] all queued; supervisor done at $(date -Iseconds)" \
    | tee -a "$LAUNCH_LOG"
