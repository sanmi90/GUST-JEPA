#!/usr/bin/env bash
# Session 20 Track A downstream eval launcher: 15 chains (5 cells x 3 seeds),
# distributed across all 4 cards (2 RTX cuda 2,3 + 2 L40S cuda 0,1) under the
# session-scoped bypass. Each chain = encode -> predictor -> rollout (_eval_one.sh).
# Run AFTER launch_track_a.sh reports ALL ENCODER TRAINING COMPLETE.
# Usage: bash scripts/session20/eval_track_a.sh
set -uo pipefail
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
REPO=$(pwd)
ONE="$REPO/scripts/session20/_eval_one.sh"; chmod +x "$ONE"
EV="$REPO/outputs/session20/track_a"; mkdir -p "$EV"

CELLS=(A1_pred_cnnvit A2_pred_cnn A3_recon_cnnvit A4_recon_cnn A5_pred_nowake)
# Build 15 "cell seed" jobs; assign gpu round-robin over cuda 2,3 (RTX) and 0,1 (L40S).
jobs_for_gpu() {  # gpu_target  -> emit "cell seed gpu" lines for jobs assigned to it
    local target=$1 idx=0
    for cell in "${CELLS[@]}"; do
        for s in 0 1 2; do
            local gpu=$(( idx % 4 )); idx=$((idx+1))
            # map slot 0,1,2,3 -> cuda 2,3,0,1 (RTX first)
            local cuda=(2 3 0 1); cuda=${cuda[$gpu]}
            [[ "$cuda" == "$target" ]] && echo "$cell $s $cuda"
        done
    done
}
for cuda in 2 3 0 1; do
    jobs_for_gpu "$cuda" | xargs -P 2 -L1 bash "$ONE" > "$EV/eval_cuda${cuda}.log" 2>&1 &
done
wait
echo "[eval-track-a] ALL EVAL CHAINS COMPLETE at $(date -Iseconds)"
