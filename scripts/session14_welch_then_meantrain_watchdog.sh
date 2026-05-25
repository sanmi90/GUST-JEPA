#!/usr/bin/env bash
# Wait for the Welch t-test summary to land, then fire Path 2 (full retrain on
# spanwise-mean omega) on whichever GPU is free.

set -euo pipefail
cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"

LOG="outputs/runs/session14/path2_meantrain/watchdog.log"
mkdir -p "$(dirname "$LOG")"
echo "[path2-watchdog] start at $(date -Iseconds)" | tee -a "$LOG"

WELCH_JSON="outputs/session14/thrust6_welch_summary.json"

# Poll for the Welch summary file
while [ ! -f "$WELCH_JSON" ]; do
    sleep 120
done
echo "[path2-watchdog] welch summary observed at $(date -Iseconds)" | tee -a "$LOG"

# Wait an extra 60 s for any GPU release
sleep 60

# Pick GPU 0 (the long-running training preference). The watchdog has waited
# for the welch eval to finish so cuda:0 should be free.
GPU=0
echo "[path2-watchdog] launching Path 2 on gpu=$GPU at $(date -Iseconds)" \
    | tee -a "$LOG"

nohup bash scripts/session14_path2_meantrain.sh "$GPU" \
    > outputs/runs/session14/path2_meantrain/launch.nohup.log 2>&1 &
echo "[path2-watchdog] Path 2 PID=$!" | tee -a "$LOG"
