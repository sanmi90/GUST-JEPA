#!/usr/bin/env bash
# Wait for SL decoder seeds 1 and 2 (parallel) to finish, then fire the
# extended-metrics eval + Welch t-test pipeline.

set -euo pipefail
cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"

ROOT="outputs/runs/session14/thrust6"
LOG="${ROOT}/sl_decoders/eval_watchdog.log"
echo "[eval-watchdog] start at $(date -Iseconds)" | tee -a "$LOG"

# Wait until both seed 1 and seed 2 have iter012000.pt checkpoint files.
while true; do
    s1="${ROOT}/jepa_d64_seed1/encoder/decoder_specloss_recipe/decoder_iter012000.pt"
    s2="${ROOT}/jepa_d64_seed2/encoder/decoder_specloss_recipe/decoder_iter012000.pt"
    if [ -f "$s1" ] && [ -f "$s2" ]; then
        echo "[eval-watchdog] both seed 1 and seed 2 finished at $(date -Iseconds)" \
            | tee -a "$LOG"
        break
    fi
    sleep 120
done

sleep 30

# Run eval + Welch on cuda:0 (which will be free since both decoders done)
echo "[eval-watchdog] launching eval + Welch on gpu=0 at $(date -Iseconds)" \
    | tee -a "$LOG"
nohup bash scripts/session14_thrust6_eval_and_welch.sh 0 \
    > "${ROOT}/sl_decoders/eval_welch.nohup.log" 2>&1 &
echo "[eval-watchdog] eval+welch PID=$!" | tee -a "$LOG"
