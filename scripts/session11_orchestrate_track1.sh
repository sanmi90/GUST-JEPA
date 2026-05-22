#!/usr/bin/env bash
# Session 11 Track 1 orchestrator: launches the 6 configs in pairs across
# the two RTX 6000s. Waits for each pair to complete before launching the
# next pair. After all 6 done, runs scripts/session11_summarize_track1.py
# with --run-probes to compute the wake-probe summary and apply the gate.
#
# Usage:
#   bash scripts/session11_orchestrate_track1.sh
#
# Picks configs in this order:
#   pair 1: W0_A_lam03 (gpu 0)  + W0_B_lam03 (gpu 1)
#   pair 2: W0_B_lam10 (gpu 0)  + W0_C_lam03 (gpu 1)
#   pair 3: W0_C_lam10 (gpu 0)  + W0_C_lam30 (gpu 1)
#
# If a config's run directory already contains a final checkpoint
# (checkpoint_iter020000.pt), the orchestrator skips it. So you can re-run
# safely after partial completion.

set -euo pipefail

cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"

CONFIGS=("W0_A_lam03" "W0_B_lam03" "W0_B_lam10" "W0_C_lam03" "W0_C_lam10" "W0_C_lam30")

# Pairs of (cfg_gpu0, cfg_gpu1). The script processes them in order.
PAIR1=("W0_A_lam03" "W0_B_lam03")
PAIR2=("W0_B_lam10" "W0_C_lam03")
PAIR3=("W0_C_lam10" "W0_C_lam30")

run_one() {
    local cfg=$1
    local gpu=$2
    local final="outputs/runs/session11/${cfg}/checkpoint_iter020000.pt"
    if [ -f "$final" ]; then
        echo "[orch] $cfg already complete ($final)"
        return 0
    fi
    bash scripts/session11_launch_track1.sh "$cfg" "$gpu"
}

wait_for_finish() {
    local cfg=$1
    local final="outputs/runs/session11/${cfg}/checkpoint_iter020000.pt"
    echo "[orch] waiting for $cfg to write $final ..."
    until [ -f "$final" ]; do
        sleep 60
        if ! pgrep -f "session11_${cfg}" >/dev/null 2>&1 && \
           ! pgrep -f "tag-suffix session11_${cfg}" >/dev/null 2>&1; then
            local iter=$(grep -oE "iter [0-9]+/20000" "outputs/runs/session11/${cfg}/launch.log" 2>/dev/null | tail -1)
            echo "[orch] WARNING: $cfg PID gone but no final ckpt; last log: $iter"
            return 1
        fi
    done
    echo "[orch] $cfg complete."
}

launch_pair() {
    local cfg0=$1
    local cfg1=$2
    echo "[orch] launching pair: $cfg0 (gpu 0) + $cfg1 (gpu 1)"
    run_one "$cfg0" 0
    run_one "$cfg1" 1
    wait_for_finish "$cfg0" || true
    wait_for_finish "$cfg1" || true
}

launch_pair "${PAIR1[@]}"
launch_pair "${PAIR2[@]}"
launch_pair "${PAIR3[@]}"

echo "[orch] all 6 Track 1 configs complete (or skipped). Running summary."
python scripts/session11_summarize_track1.py --run-probes --gpu 0
echo "[orch] done."
