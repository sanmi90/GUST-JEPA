#!/usr/bin/env bash
# Run B1 rollouts for all 8 baselines in 4-way parallel across the GPUs.
# Reads converged predictor checkpoint_iter020000.pt for each baseline.
# Writes rollouts to outputs/session18/exp_b1_test3/rollouts_${tag}_noBN/.

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
source "$REPO/.venv/bin/activate"
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export VORTEX_JEPA_CACHE="${VORTEX_JEPA_CACHE:-$PREVENT_ROOT/data/processed/vortex-jepa}"
export VORTEX_JEPA_ALLOW_NON_RTX6000=1

LOG="$REPO/outputs/runs/stage8_b1_rollouts.log"
mkdir -p "$(dirname "$LOG")"
echo "[rollouts] start at $(date -Iseconds)" | tee "$LOG"

rollout_chain () {
    local gpu=$1
    shift
    for spec in "$@"; do
        local tag="${spec%@*}"
        local dirname="${spec#*@}"
        if [[ "$dirname" == "$spec" ]]; then
            dirname="$tag"
        fi
        local PRED="$REPO/outputs/session18/exp_b1_test3/predictor_${dirname}_noBN/checkpoint_iter020000.pt"
        local OUT_DIR="$REPO/outputs/session18/exp_b1_test3/rollouts_${dirname}_noBN"
        if [[ ! -f "$PRED" ]]; then
            echo "[rollouts][gpu$gpu] SKIP $tag (predictor missing: $PRED)" >> "$LOG"
            continue
        fi
        if [[ -f "$OUT_DIR/rollouts_test_b.npz" && -f "$OUT_DIR/rollouts_test_c.npz" ]]; then
            echo "[rollouts][gpu$gpu] SKIP $tag (rollouts exist)" >> "$LOG"
            continue
        fi
        echo "[rollouts][gpu$gpu] rolling $tag at $(date -Iseconds)" >> "$LOG"
        t0=$(date +%s)
        python -u scripts/session18/eval_baseline_rollouts.py \
            --latents-dir "$REPO/outputs/session18/exp_b1/latents_${tag}" \
            --predictor "$PRED" \
            --tag "${dirname}_noBN" \
            --gpu "$gpu" \
            --output-dir "$OUT_DIR" \
            >> "$LOG" 2>&1
        rc=$?
        dt=$(($(date +%s) - t0))
        if [[ $rc -eq 0 ]]; then
            echo "[rollouts][gpu$gpu] OK    $tag  ${dt}s" >> "$LOG"
        else
            echo "[rollouts][gpu$gpu] FAIL  $tag  rc=$rc  ${dt}s" >> "$LOG"
        fi
    done
}

# 8 rollouts across 4 GPUs:
rollout_chain 0 pod_d16 pod_d64                    &
rollout_chain 1 fukami_d3 fukami_d32               &
rollout_chain 2 fukami_d64 jepa_d32                &
rollout_chain 3 pod_d32 jepa_d64@jepa_d64_test1    &
wait

echo "[rollouts] end at $(date -Iseconds)" | tee -a "$LOG"
