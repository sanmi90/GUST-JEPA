#!/usr/bin/env bash
# Comprehensive B1 v2 driver:
#  1. Encode Fukami d=64 latents (newly-trained checkpoint)
#  2. Encode JEPA d=32 seed latents (3 seeds)
#  3. Symlink latents_jepa_d32 -> session14/latents/S12_E_d32
#  4. Launch 7 predictor trainings in parallel across the 4 GPUs:
#       cuda:0 (L40S #1):       pod_d16  -> pod_d64
#       cuda:1 (L40S #2):       fukami_d3 -> fukami_d32
#       cuda:2 (RTX 6000 #1):   fukami_d64 -> jepa_d32
#       cuda:3 (RTX 6000 #2):   pod_d32
#  predictor_jepa_d64_test1_noBN is already trained (v2) and skipped here.

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
source "$REPO/.venv/bin/activate"
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export VORTEX_JEPA_CACHE="${VORTEX_JEPA_CACHE:-$PREVENT_ROOT/data/processed/vortex-jepa}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
export VORTEX_JEPA_ALLOW_NON_RTX6000=1

LOG="$REPO/outputs/runs/stage8_b1_full_v2.log"
mkdir -p "$(dirname "$LOG")"
echo "[b1-full] start at $(date -Iseconds)" | tee "$LOG"

# ---- 1. Encode all 6 baselines under split_v2 ----
for d in 3 32 64; do
    echo "[b1-full] encoding Fukami d=$d latents (cuda:0)" | tee -a "$LOG"
    python -u scripts/session18/encode_baseline_latents.py \
        --baseline fukami --d "$d" --gpu 0 \
        --checkpoint "outputs/session18/exp_b1/fukami_ae_d${d}/checkpoint_iter006000.pt" \
        >> "$LOG" 2>&1 || echo "[b1-full] FAIL encode fukami d=$d" | tee -a "$LOG"
done
for d in 16 32 64; do
    echo "[b1-full] encoding POD d=$d latents (CPU)" | tee -a "$LOG"
    python -u scripts/session18/encode_baseline_latents.py \
        --baseline pod --d "$d" \
        --basis "outputs/session18/exp_b1/pod_d${d}/pod_basis.npz" \
        >> "$LOG" 2>&1 || echo "[b1-full] FAIL encode pod d=$d" | tee -a "$LOG"
done

# ---- 2. Symlink JEPA d=32 production latents into B1 path ----
if [[ ! -e "$REPO/outputs/session18/exp_b1/latents_jepa_d32" ]]; then
    ln -sfn ../../session14/latents/S12_E_d32 \
        "$REPO/outputs/session18/exp_b1/latents_jepa_d32"
    echo "[b1-full] symlinked latents_jepa_d32 -> session14/latents/S12_E_d32" | tee -a "$LOG"
fi
ls -d "$REPO"/outputs/session18/exp_b1/latents_jepa_d32 "$REPO"/outputs/session18/exp_b1/latents_fukami_d64 2>/dev/null | tee -a "$LOG"

# ---- 3. Launch 4 parallel predictor chains, one per GPU ----
# Each chain takes (gpu_index, "spec1 spec2 ...") where spec is "tag" (uses
# predictor_${tag}_noBN dir) or "tag@dirsuffix" to override the output dir
# suffix (e.g., "jepa_d64@jepa_d64_test1" -> predictor_jepa_d64_test1_noBN).
#
# Trains each baseline to full convergence with the script's default
# --max-iters 20000. Comparison across baselines is between converged
# models; the iter count itself does not need to match across rows.
train_chain () {
    local gpu=$1
    shift
    for spec in "$@"; do
        local tag="${spec%@*}"
        local dirname="${spec#*@}"
        if [[ "$dirname" == "$spec" ]]; then
            dirname="$tag"  # no @suffix; use tag directly
        fi
        local OUT_DIR="$REPO/outputs/session18/exp_b1_test3/predictor_${dirname}_noBN"
        if [[ -f "$OUT_DIR/checkpoint_iter020000.pt" ]]; then
            echo "[b1-full][gpu$gpu] SKIP $tag (checkpoint_iter020000 exists)" >> "$LOG"
            continue
        fi
        echo "[b1-full][gpu$gpu] training $tag -> $OUT_DIR at $(date -Iseconds)" >> "$LOG"
        t0=$(date +%s)
        python -u scripts/session18/train_baseline_predictor.py \
            --latents-dir "$REPO/outputs/session18/exp_b1/latents_${tag}" \
            --tag "${dirname}_noBN" --no-output-bn \
            --gpu "$gpu" \
            --output-dir "$OUT_DIR" \
            >> "$LOG" 2>&1
        rc=$?
        dt=$(($(date +%s) - t0))
        if [[ $rc -eq 0 ]]; then
            echo "[b1-full][gpu$gpu] OK    $tag  ${dt}s" >> "$LOG"
        else
            echo "[b1-full][gpu$gpu] FAIL  $tag  rc=$rc  ${dt}s" >> "$LOG"
        fi
    done
}

# 8 predictors across 4 GPUs, full convergence (~1h each at max-iters=20000):
# cuda:0 (L40S #1):  pod_d16 -> pod_d64       [~2h]
# cuda:1 (L40S #2):  fukami_d3 -> fukami_d32  [~2h]
# cuda:2 (RTX 6000 #1): fukami_d64 -> jepa_d32 [~2h]
# cuda:3 (RTX 6000 #2): pod_d32 -> jepa_d64@jepa_d64_test1 [~2h]
train_chain 0 pod_d16 pod_d64                          &
train_chain 1 fukami_d3 fukami_d32                     &
train_chain 2 fukami_d64 jepa_d32                      &
train_chain 3 pod_d32 jepa_d64@jepa_d64_test1          &
wait

echo "[b1-full] all chains complete at $(date -Iseconds)" | tee -a "$LOG"
