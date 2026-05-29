#!/usr/bin/env bash
# Stage 8 B1 prep on L40S #2 (cuda:1 in torch view; nvidia-smi index 3).
# Encode 5 ready baselines (Fukami d=3, d=32 + POD d=16, 32, 64), symlink
# JEPA d=64 latents, then train 6 baseline predictors sequentially.
#
# Honors the user's one-off L40S exception via VORTEX_JEPA_ALLOW_NON_RTX6000=1.
# Fukami d=64 and JEPA d=32 predictors are trained later, after their
# encoders/latents land (Fukami d=64 finishing on L40S #1; JEPA d=32 on GPU 0).

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
source "$REPO/.venv/bin/activate"
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export VORTEX_JEPA_CACHE="${VORTEX_JEPA_CACHE:-$PREVENT_ROOT/data/processed/vortex-jepa}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
export VORTEX_JEPA_ALLOW_NON_RTX6000=1

LOG="$REPO/outputs/runs/stage8_b1_prep_l40s.log"
mkdir -p "$(dirname "$LOG")"
echo "[b1-prep] start at $(date -Iseconds) on cuda:1 (L40S #2)" | tee "$LOG"

# ---- 1. Encode baseline latents (Fukami on GPU, POD on CPU) ----
echo | tee -a "$LOG"
echo "[b1-prep] encoding Fukami d=3 latents (cuda:1)" | tee -a "$LOG"
python -u scripts/session18/encode_baseline_latents.py \
    --baseline fukami --d 3 --gpu 1 \
    --checkpoint outputs/session18/exp_b1/fukami_ae_d3/checkpoint_iter006000.pt \
    >> "$LOG" 2>&1 || echo "[b1-prep] FAIL encode fukami d=3" | tee -a "$LOG"

echo | tee -a "$LOG"
echo "[b1-prep] encoding Fukami d=32 latents (cuda:1)" | tee -a "$LOG"
python -u scripts/session18/encode_baseline_latents.py \
    --baseline fukami --d 32 --gpu 1 \
    --checkpoint outputs/session18/exp_b1/fukami_ae_d32/checkpoint_iter006000.pt \
    >> "$LOG" 2>&1 || echo "[b1-prep] FAIL encode fukami d=32" | tee -a "$LOG"

for d in 16 32 64; do
    echo | tee -a "$LOG"
    echo "[b1-prep] encoding POD d=$d latents (CPU)" | tee -a "$LOG"
    python -u scripts/session18/encode_baseline_latents.py \
        --baseline pod --d "$d" \
        --basis "outputs/session18/exp_b1/pod_d${d}/pod_basis.npz" \
        >> "$LOG" 2>&1 || echo "[b1-prep] FAIL encode pod d=$d" | tee -a "$LOG"
done

# Symlink JEPA d=64 latents per RERUN_MANIFEST Stage 8
if [[ ! -e "$REPO/outputs/session18/exp_b1/latents_jepa_d64" ]]; then
    ln -sfn ../../session14/latents/S12_E_d64 \
        "$REPO/outputs/session18/exp_b1/latents_jepa_d64"
    echo "[b1-prep] symlinked outputs/session18/exp_b1/latents_jepa_d64 -> session14/latents/S12_E_d64" | tee -a "$LOG"
fi

ls -d "$REPO"/outputs/session18/exp_b1/latents_* 2>/dev/null | tee -a "$LOG"

# ---- 2. Train predictors sequentially on L40S #2 (cuda:1) ----
mkdir -p "$REPO/outputs/session18/exp_b1_test3"

# Order: JEPA d=64 first (headline), then Fukami small-to-large, then POD small-to-large
TAGS=("jepa_d64" "fukami_d3" "fukami_d32" "pod_d16" "pod_d32" "pod_d64")

for tag in "${TAGS[@]}"; do
    OUT_DIR="$REPO/outputs/session18/exp_b1_test3/predictor_${tag}_noBN"
    if [[ -f "$OUT_DIR/checkpoint_iter006000.pt" ]]; then
        echo "[b1-prep] SKIP predictor $tag (checkpoint exists)" | tee -a "$LOG"
        continue
    fi
    echo | tee -a "$LOG"
    echo "[b1-prep] training predictor $tag on cuda:1 (L40S #2) at $(date -Iseconds)" | tee -a "$LOG"
    t0=$(date +%s)
    python -u scripts/session18/train_baseline_predictor.py \
        --latents-dir "$REPO/outputs/session18/exp_b1/latents_${tag}" \
        --tag "${tag}_noBN" --no-output-bn \
        --gpu 1 \
        --output-dir "$OUT_DIR" \
        >> "$LOG" 2>&1
    rc=$?
    dt=$(($(date +%s) - t0))
    if [[ $rc -eq 0 ]]; then
        echo "[b1-prep] OK    predictor $tag  ${dt}s" | tee -a "$LOG"
    else
        echo "[b1-prep] FAIL  predictor $tag  rc=$rc  ${dt}s" | tee -a "$LOG"
    fi
done

echo | tee -a "$LOG"
echo "[b1-prep] end at $(date -Iseconds)" | tee -a "$LOG"
