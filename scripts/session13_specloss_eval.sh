#!/usr/bin/env bash
# Session 13 Task 1: extended evaluation of the SL-decoder retrains of
# Session 12 Directions C/D/E/F (using decoder_specloss_recipe trained per
# scripts/session13_queue_specloss_retrains.sh). All SL retrains are capped
# at iter 12000 (peak test_a ratio is at iter 4-8k, iter 12000 is within 2%
# of peak). For the two configs killed mid-30k-run (C_lam200 and D_coarse288)
# the iter 12000 checkpoint is selected explicitly to match.
#
# Output: writes extended_metrics.json next to each decoder checkpoint.
# Eval is sequential on one GPU, ~10 min per config x 9 configs ~= 1.5 h.
#
# Usage:
#   bash scripts/session13_specloss_eval.sh <gpu>

set -euo pipefail

GPU=${1:?gpu required (0|1)}

cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

LOG="outputs/runs/session13/specloss_eval.log"
mkdir -p "$(dirname "$LOG")"
echo "[s13-eval] starting SL-decoder extended eval on gpu=$GPU" | tee "$LOG"

# Format: <tag>|<encoder_run>|<decoder_run>
CONFIGS=(
    "S12_C_lam200_SL|outputs/runs/session12/S12_C_lam200/encoder|outputs/runs/session12/S12_C_lam200/encoder/decoder_specloss_recipe"
    "S12_C_lam300_SL|outputs/runs/session12/S12_C_lam300/encoder|outputs/runs/session12/S12_C_lam300/encoder/decoder_specloss_recipe"
    "S12_C_lam500_SL|outputs/runs/session12/S12_C_lam500/encoder|outputs/runs/session12/S12_C_lam500/encoder/decoder_specloss_recipe"
    "S12_D_coarse288_SL|outputs/runs/session12/S12_D_coarse288/encoder|outputs/runs/session12/S12_D_coarse288/encoder/decoder_specloss_recipe"
    "S12_D_coarse512_SL|outputs/runs/session12/S12_D_coarse512/encoder|outputs/runs/session12/S12_D_coarse512/encoder/decoder_specloss_recipe"
    "S12_E_d64_SL|outputs/runs/session12/S12_E_d64/encoder|outputs/runs/session12/S12_E_d64/encoder/decoder_specloss_recipe"
    "S12_F_TC0p01_SL|outputs/runs/session12/S12_F_TC0p01/encoder|outputs/runs/session12/S12_F_TC0p01/encoder/decoder_specloss_recipe"
    "S12_F_TC0p03_SL|outputs/runs/session12/S12_F_TC0p03/encoder|outputs/runs/session12/S12_F_TC0p03/encoder/decoder_specloss_recipe"
    "S12_F_TC0p10_SL|outputs/runs/session12/S12_F_TC0p10/encoder|outputs/runs/session12/S12_F_TC0p10/encoder/decoder_specloss_recipe"
)

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r tag enc dec <<< "$entry"
    ckpt="$dec/decoder_iter012000.pt"
    echo "[s13-eval] $(date -Iseconds) eval $tag (enc=$enc, ckpt=$ckpt)" | tee -a "$LOG"
    out_dir="$dec/eval"
    mkdir -p "$out_dir"
    python -u scripts/session10_evaluate.py \
        --encoder-run "$enc" \
        --decoder-run "$dec" \
        --decoder-checkpoint "$ckpt" \
        --gpu "$GPU" \
        --output-json "$out_dir/extended_metrics.json" \
        2>&1 | tee -a "$LOG"
done

echo "[s13-eval] $(date -Iseconds) done" | tee -a "$LOG"
