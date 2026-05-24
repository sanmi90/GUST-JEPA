#!/usr/bin/env bash
# Session 12 Phase 5: extended evaluation (decoder metrics + 2D power spectrum)
# on all finished configurations.
#
# Usage:
#   bash scripts/session12_phase5_eval.sh <gpu>
#
# Loops over each (encoder_run, decoder_run) pair, calls session10_evaluate.py
# (which now also computes the 2D premultiplied wake power spectrum metric
# added in Task 4), and writes extended_metrics.json to the decoder run
# directory. Sequential on one GPU, ~10 min per config x 13 configs ~= 2 h.
#
# Direction B (GAN refiner) is NOT included here because the standard
# session10_evaluate.py does not load the refiner. A separate one-off eval
# is needed for it.

set -euo pipefail

GPU=${1:?gpu required (0|1)}

cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

LOG="outputs/runs/session12/phase5_eval.log"
mkdir -p "$(dirname "$LOG")"
echo "[phase5] starting Session 12 extended eval on gpu=$GPU" | tee "$LOG"

# Format: <tag> <encoder_run_dir> <decoder_run_dir>
CONFIGS=(
    "W0_C_lam100|outputs/runs/session11/W0_C_lam100|outputs/runs/session11/W0_C_lam100/decoder_E1_recipe"
    "S12_A_default|outputs/runs/session11/W0_C_lam100|outputs/runs/session12/S12_A_specloss_default"
    "S12_A_low|outputs/runs/session11/W0_C_lam100|outputs/runs/session12/S12_A_specloss_low"
    "S12_A_high|outputs/runs/session11/W0_C_lam100|outputs/runs/session12/S12_A_specloss_high"
    "S12_C_lam200|outputs/runs/session12/S12_C_lam200/encoder|outputs/runs/session12/S12_C_lam200/encoder/decoder_E1_recipe"
    "S12_C_lam300|outputs/runs/session12/S12_C_lam300/encoder|outputs/runs/session12/S12_C_lam300/encoder/decoder_E1_recipe"
    "S12_C_lam500|outputs/runs/session12/S12_C_lam500/encoder|outputs/runs/session12/S12_C_lam500/encoder/decoder_E1_recipe"
    "S12_D_coarse288|outputs/runs/session12/S12_D_coarse288/encoder|outputs/runs/session12/S12_D_coarse288/encoder/decoder_E1_recipe"
    "S12_D_coarse512|outputs/runs/session12/S12_D_coarse512/encoder|outputs/runs/session12/S12_D_coarse512/encoder/decoder_E1_recipe"
    "S12_E_d64|outputs/runs/session12/S12_E_d64/encoder|outputs/runs/session12/S12_E_d64/encoder/decoder_E1_recipe"
    "S12_F_TC0p01|outputs/runs/session12/S12_F_TC0p01/encoder|outputs/runs/session12/S12_F_TC0p01/encoder/decoder_E1_recipe"
    "S12_F_TC0p03|outputs/runs/session12/S12_F_TC0p03/encoder|outputs/runs/session12/S12_F_TC0p03/encoder/decoder_E1_recipe"
    "S12_F_TC0p10|outputs/runs/session12/S12_F_TC0p10/encoder|outputs/runs/session12/S12_F_TC0p10/encoder/decoder_E1_recipe"
)

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r tag enc dec <<< "$entry"
    echo "[phase5] $(date -Iseconds) eval $tag (enc=$enc, dec=$dec)" | tee -a "$LOG"
    out_dir="$dec/eval"
    mkdir -p "$out_dir"
    python -u scripts/session10_evaluate.py \
        --encoder-run "$enc" \
        --decoder-run "$dec" \
        --gpu "$GPU" \
        --output-json "$out_dir/extended_metrics.json" \
        2>&1 | tee -a "$LOG"
done

echo "[phase5] $(date -Iseconds) done" | tee -a "$LOG"
