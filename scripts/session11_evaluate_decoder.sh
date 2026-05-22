#!/usr/bin/env bash
# Run the Session 10 extended evaluation + Figure 3 generation for a
# Session 11 (encoder, decoder) pair.
#
# Usage:
#   bash scripts/session11_evaluate_decoder.sh <encoder_run_dir> <decoder_run_dir> <gpu>
#
# encoder_run_dir: e.g. outputs/runs/session11/W0_C_lam10
# decoder_run_dir: e.g. outputs/runs/session11/W0_C_lam10/decoder_E1_recipe
# gpu: 0 or 1

set -euo pipefail

ENC_RUN=${1:?encoder_run_dir required}
DEC_RUN=${2:?decoder_run_dir required}
GPU=${3:?gpu required (0 or 1)}

cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

OUT_DIR="${DEC_RUN}/eval"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/eval.log"

echo "[eval] encoder=$ENC_RUN decoder=$DEC_RUN gpu=$GPU" | tee "$LOG"

# Session 10 extended metrics (per-encounter wake_field_MSE, wake_enstrophy,
# radial_spectrum_l2, etc.)
python -u scripts/session10_evaluate.py \
    --encoder-run "$ENC_RUN" \
    --decoder-run "$DEC_RUN" \
    --gpu "$GPU" \
    --output-json "${OUT_DIR}/extended_metrics.json" \
    2>&1 | tee -a "$LOG"

# Figure 3 on the canonical Test B encounter (G+1.00_D1.00_Y+0.10 enc 0, idx 0)
python -u scripts/session9_decoder_fig3_pipeline.py \
    --encoder-run "$ENC_RUN" \
    --decoder-checkpoint "${DEC_RUN}/$(ls "$DEC_RUN" | grep -E 'decoder_iter[0-9]+\.pt' | sort | tail -1)" \
    --output-dir "$OUT_DIR" \
    --gpu "$GPU" \
    --fig-test-b-idx 0 \
    --label "$(basename "$ENC_RUN")" \
    2>&1 | tee -a "$LOG"

echo "[eval] done. Outputs in $OUT_DIR" | tee -a "$LOG"
