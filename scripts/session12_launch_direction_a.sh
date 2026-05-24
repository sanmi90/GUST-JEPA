#!/usr/bin/env bash
# Session 12 Direction A launcher: PRF 2026 SL loss decoder retrain.
#
# Usage:
#   bash scripts/session12_launch_direction_a.sh <variant> <gpu>
#
# variant: one of {default, low, high}
# gpu: 0 or 1 (selector into the RTX 6000 subset; D40)
#
# Reuses the Session 11 W0_C_lam100 frozen encoder and trains a new LapFiLM
# decoder with the Balasubramanian PRF 2026 SL loss
# (region + pyramid + enstrophy + circulation + gradient + spectral amplitude).
# 30k iters (vs the standard 20k) because spectral losses converge slower
# (PRF 2026 trained for 75-250 epochs).
#
# Variants sweep the SL term weights:
#   default:  lambda_gradient=1.0, lambda_spectral_amp=1.0  (PRF Appendix B default)
#   low:      lambda_gradient=0.3, lambda_spectral_amp=0.3
#   high:     lambda_gradient=3.0, lambda_spectral_amp=3.0

set -euo pipefail

VARIANT=${1:?variant required (default|low|high)}
GPU=${2:?gpu required (0|1)}

case "$VARIANT" in
    default) LAM_G=1.0; LAM_S=1.0 ;;
    low)     LAM_G=0.3; LAM_S=0.3 ;;
    high)    LAM_G=3.0; LAM_S=3.0 ;;
    *) echo "error: unknown variant $VARIANT" >&2; exit 2 ;;
esac

ENCODER_RUN="outputs/runs/session11/W0_C_lam100"
OUT_DIR="outputs/runs/session12/S12_A_specloss_${VARIANT}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/launch.log"

cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

echo "[s12-a] variant=$VARIANT gpu=$GPU lambda_grad=$LAM_G lambda_spec=$LAM_S out=$OUT_DIR" | tee "$LOG"

nohup python -u scripts/session9_train_decoder.py \
    --encoder-run "$ENCODER_RUN" \
    --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \
    --decoder-type lapfilm \
    --decoder-upsample pixelshuffle \
    --decoder-loss region_pyr_specloss \
    --lambda-region 1.0 --lambda-pyramid 0.4 \
    --lambda-gradient "$LAM_G" --lambda-spectral-amp "$LAM_S" \
    --lambda-enstrophy 0.02 --lambda-circulation 0.01 \
    --spectral-window hann --spectral-wake-only \
    --max-iters 30000 \
    --B 16 --T 32 --seed 42 \
    --gpu "$GPU" \
    --output-dir "$OUT_DIR" \
    --eval-every 2000 --checkpoint-every 2000 --log-every 200 \
    >> "$LOG" 2>&1 &

echo "[s12-a] PID=$!" | tee -a "$LOG"
