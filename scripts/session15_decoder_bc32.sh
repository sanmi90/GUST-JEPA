#!/usr/bin/env bash
# Train a LapFiLM SL decoder at base_channels=32 (half of production 64) on
# the production E d=64 encoder. Apples-to-apples lean-decoder ablation:
# does halving the decoder width hurt SSIM / spectral fidelity?
#
# Production decoder: channels=(64, 64, 48, 32, 24), ~700k params.
# This ablation:      channels=(32, 32, 24, 16, 12), ~200k params.
#
# Usage: bash scripts/session15_decoder_bc32.sh <gpu>

set -euo pipefail
GPU=${1:?gpu required (0|1)}
cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

ENC_RUN="outputs/runs/session12/S12_E_d64/encoder"
OUT_DIR="outputs/runs/session15/decoder_bc32"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/launch.log"
echo "[s15-dec32] start at $(date -Iseconds) on gpu=$GPU" | tee "$LOG"

nohup python -u scripts/session9_train_decoder.py \
    --encoder-run "$ENC_RUN" \
    --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \
    --decoder-type lapfilm --decoder-upsample pixelshuffle \
    --decoder-base-ch 32 \
    --decoder-loss region_pyr_specloss \
    --lambda-region 1.0 --lambda-pyramid 0.4 \
    --lambda-gradient 1.0 --lambda-spectral-amp 1.0 \
    --lambda-enstrophy 0.02 --lambda-circulation 0.01 \
    --spectral-window hann --spectral-wake-only \
    --max-iters 12000 \
    --B 16 --T 32 --seed 42 \
    --gpu "$GPU" \
    --output-dir "$OUT_DIR" \
    --eval-every 2000 --checkpoint-every 2000 --log-every 200 \
    > "$OUT_DIR/train.nohup.log" 2>&1 &
PID=$!
echo "[s15-dec32] PID=$PID" | tee -a "$LOG"
