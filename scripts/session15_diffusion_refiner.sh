#!/usr/bin/env bash
# Session 15 / S15-T5: diffusion refinement on top of the production
# slice-trained E d=64 + SL decoder (D99 winner).
#
# After training, the refiner takes (sl_decoded_omega, z) and produces
# a refined omega. The expectation is that the refiner closes the pixel
# SSIM gap on Test B/C while preserving the spectral fidelity.
#
# Usage:
#   bash scripts/session15_diffusion_refiner.sh <gpu>

set -euo pipefail

GPU=${1:?gpu required (0|1)}

cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

OUT_DIR="outputs/runs/session15/diffusion_refiner"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/launch.log"
echo "[s15-diff] start at $(date -Iseconds) on gpu=$GPU" | tee "$LOG"

# Frozen models from D99 production
ENCODER_CKPT="outputs/runs/session12/S12_E_d64/encoder/checkpoint_iter020000.pt"
DECODER_CKPT="outputs/runs/session12/S12_E_d64/encoder/decoder_specloss_recipe/decoder_iter012000.pt"
PIPE_MANIFEST="outputs/data_pipeline/v1/manifest.json"

nohup python -u -m src.training.train_diffusion_refiner \
    --encoder-checkpoint "$ENCODER_CKPT" \
    --decoder-checkpoint "$DECODER_CKPT" \
    --omega-pipeline-manifest "$PIPE_MANIFEST" \
    --partition v1 \
    --output-dir "$OUT_DIR" \
    --gpu "$GPU" \
    --seed 42 \
    --latent-dim 64 \
    --max-iters 15000 \
    --B 8 --T 32 \
    --num-workers 4 \
    --refiner-base-channels 32 \
    --refiner-ch-mult 1 2 4 \
    --refiner-resblocks 2 \
    --refiner-dropout 0.1 \
    --cond-emb-dim 256 \
    --n-diffusion-steps 1000 \
    --lr 1e-4 \
    --weight-decay 0.0 \
    --warmup-frac 0.05 \
    --grad-clip 1.0 \
    --log-every 50 \
    --checkpoint-every 2500 \
    --sample-every 2500 \
    --sample-n-steps 30 \
    > "$OUT_DIR/train.nohup.log" 2>&1 &

PID=$!
echo "[s15-diff] PID=$PID" | tee -a "$LOG"
