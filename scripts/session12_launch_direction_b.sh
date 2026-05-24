#!/usr/bin/env bash
# Session 12 Direction B launcher: GAN refiner on the frozen E1 decoder.
#
# Usage:
#   bash scripts/session12_launch_direction_b.sh <gpu>
#
# gpu: 0 or 1 (selector into the RTX 6000 subset; D40)
#
# Trains a WakeRefiner residual head and a PatchGAN discriminator on top
# of the Session 11 W0_C_lam100 encoder + decoder_E1_recipe pair (both
# FROZEN). 20k iters at B=16, T=32, seed=42; discriminator warmup of
# 1k iters (refiner trains on L_recon only until step 1000).
# Conservative defaults per SESSION12_CRISP_WAKE.md Direction B.

set -euo pipefail

GPU=${1:?gpu required (0|1)}

ENCODER_RUN="outputs/runs/session11/W0_C_lam100"
DECODER_CKPT="outputs/runs/session11/W0_C_lam100/decoder_E1_recipe/decoder_iter020000.pt"
OUT_DIR="outputs/runs/session12/S12_B_gan_refine"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/launch.log"

cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

echo "[s12-b] gpu=$GPU encoder=$ENCODER_RUN decoder=$DECODER_CKPT out=$OUT_DIR" \
    | tee "$LOG"

nohup python -u scripts/session12_train_refiner.py \
    --encoder-run "$ENCODER_RUN" \
    --decoder-checkpoint "$DECODER_CKPT" \
    --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \
    --lambda-adv 0.05 \
    --lr-refiner 1e-4 \
    --lr-disc 4e-4 \
    --disc-warmup-iters 1000 \
    --refiner-channels 64 \
    --refiner-blocks 6 \
    --lambda-region 1.0 --lambda-pyramid 0.4 --lambda-ffl 0.0 \
    --lambda-enstrophy 0.02 --lambda-circulation 0.01 \
    --max-iters 20000 \
    --B 16 --T 32 --seed 42 \
    --gpu "$GPU" \
    --output-dir "$OUT_DIR" \
    --eval-every 2000 --checkpoint-every 2000 --log-every 200 \
    >> "$LOG" 2>&1 &

echo "[s12-b] PID=$!" | tee -a "$LOG"
