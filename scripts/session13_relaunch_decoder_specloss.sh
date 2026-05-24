#!/usr/bin/env bash
# Session 13 first task: re-evaluate Session 12 Directions C/D/E/F with the
# PRF 2026 spectral loss decoder (region_pyr_specloss) instead of the
# E1 recipe (region_pyr_ffl), per D98 finding that SL is necessary to
# preserve 2D spectral fidelity under data evolution.
#
# Usage:
#   bash scripts/session13_relaunch_decoder_specloss.sh <encoder_run_dir> <gpu>
#
# Decoder writes to ${encoder_run_dir}/decoder_specloss_recipe/. Uses Direction
# A default weights (lambda_gradient = lambda_spectral_amp = 1.0). 30k iters
# (matches Direction A so SL has time to converge). All other E1 weights
# preserved from session11_launch_decoder.sh.

set -euo pipefail

ENC_RUN=${1:?encoder_run_dir required}
GPU=${2:?gpu required (0|1)}
DEC_DIR="${ENC_RUN}/decoder_specloss_recipe"
LOG="${ENC_RUN}/../decoder_specloss_launch.log"
mkdir -p "$DEC_DIR"

cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

echo "[s13-specloss] encoder=$ENC_RUN gpu=$GPU out=$DEC_DIR" | tee "$LOG"

nohup python -u scripts/session9_train_decoder.py \
    --encoder-run "$ENC_RUN" \
    --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \
    --decoder-type lapfilm \
    --decoder-upsample pixelshuffle \
    --decoder-loss region_pyr_specloss \
    --lambda-region 1.0 --lambda-pyramid 0.4 \
    --lambda-gradient 1.0 --lambda-spectral-amp 1.0 \
    --lambda-enstrophy 0.02 --lambda-circulation 0.01 \
    --spectral-window hann --spectral-wake-only \
    --max-iters 30000 \
    --B 16 --T 32 --seed 42 \
    --gpu "$GPU" \
    --output-dir "$DEC_DIR" \
    --eval-every 2000 --checkpoint-every 2000 --log-every 200 \
    >> "$LOG" 2>&1 &

echo "[s13-specloss] PID=$!" | tee -a "$LOG"
