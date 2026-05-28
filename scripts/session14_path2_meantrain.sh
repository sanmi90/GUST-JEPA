#!/usr/bin/env bash
# Session 14 Path 2: train an E d=64 + SL decoder on spanwise-mean omega.
# Compares to the slice-trained production E d=64 + SL.
#
# Stages:
#   1. Build v1_mean cache (~30-60 min, CPU + disk I/O)
#   2. Build v1_mean pipeline manifest (~10 min, CPU)
#   3. Train E d=64 encoder on v1_mean (~5h GPU, on requested --gpu)
#   4. Train SL decoder on the new encoder (~2h GPU)
#   5. Extended-metrics eval (~10 min GPU)
#
# Usage:
#   bash scripts/session14_path2_meantrain.sh <gpu>

set -euo pipefail

GPU=${1:?gpu required (0|1)}

cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

ROOT="outputs/runs/session14/path2_meantrain"
mkdir -p "$ROOT"
LOG="$ROOT/launch.log"
echo "[path2] start at $(date -Iseconds) on gpu=$GPU" | tee "$LOG"

# -----------------------------------------------------------------------------
# Stage 1: build v1_mean cache
# -----------------------------------------------------------------------------
echo "[path2] Stage 1: building v1_mean cache" | tee -a "$LOG"
python scripts/build_omega_mean_cache.py \
    --split configs/splits/split_v2.json \
    2>&1 | tee -a "$LOG"

# -----------------------------------------------------------------------------
# Stage 2: build v1_mean pipeline manifest
# -----------------------------------------------------------------------------
echo "[path2] Stage 2: building v1_mean pipeline manifest" | tee -a "$LOG"
python scripts/build_omega_mean_pipeline.py \
    --split configs/splits/split_v2.json \
    2>&1 | tee -a "$LOG"

# -----------------------------------------------------------------------------
# Stage 3: train E d=64 encoder on v1_mean
# -----------------------------------------------------------------------------
ENC_DIR="$ROOT/encoder"
mkdir -p "$ENC_DIR"
echo "[path2] Stage 3: training E d=64 encoder on v1_mean" | tee -a "$LOG"
VORTEX_JEPA_CACHE="$PREVENT_ROOT/data/processed/vortex-jepa/v1_mean" \
python -u -m src.training.train_jepa \
    --all-train \
    --max-iters 20000 \
    --seed 42 \
    --d 64 \
    --B 16 --T 32 --H-roll 8 \
    --lambda-sigreg 0.01 \
    --lr-encoder 1.5e-4 --lr-predictor 5e-4 \
    --weight-decay 0.05 --warmup-frac 0.05 \
    --num-workers 4 \
    --gpu "$GPU" \
    --output-dir "$ENC_DIR" \
    --log-every 50 --diagnostic-every 500 --checkpoint-every 2000 \
    --projection-norm batchnorm --anticollapse sigreg \
    --tag-suffix path2_meantrain \
    --observable-head cl_future --observable-head-weight 0.01 \
    --observable-head-deltas 0 \
    --wake-observable-type patch_signed_spectrum --lambda-wake 1.00 \
    --wake-loss smooth_l1 --wake-loss-beta 0.5 --wake-head-hidden 128 \
    --omega-pipeline-manifest outputs/data_pipeline/v1_mean/manifest.json \
    --wandb-mode offline \
    2>&1 | tee -a "$LOG"

# -----------------------------------------------------------------------------
# Stage 4: SL decoder on the new encoder
# -----------------------------------------------------------------------------
DEC_DIR="$ENC_DIR/decoder_specloss_recipe"
mkdir -p "$DEC_DIR"
echo "[path2] Stage 4: training SL decoder on v1_mean encoder" | tee -a "$LOG"
VORTEX_JEPA_CACHE="$PREVENT_ROOT/data/processed/vortex-jepa/v1_mean" \
python -u scripts/session9_train_decoder.py \
    --encoder-run "$ENC_DIR" \
    --omega-pipeline-manifest outputs/data_pipeline/v1_mean/manifest.json \
    --decoder-type lapfilm \
    --decoder-upsample pixelshuffle \
    --decoder-loss region_pyr_specloss \
    --lambda-region 1.0 --lambda-pyramid 0.4 \
    --lambda-gradient 1.0 --lambda-spectral-amp 1.0 \
    --lambda-enstrophy 0.02 --lambda-circulation 0.01 \
    --spectral-window hann --spectral-wake-only \
    --max-iters 12000 \
    --B 16 --T 32 --seed 42 \
    --gpu "$GPU" \
    --output-dir "$DEC_DIR" \
    --eval-every 2000 --checkpoint-every 2000 --log-every 200 \
    2>&1 | tee -a "$LOG"

# -----------------------------------------------------------------------------
# Stage 5: extended-metrics eval
# -----------------------------------------------------------------------------
EVAL_DIR="$DEC_DIR/eval"
mkdir -p "$EVAL_DIR"
echo "[path2] Stage 5: extended-metrics eval" | tee -a "$LOG"
VORTEX_JEPA_CACHE="$PREVENT_ROOT/data/processed/vortex-jepa/v1_mean" \
python -u scripts/session10_evaluate.py \
    --encoder-run "$ENC_DIR" \
    --decoder-run "$DEC_DIR" \
    --decoder-checkpoint "$DEC_DIR/decoder_iter012000.pt" \
    --gpu "$GPU" \
    --output-json "$EVAL_DIR/extended_metrics.json" \
    2>&1 | tee -a "$LOG"

echo "[path2] all done at $(date -Iseconds)" | tee -a "$LOG"
