#!/usr/bin/env bash
# Launch a Session 10 E1-recipe LapFiLM decoder on a Session 11 JEPA encoder.
#
# Usage:
#   bash scripts/session11_launch_decoder.sh <encoder_run_dir> <gpu>
#
# encoder_run_dir: a Session 11 run directory containing checkpoint_iter*.pt
#                  (e.g. outputs/runs/session11/W0_C_lam10)
# gpu: 0 or 1 (selector into the RTX 6000 subset)
#
# Decoder output lives at ${encoder_run_dir}/decoder_E1_recipe and uses the
# Session 10 E1 loss recipe (region + Charbonnier pyramid + enstrophy +
# circulation; FFL disabled per the plan -- E1 is the better wake-physics
# recipe per D74). 20k iters at B=16, T=32, seed=42.

set -euo pipefail

ENC_RUN=${1:?encoder_run_dir required}
GPU=${2:?gpu required (0 or 1)}
DEC_DIR="${ENC_RUN}/decoder_E1_recipe"
LOG="${ENC_RUN}/decoder_launch.log"
mkdir -p "$DEC_DIR"

cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

echo "[launch-decoder] encoder=$ENC_RUN gpu=$GPU out=$DEC_DIR" | tee "$LOG"

nohup python -u scripts/session9_train_decoder.py \
    --encoder-run "$ENC_RUN" \
    --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \
    --decoder-type lapfilm \
    --decoder-upsample pixelshuffle \
    --decoder-loss region_pyr_ffl \
    --lambda-region 1.0 --lambda-pyramid 0.4 --lambda-ffl 0.0 \
    --lambda-enstrophy 0.02 --lambda-circulation 0.01 \
    --max-iters 20000 \
    --B 16 --T 32 --seed 42 \
    --gpu "$GPU" \
    --output-dir "$DEC_DIR" \
    --eval-every 2000 --checkpoint-every 2000 --log-every 200 \
    >> "$LOG" 2>&1 &

echo "[launch-decoder] PID=$!" | tee -a "$LOG"
