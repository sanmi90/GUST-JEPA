#!/usr/bin/env bash
# Session 12 Direction E launcher: d=64 latent (breaks the Sessions 7-8 d=32 lock).
#
# Usage:
#   bash scripts/session12_launch_direction_e.sh <gpu>
#
# gpu: 0 or 1 (selector into the RTX 6000 subset; D40)
#
# Retrains a JEPA encoder from scratch with the W0_C_lam100 recipe but at d=64
# instead of the LeWM-locked d=32. After encoder training, run
# session11_launch_decoder.sh to retrain the E1 LapFiLM decoder. Note the
# decoder may need a slight adjustment to handle the doubled input dim (the
# LapFiLM init_proj is a Linear(latent_dim, base_ch * base_h * base_w); this
# scales linearly with d).
#
# Reference: SESSION12_CRISP_WAKE.md "Direction E".

set -euo pipefail

GPU=${1:?gpu required (0|1)}

OUT_DIR="outputs/runs/session12/S12_E_d64/encoder"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/../launch.log"

cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

echo "[s12-e] d=64 gpu=$GPU out=$OUT_DIR" | tee "$LOG"

nohup python -u -m src.training.train_jepa \
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
    --output-dir "$OUT_DIR" \
    --log-every 50 --diagnostic-every 500 --checkpoint-every 2000 \
    --projection-norm batchnorm --anticollapse sigreg \
    --tag-suffix "session12_S12_E_d64" \
    --observable-head cl_future --observable-head-weight 0.01 \
    --observable-head-deltas 0 \
    --wake-observable-type patch_signed_spectrum --lambda-wake 1.00 \
    --wake-loss smooth_l1 --wake-loss-beta 0.5 --wake-head-hidden 128 \
    --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \
    --wandb-mode offline \
    >> "$LOG" 2>&1 &

echo "[s12-e] PID=$!" | tee -a "$LOG"
