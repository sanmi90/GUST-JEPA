#!/usr/bin/env bash
# No-conditioning (F-NC) ablation: JEPA d=64 with --predictor-cond-dim 0, so the
# predictor NEVER sees the gust parameters (G,D,Y). Identical to the production d64
# recipe (S12_E_d64: 20k it, seed 42, lift wt 0.01, wake patch_signed_spectrum
# lambda 1.0, SIGReg 0.01, batchnorm) in every other respect, so the only change
# vs the c-conditioned d64 baseline is the conditioning. v2 split. RTX 6000 gpu arg.
#   bash scripts/session23/train_noc.sh [gpu]
set -euo pipefail
cd /home/carlos/GUST-JEPA
source .venv/bin/activate
export PREVENT_ROOT=$HOME/PREVENT WANDB_PROJECT=vortex-jepa
GPU="${1:-0}"
OUT=outputs/runs/session23/JEPA_d64_noc
mkdir -p "$OUT"
echo "[$(date +%H:%M:%S)] START JEPA_d64_noc (d=64 cond-dim=0 gpu=$GPU)"
python -m src.training.train_jepa \
  --all-train --max-iters 20000 --seed 42 --d 64 --B 16 --T 32 --H-roll 8 \
  --lambda-sigreg 0.01 --lr-encoder 1.5e-4 --lr-predictor 5e-4 \
  --weight-decay 0.05 --warmup-frac 0.05 --num-workers 3 \
  --projection-norm batchnorm --anticollapse sigreg --encoder hybrid \
  --observable-head cl_future --observable-head-weight 0.01 --observable-head-deltas 0 \
  --wake-observable-type patch_signed_spectrum --lambda-wake 1.00 \
  --wake-loss smooth_l1 --wake-loss-beta 0.5 --wake-head-hidden 128 \
  --predictor-cond-dim 0 \
  --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json --wandb-mode offline \
  --log-every 200 --diagnostic-every 2000 --checkpoint-every 10000 \
  --gpu "$GPU" --output-dir "$OUT" --tag-suffix JEPA_d64_noc > "$OUT/train.log" 2>&1
echo "[$(date +%H:%M:%S)] DONE JEPA_d64_noc (exit $?)"
echo "JEPA_d64_noc TRAINING COMPLETE"
