#!/usr/bin/env bash
# Session 23 d-sweep + no-lift reference. Replicates the EXACT d=16 recipe
# (outputs/runs/session23/JEPA_d16 metrics.jsonl config), changing only --d and,
# for the reference, --observable-head-weight 0.0 (drop the lift head, keep wake).
# Split defaults to split_v2.json (226/42/24, v2). RTX 6000 only: --gpu 0 and 1.
#   d=8, d=4  -> card 0 (sequential; both tiny)
#   d=64 no-lift -> card 1 (parallel)
set -euo pipefail
cd /home/carlos/GUST-JEPA
source .venv/bin/activate
export PREVENT_ROOT=$HOME/PREVENT WANDB_PROJECT=vortex-jepa

MANIFEST=outputs/data_pipeline/v1/manifest.json
common=(--all-train --max-iters 20000 --seed 42 --B 16 --T 32 --H-roll 8
  --lambda-sigreg 0.01 --lr-encoder 1.5e-4 --lr-predictor 5e-4
  --weight-decay 0.05 --warmup-frac 0.05 --num-workers 3
  --projection-norm batchnorm --anticollapse sigreg --encoder hybrid
  --observable-head cl_future --observable-head-deltas 0
  --wake-observable-type patch_signed_spectrum --lambda-wake 1.00
  --wake-loss smooth_l1 --wake-loss-beta 0.5 --wake-head-hidden 128
  --omega-pipeline-manifest "$MANIFEST" --wandb-mode offline
  --log-every 200 --diagnostic-every 2000 --checkpoint-every 10000)

train () {  # $1=d  $2=gpu  $3=outdir  $4=obs_weight  $5=tag
  local out="outputs/runs/session23/$3"
  mkdir -p "$out"
  echo "[$(date +%H:%M:%S)] START $5 (d=$1 gpu=$2 obs_w=$4) -> $out"
  python -m src.training.train_jepa "${common[@]}" \
    --d "$1" --gpu "$2" --output-dir "$out" --tag-suffix "$5" \
    --observable-head-weight "$4" > "$out/train.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $5 (exit $?)"
}

# card 1: the big no-lift d=64 reference, in the background
train 64 1 JEPA_d64_nolift 0.0 JEPA_d64_nolift &
BG=$!

# card 0: d=8 then d=4, sequential
train 8 0 JEPA_d8 0.01 JEPA_d8
train 4 0 JEPA_d4 0.01 JEPA_d4

wait "$BG"
echo "[$(date +%H:%M:%S)] ALL TRAININGS COMPLETE"
