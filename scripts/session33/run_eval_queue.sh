#!/bin/bash
# SESSION 33 eval queue -- Q1/Q2 for the dimension plateau, min-d panel and
# 3-seed spine pass (SESSION_33_MANUSCRIPT_V3.md Section 11 items 2 and 8).
# Mirrors scripts/session32/run_track_p_eval.sh (same frozen-probe harness,
# same Q2 --no-native --no-transformer matched-predictor setting).
# Usage: scripts/session33/run_eval_queue.sh [gpu]
set -eu
cd /home/carlos/GUST-JEPA
source .venv/bin/activate
export PREVENT_ROOT=$HOME/PREVENT WANDB_PROJECT=vortex-jepa \
       OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/home/carlos/GUST-JEPA

RUNS=outputs/runs/session33
S33=outputs/session33
PIPE=outputs/data_pipeline/v2p2/manifest.json
SPLIT=configs/splits/split_v2p2.json
WIN=outputs/session31/windows_v2p2.json
CKPT=checkpoint_iter010000.pt
CACHE=$S33/q1_d_latents
GPU=${1:-0}

D_MODELS="jepa_pool_d16 jepa_pool_d64 jepa_pool_d8 jepa_pool_d4 fukami_wake_d16 fukami_wake_d8 fukami_wake_d4"
SEED_MODELS="jepa_pool_s1 jepa_pool_s2 supervised_only_pool_s1 supervised_only_pool_s2"

mkdir -p "$S33" "$CACHE"

echo "[s33-eval] Q1 represent (dimension set) @ $(date -Iseconds)"
taskset -c 16-23 python -m src.evaluation.represent --models $D_MODELS \
  --runs-base "$RUNS" --checkpoint "$CKPT" \
  --partition v2p2 --split "$SPLIT" --pipeline-manifest "$PIPE" \
  --windows "$WIN" --out "$S33/q1_d.json" \
  --cache-dir "$CACHE" --gpu "$GPU" --decoder-steps 6000

echo "[s33-eval] Q1 represent (seed set) @ $(date -Iseconds)"
taskset -c 16-23 python -m src.evaluation.represent --models $SEED_MODELS \
  --runs-base "$RUNS" --checkpoint "$CKPT" \
  --partition v2p2 --split "$SPLIT" --pipeline-manifest "$PIPE" \
  --windows "$WIN" --out "$S33/q1_seeds.json" \
  --cache-dir "$CACHE" --gpu "$GPU" --decoder-steps 6000

echo "[s33-eval] Q2 rollout (plateau pair) @ $(date -Iseconds)"
taskset -c 16-23 python -m src.evaluation.rollout --models jepa_pool_d16 jepa_pool_d64 \
  --runs-base "$RUNS" --checkpoint "$CKPT" --cache-dir "$CACHE" \
  --out "$S33/q2_d.json" --no-native --no-transformer \
  --windows "$WIN" --gpu "$GPU"

echo "[s33-eval] Q2 rollout (seed set) @ $(date -Iseconds)"
taskset -c 16-23 python -m src.evaluation.rollout --models $SEED_MODELS \
  --runs-base "$RUNS" --checkpoint "$CKPT" --cache-dir "$CACHE" \
  --out "$S33/q2_seeds.json" --no-native --no-transformer \
  --windows "$WIN" --gpu "$GPU"

echo "[s33-eval] complete @ $(date -Iseconds)"
