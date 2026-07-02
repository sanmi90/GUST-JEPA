#!/bin/bash
# SESSION 32 Track P eval -- reuse the Session 31 frozen-probe harness on the 5
# pooled kit models (jepa_wake_pool = jepa_pool is reused from S31, not re-run).
# Q1 (represent): decode floor + frozen Ridge/MLP probes.
# Q2 (rollout):   matched ResUNet predictor (resunet_matched) forecast/merit,
#                 --no-native --no-transformer to match the S31 gate-E ablation eval.
# Test C untouched (represent encodes train + test_b only). Offline, RTX 6000, cores 16-23.
set -eu
cd /home/carlos/GUST-JEPA
source .venv/bin/activate
export PREVENT_ROOT=$HOME/PREVENT WANDB_PROJECT=vortex-jepa \
       OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/home/carlos/GUST-JEPA

RUNS=outputs/runs/session32
S32=outputs/session32
PIPE=outputs/data_pipeline/v2p2/manifest.json
SPLIT=configs/splits/split_v2p2.json
WIN=outputs/session31/windows_v2p2.json
CKPT=checkpoint_iter010000.pt
CACHE=$S32/q1_pool_latents
MODELS="jepa_nowake_pool ae_wake_pool ae_nowake_pool supervised_only_pool regAE_pool"
GPU=${1:-0}

mkdir -p "$S32"

echo "[track-p-eval] Q1 represent @ $(date -Iseconds)"
taskset -c 16-23 python -m src.evaluation.represent --models $MODELS \
  --runs-base "$RUNS" --checkpoint "$CKPT" \
  --partition v2p2 --split "$SPLIT" --pipeline-manifest "$PIPE" \
  --windows "$WIN" --out "$S32/q1_pool.json" \
  --cache-dir "$CACHE" --gpu "$GPU" --decoder-steps 6000

echo "[track-p-eval] Q2 rollout (resunet_matched only) @ $(date -Iseconds)"
taskset -c 16-23 python -m src.evaluation.rollout --models $MODELS \
  --runs-base "$RUNS" --checkpoint "$CKPT" --cache-dir "$CACHE" \
  --out "$S32/q2_pool.json" --no-native --no-transformer \
  --windows "$WIN" --gpu "$GPU"

echo "[track-p-eval] complete @ $(date -Iseconds)"
