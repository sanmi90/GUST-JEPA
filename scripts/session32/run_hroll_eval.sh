#!/bin/bash
# SESSION 32 H_roll ablation eval for jepa_nowake_pool_hroll1 (H_roll=1).
# Q1 represent + Q2 rollout (resunet_matched only, --no-native --no-transformer) --
# byte-identical protocol to Track P (run_track_p_eval.sh), reusing the shared
# q1_pool_latents cache (fields_*.npz already present; represent will not overwrite).
# Test C untouched. Offline, RTX 6000, cores 0-15.
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
MODEL=jepa_nowake_pool_hroll1
GPU=${1:-0}

echo "[hroll-eval] Q1 represent @ $(date -Iseconds)"
taskset -c 0-15 python -m src.evaluation.represent --models $MODEL \
  --runs-base "$RUNS" --checkpoint "$CKPT" \
  --partition v2p2 --split "$SPLIT" --pipeline-manifest "$PIPE" \
  --windows "$WIN" --out "$S32/q1_hroll1.json" \
  --cache-dir "$CACHE" --gpu "$GPU" --decoder-steps 6000

echo "[hroll-eval] Q2 rollout (resunet_matched only) @ $(date -Iseconds)"
taskset -c 0-15 python -m src.evaluation.rollout --models $MODEL \
  --runs-base "$RUNS" --checkpoint "$CKPT" --cache-dir "$CACHE" \
  --out "$S32/q2_hroll1.json" --no-native --no-transformer \
  --windows "$WIN" --gpu "$GPU"

echo "[hroll-eval] complete @ $(date -Iseconds)"
