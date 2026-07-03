#!/bin/bash
# SESSION 33 D250 vec eval chain -- the full frozen battery on the native
# pooled pipeline retrains (jepa_pool_vec s0/s1/s2 + jepa_nowake_pool_vec).
# Mirrors run_eval_queue.sh / run_track_p_eval.sh settings exactly; the only
# new pieces are the two thin wrappers vec_o1_recovery / vec_envelope.
# Usage: scripts/session33/run_vec_eval.sh [gpu]
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
CACHE=$S33/q1_vec_latents
GPU=${1:-0}

VEC_MODELS="jepa_pool_vec jepa_pool_vec_s1 jepa_pool_vec_s2 jepa_nowake_pool_vec"

mkdir -p "$S33" "$CACHE"

echo "[vec-eval] Q1 represent @ $(date -Iseconds)"
taskset -c 16-23 python -m src.evaluation.represent --models $VEC_MODELS \
  --runs-base "$RUNS" --checkpoint "$CKPT" \
  --partition v2p2 --split "$SPLIT" --pipeline-manifest "$PIPE" \
  --windows "$WIN" --out "$S33/q1_vec.json" \
  --cache-dir "$CACHE" --gpu "$GPU" --decoder-steps 6000

# pressure caches for O1 (frozen session31 artefacts, symlinked not copied)
for s in train test_b; do
  [ -e "$CACHE/pressure_${s}.npz" ] || \
    ln -s "$(pwd)/outputs/session31/q1_latents/pressure_${s}.npz" "$CACHE/pressure_${s}.npz"
done

echo "[vec-eval] Q2 rollout (matched protocol) @ $(date -Iseconds)"
taskset -c 16-23 python -m src.evaluation.rollout --models $VEC_MODELS \
  --runs-base "$RUNS" --checkpoint "$CKPT" --cache-dir "$CACHE" \
  --out "$S33/q2_vec.json" --no-native --no-transformer \
  --windows "$WIN" --gpu "$GPU"

echo "[vec-eval] Q2 rollout (native own-predictor) @ $(date -Iseconds)"
taskset -c 16-23 python -m src.evaluation.rollout --models jepa_pool_vec jepa_nowake_pool_vec \
  --runs-base "$RUNS" --checkpoint "$CKPT" --cache-dir "$CACHE" \
  --out "$S33/q2_vec_native.json" --no-transformer \
  --windows "$WIN" --gpu "$GPU"

echo "[vec-eval] O1 recovery (jepa_pool_vec) @ $(date -Iseconds)"
taskset -c 16-23 python -m scripts.session33.vec_o1_recovery --gpu "$GPU"

echo "[vec-eval] filter envelope (jepa_pool_vec, frozen D220) @ $(date -Iseconds)"
taskset -c 16-23 python -m scripts.session33.vec_envelope --gpu "$GPU"

echo "[vec-eval] complete @ $(date -Iseconds)"
