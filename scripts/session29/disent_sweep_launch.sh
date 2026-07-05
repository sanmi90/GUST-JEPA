#!/usr/bin/env bash
# Q1 experiment: does a Discetti-style independence / disentanglement loss make
# the wake linearly readable WITHOUT the wake head?
#
# Trains a PURE predictive JEPA (SIGReg anti-collapse ON, NO lift head, NO wake
# head: --observable-head-weight 0 and --lambda-wake 0) with the off-diagonal
# covariance / decorrelation penalty added
# (--total-correlation-weight; the JEPA-native surrogate of the decorrelation
# regulariser of Wang, Tirelli, Discetti, Ianiro arXiv:2604.18059, 2026).
#
# Usage:  disent_sweep_launch.sh <tc_weight> <gpu> <max_iters>
#   screen both weights cheaply (short max_iters), pick the winner by repr-wake
#   R^2 + PR, then re-run only the winner to convergence (full max_iters).
#
# Recipe = scripts/session28/_run_one.sh jepa_common + the no-wake cell
# (ctrl_pred_vit_nowake), so it is apples-to-apples with the 0.11 baseline.
# RTX 6000 only; CPU-capped by the caller (taskset -c 0-15).
set -uo pipefail
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE

TCW="${1:?tc weight}"
GPU="${2:-0}"
ITERS="${3:-8000}"
SPLIT="configs/splits/split_v2p1.json"
MANIFEST="outputs/data_pipeline/v2p1/manifest.json"
TAG="disent_tc${TCW}_nowake_s0_i${ITERS}"
OUT="outputs/runs/session29/${TAG}/encoder"

jepa_common=(--all-train --B 16 --T 32 --H-roll 8
    --lambda-sigreg 0.01 --lr-encoder 1.5e-4 --lr-predictor 5e-4 --weight-decay 0.05
    --warmup-frac 0.05 --grad-clip 1.0 --num-workers 3 --gpu "$GPU"
    --projection-norm batchnorm --anticollapse sigreg
    --observable-head cl_future --observable-head-weight 0.01 --observable-head-deltas 0
    --partition v2p1 --split "$SPLIT" --omega-pipeline-manifest "$MANIFEST"
    --wandb-mode offline --log-every 200 --diagnostic-every 2000 --checkpoint-every 2000)

if [[ -f "$OUT/checkpoint_iter$(printf '%06d' "$ITERS").pt" ]]; then
    echo "[disent] SKIP $TAG (final ckpt exists)"; exit 0
fi
mkdir -p "$OUT"
echo "[disent] START $TAG tc=$TCW gpu=$GPU iters=$ITERS at $(date -Iseconds)"
python -u -m src.training.train_jepa "${jepa_common[@]}" \
    --predictor-cond-dim 0 --encoder hybrid \
    --lambda-wake 0.0 --observable-head-weight 0.0 \
    --d 64 --seed 0 \
    --total-correlation-weight "$TCW" --max-iters "$ITERS" \
    --tag-suffix "s29_${TAG}" --output-dir "$OUT" > "$OUT/train.log" 2>&1
echo "[disent] DONE $TAG rc=$? at $(date -Iseconds)"
