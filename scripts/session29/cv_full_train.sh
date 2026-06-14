#!/usr/bin/env bash
# SESSION29 Track C-full: per-fold retrain (jepa + fukami) + encode the held-out
# fold, on a case-disjoint CV fold split. Mirrors the production recipe of
# scripts/session28/_run_one.sh (jepa_common + wake_on; fuk_matched), swapping in
# the fold split. POD per fold is recomputed by group_case_cv_probe.py (cheap, CPU).
#
# Usage: bash cv_full_train.sh <fold> <gpu> [max_iters]
#   gpu = 0-indexed RTX 6000; max_iters default 20000 (production); use small for smoke.
# Caveat (documented): the omega-pipeline train_std is the fixed v2p1 value; a fold
# holds out 16 cases, so the global scale is a ~1% approximation, not refit per fold.
set -uo pipefail
fold=$1; gpu=$2; iters=${3:-20000}
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
REPO=$(pwd); source "$REPO/.venv/bin/activate"
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}" WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8

SPLIT="configs/splits/cv_folds/fold${fold}.json"
MAN="outputs/data_pipeline/v2p1/manifest.json"
ROOT="outputs/runs/session29/cvfull/fold${fold}"
LAT="outputs/session28/latents/cvfull/fold${fold}"
[[ -f "$SPLIT" ]] || { echo "[cvfull] MISSING $SPLIT"; exit 1; }
sm=""; [[ "$iters" -lt 1000 ]] && sm="_smoke"

train_jepa_fold() {
  local out="$ROOT/jepa${sm}/encoder"
  [[ -f "$out/checkpoint_iter$(printf '%06d' "$iters").pt" ]] && { echo "[cvfull][f$fold] SKIP jepa"; return 0; }
  mkdir -p "$out"
  nice -n 10 python -u -m src.training.train_jepa \
    --all-train --max-iters "$iters" --B 16 --T 32 --H-roll 8 \
    --lambda-sigreg 0.01 --lr-encoder 1.5e-4 --lr-predictor 5e-4 --weight-decay 0.05 \
    --warmup-frac 0.05 --grad-clip 1.0 --num-workers 3 --gpu "$gpu" \
    --projection-norm batchnorm --anticollapse sigreg \
    --observable-head cl_future --observable-head-weight 0.01 --observable-head-deltas 0 \
    --partition v2p1 --split "$SPLIT" --omega-pipeline-manifest "$MAN" \
    --wandb-mode offline --log-every 200 --diagnostic-every 2000 --checkpoint-every "$iters" \
    --wake-observable-type patch_signed_spectrum --lambda-wake 1.00 \
    --wake-loss smooth_l1 --wake-loss-beta 0.5 --wake-head-hidden 128 \
    --d 64 --seed 42 --predictor-cond-dim 0 \
    --tag-suffix "s29_cvfull_f${fold}_jepa" --output-dir "$out" > "$out/train.log" 2>&1
  echo "[cvfull][f$fold] jepa rc=$?"
}

train_fukami_fold() {
  local out="$ROOT/fukami${sm}/encoder"
  [[ -f "$out/checkpoint_iter$(printf '%06d' "$iters").pt" ]] && { echo "[cvfull][f$fold] SKIP fukami"; return 0; }
  mkdir -p "$out"
  nice -n 10 python -u -m src.training.session9_train_fukami \
    --all-train --max-iters "$iters" --B 16 --T 32 --gpu "$gpu" \
    --partition v2p1 --split "$SPLIT" --omega-pipeline-manifest "$MAN" \
    --recon-loss-type mse \
    --observable-head cl_future --observable-head-deltas 0 --lambda-lift 0.01 \
    --wake-observable-type patch_signed_spectrum --lambda-wake 1.0 \
    --wake-loss smooth_l1 --wake-loss-beta 0.5 --wake-head-hidden 128 \
    --d 64 --num-workers 3 --wandb-mode offline \
    --log-every 200 --diagnostic-every 2000 --checkpoint-every "$iters" \
    --tag-suffix "s29_cvfull_f${fold}_fukami" --output-dir "$out" > "$out/train.log" 2>&1
  echo "[cvfull][f$fold] fukami rc=$?"
}

encode_fold() {  # encode <model> <ckpt-dir>
  local model=$1 ckptdir=$2 out="$LAT/$1${sm}"
  local ck="$ckptdir/checkpoint_iter$(printf '%06d' "$iters").pt"
  [[ -f "$out/test_b.npz" ]] && { echo "[cvfull][f$fold] SKIP encode $model"; return 0; }
  [[ -f "$ck" ]] || { echo "[cvfull][f$fold] no ckpt for $model"; return 1; }
  mkdir -p "$out"
  nice -n 10 python scripts/session18/encode_baseline_latents.py \
    --baseline "$model" --d 64 --checkpoint "$ck" \
    --partition v2p1 --split "$SPLIT" --pipeline-manifest "$MAN" \
    --splits train test_b --gpu "$gpu" --output-dir "$out" > "$out/encode.log" 2>&1
  echo "[cvfull][f$fold] encode $model rc=$?"
}

echo "[cvfull][f$fold] START iters=$iters gpu=$gpu $(date -Iseconds)"
train_jepa_fold
train_fukami_fold
encode_fold jepa "$ROOT/jepa${sm}/encoder"
encode_fold fukami "$ROOT/fukami${sm}/encoder"
echo "[cvfull][f$fold] DONE $(date -Iseconds)"
