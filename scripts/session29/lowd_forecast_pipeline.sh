#!/usr/bin/env bash
# Low-order matched-head AE-vs-JEPA FORECAST test: for each family latent set,
# train an IDENTICAL matched transformer predictor (cond-dim 0, no output BN, the
# Phase-B recipe) on the frozen latents, then roll it from the pre-impact context.
# This removes the predictor as a confound: the comparison is purely whether the
# JEPA latent SPACE is more forecastable than the matched-head AE latent space.
# The per-horizon probe (h=1..16) is a separate python step.
# RTX 6000 only; caller caps CPU (taskset -c 0-15).
set -uo pipefail
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}" WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE
GPU="${1:-1}"
LAT=outputs/session28/latents
PRED=outputs/session29/lowd_predictors
ROLL=outputs/session29/lowd_rollouts

declare -A LATDIR=(
  [jepa_d4]="$LAT/jepa_tf_noc_d4_s42"   [jepa_d8]="$LAT/jepa_tf_noc_d8_s42"
  [jepa_d16]="$LAT/jepa_tf_noc_d16_s42"
  [ae_d4]="$LAT/ctrl_recon_cnnvit_d4_s42" [ae_d8]="$LAT/ctrl_recon_cnnvit_d8_s42"
  [ae_d16]="$LAT/ctrl_recon_cnnvit_d16_s42"
  [ae_d64]="$LAT/ctrl_recon_cnnvit_s0"   [jepa_d64]="$LAT/jepa_tf_noc_d64_s42"
  [regae_d64]="$LAT/regae/cnn_vit_s0"
)
KEYS=(${2:-ae_d4 ae_d8})

for key in "${KEYS[@]}"; do
  lat="${LATDIR[$key]}"; pout="$PRED/$key"; rout="$ROLL/$key"
  if [[ ! -f "$lat/train.npz" ]]; then echo "[lowd-fc] MISSING latents $key ($lat)"; continue; fi
  if [[ ! -f "$pout/checkpoint_iter020000.pt" ]]; then
    mkdir -p "$pout"; echo "[lowd-fc] TRAIN predictor $key at $(date -Iseconds)"
    nice -n 10 python scripts/session18/train_baseline_predictor.py \
      --latents-dir "$lat" --tag "s29_$key" --gpu "$GPU" --seed 42 \
      --cond-dim 0 --no-output-bn --output-dir "$pout" > "$pout/train.log" 2>&1 \
      || { echo "[lowd-fc] FAILED predictor $key"; continue; }
  fi
  if [[ ! -f "$rout/test_b.npz" ]]; then
    mkdir -p "$rout"; echo "[lowd-fc] ROLL $key at $(date -Iseconds)"
    nice -n 10 python scripts/session18/eval_baseline_rollouts.py \
      --latents-dir "$lat" --predictor "$pout/checkpoint_iter020000.pt" \
      --tag "s29_$key" --gpu "$GPU" --output-dir "$rout" > "$rout/eval.log" 2>&1 \
      || { echo "[lowd-fc] FAILED rollout $key"; continue; }
  fi
  echo "[lowd-fc] DONE $key at $(date -Iseconds)"
done
echo "[lowd-fc] PIPELINE COMPLETE at $(date -Iseconds)"
