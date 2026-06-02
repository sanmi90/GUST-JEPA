#!/usr/bin/env bash
# Session 23 closure chain for the new JEPA encoders (d=8, d=4, no-lift d=64).
# Reproduces the EXACT pipeline behind outputs/session23_closure/closure_r2_dimsweep_d16.csv
# (the source of tab:closure). Per encoder:
#   1. encode  -> outputs/session18/exp_b1/latents_{tag}          (v2 split)
#   2. predictor (train_baseline_predictor --no-output-bn, 20k)  -> exp_b1_test3/predictor_{tag}
#   3. rollouts (eval_baseline_rollouts)                          -> exp_b1_test3/rollouts_{tag}
# Then scripts/session23/closure_r2_dsweep.py reuses scripts/session20/exp_closure_r2.py
# to write outputs/session23_closure/closure_r2_dsweep.csv (r2 + mae + bootstrap CIs).
# tag = jepa_d{D}_noBN  (full noBN tag is used for BOTH latents_{tag} and rollouts_{tag}).
# RTX 6000 only: gpu arg (default 0).
#   bash scripts/session23/closure_dsweep.sh [gpu]
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
source "$REPO/.venv/bin/activate"
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}" WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
GPU="${1:-0}"

LATROOT="outputs/session18/exp_b1"
T3="outputs/session18/exp_b1_test3"

# run_dir  d  tag
runs=(
  "JEPA_d8         8  jepa_d8_noBN"
  "JEPA_d4         4  jepa_d4_noBN"
  "JEPA_d64_nolift 64 jepa_d64_nolift_noBN"
)

for row in "${runs[@]}"; do
  read -r run d tag <<<"$row"
  ENC="outputs/runs/session23/$run/checkpoint_iter020000.pt"
  LAT="$LATROOT/latents_$tag"
  PREDIR="$T3/predictor_$tag"
  PRED="$PREDIR/checkpoint_iter020000.pt"
  ROLL="$T3/rollouts_$tag"
  if [[ ! -f "$ENC" ]]; then echo "[FATAL] encoder missing: $ENC"; exit 2; fi

  echo ">>> [$tag] 1/3 encode latents (d=$d) -> $LAT"
  if [[ ! -f "$LAT/test_b.npz" ]]; then
    python scripts/session18/encode_baseline_latents.py \
      --baseline jepa --d "$d" --checkpoint "$ENC" --output-dir "$LAT" --gpu "$GPU"
  else echo "    skip (latents present)"; fi

  echo ">>> [$tag] 2/3 train noBN predictor -> $PREDIR"
  if [[ ! -f "$PRED" ]]; then
    python scripts/session18/train_baseline_predictor.py \
      --latents-dir "$LAT" --tag "$tag" --output-dir "$PREDIR" \
      --no-output-bn --gpu "$GPU" --seed 0 --num-workers 0
  else echo "    skip (predictor present)"; fi

  echo ">>> [$tag] 3/3 rollouts -> $ROLL"
  if [[ ! -f "$ROLL/test_b.npz" ]]; then
    python scripts/session18/eval_baseline_rollouts.py \
      --latents-dir "$LAT" --predictor "$PRED" --tag "$tag" --output-dir "$ROLL" --gpu "$GPU"
  else echo "    skip (rollouts present)"; fi
done

echo ">>> closure R^2 + MAE (reuses exp_closure_r2) -> outputs/session23_closure/closure_r2_dsweep.csv"
python scripts/session23/closure_r2_dsweep.py

echo "=== CLOSURE DSWEEP COMPLETE -> outputs/session23_closure/closure_r2_dsweep.csv ==="
