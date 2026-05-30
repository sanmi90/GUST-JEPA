#!/usr/bin/env bash
# Session 20 Track A downstream eval for one cell+seed on one card:
# encode per-frame latents -> train unified no-output-BN predictor -> roll out.
# Reuses the B1 machinery (encode_baseline_latents / train_baseline_predictor /
# eval_baseline_rollouts) so all 5 cells are evaluated under the identical
# predictor recipe. Usage: _eval_one.sh <cell> <seed> <gpu>
set -uo pipefail
cell=$1; seed=$2; gpu=$3
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
REPO=$(pwd)
source "$REPO/.venv/bin/activate"
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
export VORTEX_JEPA_ALLOW_NON_RTX6000=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TR="$REPO/outputs/runs/session20/track_a"
TH="$REPO/outputs/runs/session14/thrust6"
EV="$REPO/outputs/session20/track_a"
PRED_ITERS=6000

# Resolve encoder checkpoint dir + baseline family per cell. Use the LATEST
# checkpoint_iter*.pt available (JEPA converges at 20000; the Fukami AE converges
# by ~6000 per D129, so its iter010000 checkpoint is well past convergence and we
# do not wait for 20000).
case "$cell" in
  A1_pred_cnnvit) cdir="$TH/jepa_d64_seed${seed}/encoder"; kind=jepa ;;
  A2_pred_cnn)    cdir="$TR/A2_pred_cnn_seed${seed}/encoder"; kind=jepa ;;
  A5_pred_nowake) cdir="$TR/A5_pred_nowake_seed${seed}/encoder"; kind=jepa ;;
  A3_recon_cnnvit) cdir="$TR/A3_recon_cnnvit_seed${seed}"; kind=fukami ;;
  A4_recon_cnn)    cdir="$TR/A4_recon_cnn_seed${seed}"; kind=fukami ;;
  *) echo "[eval-one] unknown cell $cell"; exit 2 ;;
esac
ck=$(ls -1 "$cdir"/checkpoint_iter*.pt 2>/dev/null | sort -V | tail -1)
tag="${cell}_seed${seed}"
LAT="$EV/latents_${tag}"; PRED="$EV/predictor_${tag}_noBN"; ROLL="$EV/rollouts_${tag}_noBN"
log="$EV/eval_${tag}.log"; mkdir -p "$EV"

if [[ ! -f "$ck" ]]; then echo "[eval-one][gpu$gpu] MISSING encoder ckpt $ck" | tee -a "$log"; exit 3; fi
echo "[eval-one][gpu$gpu] START $tag (kind=$kind) at $(date -Iseconds)" | tee -a "$log"

# 1. encode latents (train/val/test_b/test_c)
if [[ ! -f "$LAT/test_c.npz" ]]; then
  python -u scripts/session18/encode_baseline_latents.py --baseline "$kind" --d 64 \
      --checkpoint "$ck" --output-dir "$LAT" --gpu "$gpu" >> "$log" 2>&1 \
      || { echo "[eval-one][gpu$gpu] FAIL encode $tag" | tee -a "$log"; exit 4; }
fi
# 2. train unified predictor (no output BN)
if [[ ! -f "$PRED/checkpoint_iter$(printf %06d $PRED_ITERS).pt" ]]; then
  python -u scripts/session18/train_baseline_predictor.py --latents-dir "$LAT" \
      --tag "${tag}_noBN" --no-output-bn --gpu "$gpu" --output-dir "$PRED" \
      --max-iters "$PRED_ITERS" >> "$log" 2>&1 \
      || { echo "[eval-one][gpu$gpu] FAIL predictor $tag" | tee -a "$log"; exit 5; }
fi
# 3. roll out
if [[ ! -f "$ROLL/test_b.npz" ]]; then
  python -u scripts/session18/eval_baseline_rollouts.py --latents-dir "$LAT" \
      --predictor "$PRED/checkpoint_iter$(printf %06d $PRED_ITERS).pt" \
      --tag "${tag}_noBN" --gpu "$gpu" --output-dir "$ROLL" >> "$log" 2>&1 \
      || { echo "[eval-one][gpu$gpu] FAIL rollout $tag" | tee -a "$log"; exit 6; }
fi
echo "[eval-one][gpu$gpu] DONE $tag rc=0 at $(date -Iseconds)" | tee -a "$log"
