#!/usr/bin/env bash
# SESSION29.8 Track A2: regularised reconstructive-AE attribution cells.
# QUEUED experiment: do NOT run until the C-full campaign frees both RTX 6000s.
#
# Question: does the reconstruction objective itself suppress wake readability, or
# did the UNregularised reconstructive latent merely lack anti-collapse geometry?
# This trains the matched reconstructive AE (mse recon + matched lift+wake heads)
# WITH the JEPA SIGReg anti-collapse term (--lambda-sigreg), at CNN and CNN+ViT,
# d=64, 5 seeds. The cnn_vit encoder already carries the batchnorm projection
# boundary SIGReg expects (src/baselines/fukami_ae.py).
#
# The --lambda-sigreg path is wired and CPU-wiring-checked (default 0.0 reproduces
# the unregularised AE byte-for-byte) but has NOT had a GPU smoke yet, because the
# cards were held by C-full at prep time. This script runs a 200-iter GPU smoke per
# architecture FIRST and aborts the full grid if either smoke fails.
#
# Usage (only after C-full is done):
#   bash scripts/session29/a2_regae_launch.sh 0    # gpu index (0 or 1)
set -euo pipefail
cd "$(dirname "$0")/../.."

gpu="${1:-0}"
# Optional architecture filter so the grid can be split across the two RTX 6000s
# (e.g. "bash a2_regae_launch.sh 0 cnn" on card 0 and "... 1 cnn_vit" on card 1).
# Default runs both architectures on the one card.
ARCHS="${2:-cnn cnn_vit}"
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}" WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
SPLIT="configs/splits/split_v2p1.json"
MAN="outputs/data_pipeline/v2p1/manifest.json"
PARTITION="v2p1"
ROOT="outputs/runs/session29_8/regae"
LAM=0.01   # SIGReg weight; matches the JEPA jepa_common recipe (--lambda-sigreg 0.01)

# Guard: refuse to contend with a running training campaign on the cards.
if pgrep -af "train_jepa|cv_full_train" | grep -vq "a2_regae_launch"; then
  if pgrep -af "cv_full_train|src.training.train_jepa" >/dev/null; then
    echo "[a2] REFUSING: a training campaign (C-full / train_jepa) is still running."
    echo "[a2] Wait for it to finish so A2 runs on a single, traceable accelerator."
    exit 3
  fi
fi

train_cell() {  # train_cell <arch> <seed> <max_iters> <tag-extra>
  local arch=$1 seed=$2 iters=$3 tag=$4
  local out="$ROOT/regae_${arch}_d64_s${seed}${tag}"
  [[ -f "$out/checkpoint_iter$(printf '%06d' "$iters").pt" ]] && { echo "[a2] SKIP $out"; return 0; }
  mkdir -p "$out"
  nice -n 10 python -u scripts/session9_train_fukami.py \
    --all-train --max-iters "$iters" --B 16 --T 32 --gpu "$gpu" \
    --partition "$PARTITION" --split "$SPLIT" --omega-pipeline-manifest "$MAN" \
    --recon-loss-type mse \
    --observable-head cl_future --observable-head-deltas 0 --lambda-lift 0.01 \
    --wake-observable-type patch_signed_spectrum --lambda-wake 1.0 \
    --wake-loss smooth_l1 --wake-loss-beta 0.5 --wake-head-hidden 128 \
    --encoder "$arch" --d 64 --lambda-sigreg "$LAM" --seed "$seed" \
    --num-workers 4 --wandb-mode offline \
    --tag-suffix "s298_regae_${arch}_s${seed}${tag}" --output-dir "$out" \
    > "$out/train.log" 2>&1
  echo "[a2] cell $out rc=$?"
}

echo "[a2] GPU smoke (200 iters) per architecture before the full grid..."
for arch in $ARCHS; do
  train_cell "$arch" 0 200 "_smoke"
  if ! ls "$ROOT/regae_${arch}_d64_s0_smoke"/checkpoint_*.pt >/dev/null 2>&1; then
    echo "[a2] SMOKE FAILED for $arch; inspect $ROOT/regae_${arch}_d64_s0_smoke/train.log and fix before the full grid."
    exit 4
  fi
done
echo "[a2] smoke OK. Launching full grid (cnn, cnn_vit) x (s0..s4) at 20000 iters."

for arch in $ARCHS; do
  for s in 0 1 2 3 4; do
    train_cell "$arch" "$s" 20000 ""
  done
done
echo "[a2] ALL A2 cells done. Next: encode latents + scripts/session29 closure/drift eval, then add rows to Table 6/7."
