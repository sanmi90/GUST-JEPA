#!/usr/bin/env bash
# Low-order matched-head reconstructive AE (fuk_matched recipe: recon MSE + the
# SAME lift+wake heads as the production JEPA, CNN+ViT encoder) at a given low
# latent dim, for the low-order AE-vs-JEPA recovery comparison. The matched-head
# JEPA side (jepa_tf_noc_d{4,8,16}_s42) already exists; only the AE is missing
# below d=64. RTX 6000 only; CPU capped by the caller (taskset -c 0-15).
set -uo pipefail
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE

D="${1:?latent dim}"
GPU="${2:-0}"
SPLIT="configs/splits/split_v2p1.json"
MANIFEST="outputs/data_pipeline/v2p1/manifest.json"
TAG="ctrl_recon_cnnvit_d${D}_s42"
OUT="outputs/runs/session28/${TAG}"

fuk_matched=(--all-train --max-iters 20000 --B 16 --T 32 --gpu "$GPU"
    --partition v2p1 --split "$SPLIT" --omega-pipeline-manifest "$MANIFEST"
    --recon-loss-type mse
    --observable-head cl_future --observable-head-deltas 0 --lambda-lift 0.01
    --wake-observable-type patch_signed_spectrum --lambda-wake 1.0
    --wake-loss smooth_l1 --wake-loss-beta 0.5 --wake-head-hidden 128
    --num-workers 3 --wandb-mode offline
    --log-every 200 --diagnostic-every 2000 --checkpoint-every 10000)

if [[ -f "$OUT/checkpoint_iter020000.pt" ]]; then
    echo "[lowd-ae] SKIP $TAG (ckpt exists)"; exit 0
fi
mkdir -p "$OUT"
echo "[lowd-ae] START $TAG d=$D gpu=$GPU at $(date -Iseconds)"
nice -n 10 python -u scripts/session9_train_fukami.py "${fuk_matched[@]}" \
    --encoder cnn_vit --d "$D" --seed 42 \
    --tag-suffix "s29_${TAG}" --output-dir "$OUT" > "$OUT/train.log" 2>&1
echo "[lowd-ae] DONE $TAG rc=$? at $(date -Iseconds)"
