#!/usr/bin/env bash
# Train one Track A cell+seed on one RTX card. Dispatched by launch_track_a.sh
# via xargs for per-card concurrency. Usage: _train_one.sh <cell> <seed> <gpu>
set -uo pipefail
cell=$1; seed=$2; gpu=$3
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
REPO=$(pwd)
source "$REPO/.venv/bin/activate"
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
# Session-scoped: user authorised all 4 cards (2 RTX 6000 + 2 L40S) this session.
# With the bypass, --gpu indexes the full CUDA enumeration: cuda 0,1 = L40S,
# cuda 2,3 = RTX 6000. Recorded gpu_name per run for honest provenance.
export VORTEX_JEPA_ALLOW_NON_RTX6000=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT="outputs/runs/session20/track_a"
MANIFEST="outputs/data_pipeline/v1/manifest.json"
NW=3  # data-loader workers per run (3 concurrent runs/card -> keep CPU sane)

jepa_out="$ROOT/${cell}_seed${seed}/encoder"
fk_out="$ROOT/${cell}_seed${seed}"
# idempotent skip
if [[ -f "$jepa_out/checkpoint_iter020000.pt" || -f "$fk_out/checkpoint_iter020000.pt" ]]; then
    echo "[one][gpu$gpu] SKIP ${cell}_seed${seed} (ckpt exists)"; exit 0
fi
echo "[one][gpu$gpu] START ${cell}_seed${seed} at $(date -Iseconds)"

common_jepa=(--all-train --max-iters 20000 --seed "$seed" --d 64 --B 16 --T 32 --H-roll 8
    --lambda-sigreg 0.01 --lr-encoder 1.5e-4 --lr-predictor 5e-4 --weight-decay 0.05
    --warmup-frac 0.05 --num-workers "$NW" --gpu "$gpu" --projection-norm batchnorm
    --anticollapse sigreg --observable-head cl_future --observable-head-weight 0.01
    --observable-head-deltas 0 --omega-pipeline-manifest "$MANIFEST" --wandb-mode offline
    --log-every 200 --diagnostic-every 2000 --checkpoint-every 10000)

case "$cell" in
  A2_pred_cnn)
    mkdir -p "$jepa_out"
    python -u -m src.training.train_jepa "${common_jepa[@]}" \
        --output-dir "$jepa_out" --tag-suffix "${cell}_seed${seed}" \
        --encoder cnn_only \
        --wake-observable-type patch_signed_spectrum --lambda-wake 1.00 \
        --wake-loss smooth_l1 --wake-loss-beta 0.5 --wake-head-hidden 128 \
        > "$ROOT/${cell}_seed${seed}/train.log" 2>&1 ;;
  A5_pred_nowake)
    mkdir -p "$jepa_out"
    python -u -m src.training.train_jepa "${common_jepa[@]}" \
        --output-dir "$jepa_out" --tag-suffix "${cell}_seed${seed}" \
        --encoder hybrid --lambda-wake 0.0 \
        > "$ROOT/${cell}_seed${seed}/train.log" 2>&1 ;;
  A3_recon_cnnvit|A4_recon_cnn)
    enc=cnn_vit; [[ "$cell" == "A4_recon_cnn" ]] && enc=cnn
    mkdir -p "$fk_out"
    python -u scripts/session9_train_fukami.py --all-train --max-iters 20000 --seed "$seed" \
        --d 64 --B 16 --T 32 --gpu "$gpu" --output-dir "$fk_out" \
        --encoder "$enc" --recon-loss-type mse \
        --observable-head cl_future --observable-head-deltas 0 --lambda-lift 0.01 \
        --wake-observable-type patch_signed_spectrum --lambda-wake 1.0 \
        --wake-loss smooth_l1 --wake-loss-beta 0.5 --wake-head-hidden 128 \
        --omega-pipeline-manifest "$MANIFEST" --wandb-mode offline \
        --tag-suffix "${cell}_seed${seed}" --num-workers "$NW" \
        --log-every 200 --diagnostic-every 2000 --checkpoint-every 10000 \
        > "$fk_out/train.log" 2>&1 ;;
  *) echo "[one] unknown cell $cell"; exit 2 ;;
esac
rc=$?
echo "[one][gpu$gpu] DONE ${cell}_seed${seed} rc=$rc at $(date -Iseconds)"
exit $rc
