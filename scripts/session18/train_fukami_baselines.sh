#!/usr/bin/env bash
# Session 18 B1 Part (a): train Fukami AE at d = 3, 32, 64 under the locked
# B1 fairness protocol (see SESSION18_B1_PROTOCOL.md).
#
# Usage:
#   bash scripts/session18/train_fukami_baselines.sh [d_list] [gpu]
#     d_list: space-separated list of d values to train (default "3 32 64").
#     gpu: 0 (default; first RTX 6000 Blackwell) or 1 (second).
#
# Examples:
#   bash scripts/session18/train_fukami_baselines.sh                  # all three on GPU 0
#   bash scripts/session18/train_fukami_baselines.sh "3 32 64" 0     # explicit
#   bash scripts/session18/train_fukami_baselines.sh "3" 1           # d=3 only on GPU 1
#   bash scripts/session18/train_fukami_baselines.sh "32 64" 1 &     # parallel-launch on GPU 1
#
# Outputs (per d):
#   outputs/session18/exp_b1/fukami_ae_d${d}/
#     checkpoint_iter020000.pt   final checkpoint
#     final_eval.json            Test A/B/C MSE + SSIM + eps_volume + ratio
#     train.log                  full training log
#     metrics.jsonl              per-iter metrics
#
# Verification gate (must pass before predictor training):
#   final_eval.json Test A SSIM_mean >= 0.60 OR Test A ratio_mean < 2.0.

set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

# Activate the project venv. Fail if not present.
if [[ ! -f "$REPO/.venv/bin/activate" ]]; then
    echo "ERROR: missing $REPO/.venv/bin/activate" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$REPO/.venv/bin/activate"

# Required env vars per CLAUDE.md.
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

# B1 protocol settings (locked across all three d values).
PIPELINE_MANIFEST="$REPO/outputs/data_pipeline/v1/manifest.json"
PARTITION="v1"
MAX_ITERS=20000
SEED=0
LR=1e-3
WEIGHT_DECAY=0.0
WARMUP=0.05
GRAD_CLIP=1.0
B=16
T=32
NUM_WORKERS=4
RECON_LOSS="charbonnier"
CHARB_EPS=0.05
ACTIVATION="relu"
OBS_HEAD="cl_future"
OBS_DELTAS="8 16 24"
OBS_HEAD_WEIGHT=1.0
LAMBDA_RECON=1.0
LAMBDA_LIFT=1.0

D_LIST="${1:-3 32 64}"
GPU="${2:-0}"

if [[ ! -f "$PIPELINE_MANIFEST" ]]; then
    echo "ERROR: pipeline manifest missing: $PIPELINE_MANIFEST" >&2
    exit 1
fi

echo "==============================================================="
echo "Session 18 B1 Part (a): Fukami AE training under fairness protocol"
echo "==============================================================="
echo "  d_list:        $D_LIST"
echo "  gpu:           $GPU"
echo "  partition:     $PARTITION"
echo "  max_iters:     $MAX_ITERS"
echo "  seed:          $SEED"
echo "  recon_loss:    $RECON_LOSS (eps=$CHARB_EPS)"
echo "  activation:    $ACTIVATION (GroupNorm enabled)"
echo "  lr:            $LR  weight_decay: $WEIGHT_DECAY"
echo "  B x T:         $B x $T"
echo "  obs head:      $OBS_HEAD (deltas $OBS_DELTAS)"
echo "  pipeline:      $PIPELINE_MANIFEST"
echo "==============================================================="

for D in $D_LIST; do
    OUT_DIR="$REPO/outputs/session18/exp_b1/fukami_ae_d${D}"
    mkdir -p "$OUT_DIR"

    if [[ -f "$OUT_DIR/checkpoint_iter020000.pt" ]]; then
        echo "[skip d=$D] $OUT_DIR/checkpoint_iter020000.pt already exists"
        continue
    fi

    TAG="session18_b1_fukami_d${D}_seed${SEED}"
    echo
    echo ">>> Training Fukami AE d=$D on GPU $GPU"
    echo ">>> Output dir: $OUT_DIR"

    python "$REPO/scripts/session9_train_fukami.py" \
        --gpu "$GPU" \
        --partition "$PARTITION" \
        --split configs/splits/split_v2.json \
        --all-train \
        --max-iters "$MAX_ITERS" \
        --seed "$SEED" \
        --d "$D" \
        --B "$B" --T "$T" \
        --observable-head "$OBS_HEAD" \
        --observable-head-deltas $OBS_DELTAS \
        --observable-head-weight "$OBS_HEAD_WEIGHT" \
        --lambda-recon "$LAMBDA_RECON" \
        --lambda-lift "$LAMBDA_LIFT" \
        --omega-pipeline-manifest "$PIPELINE_MANIFEST" \
        --recon-loss-type "$RECON_LOSS" \
        --charbonnier-epsilon "$CHARB_EPS" \
        --activation "$ACTIVATION" \
        --lr "$LR" \
        --weight-decay "$WEIGHT_DECAY" \
        --warmup-frac "$WARMUP" \
        --grad-clip "$GRAD_CLIP" \
        --num-workers "$NUM_WORKERS" \
        --tag-suffix "$TAG" \
        --wandb-mode offline \
        --output-dir "$OUT_DIR"

    echo
    echo ">>> d=$D verification gate"
    python "$REPO/scripts/session18/verify_fukami_gate.py" \
        --eval-json "$OUT_DIR/final_eval.json" \
        --d "$D"
    echo
done

echo
echo "==============================================================="
echo "Session 18 B1 Part (a) complete for d_list: $D_LIST"
echo "==============================================================="
