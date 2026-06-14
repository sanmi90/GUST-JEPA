#!/usr/bin/env bash
# SESSION29 Track E: the supervised_only control (F4 decisive cell, reconciled to v2.1).
#
# Same encoder + lift/wake heads + SIGReg anti-collapse as the production
# jepa_tf_noc recipe (_run_one.sh jepa_common + wake_on), but with
# --predictive-weight 0.0: NO predictive objective. This isolates whether wake
# supervision ALONE (no prediction, no reconstruction) produces the linearly
# readable wake latent. Anti-collapse (SIGReg) is kept so the encoder does not
# trivially collapse; the only thing removed vs jepa_tf_noc is L_pred + L_roll.
#
# Usage: bash train_supervised_only.sh <d> <seed> <gpu> [max_iters]
#   gpu is the 0-indexed RTX 6000 (require_rtx6000 handles selection).
set -uo pipefail
d=$1; seed=$2; gpu=$3; iters=${4:-20000}
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
REPO=$(pwd); source "$REPO/.venv/bin/activate"
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

SPLIT="configs/splits/split_v2p1.json"
MANIFEST="outputs/data_pipeline/v2p1/manifest.json"
tag="supervised_only_d${d}_s${seed}"
[[ "$iters" -lt 1000 ]] && tag="${tag}_smoke"
out="outputs/runs/session29/${tag}/encoder"
if [[ -f "$out/checkpoint_iter$(printf '%06d' "$iters").pt" ]]; then
    echo "[s29][gpu$gpu] SKIP $tag (ckpt exists)"; exit 0
fi
mkdir -p "$out"

common=(--all-train --max-iters "$iters" --B 16 --T 32 --H-roll 8
    --lambda-sigreg 0.01 --lr-encoder 1.5e-4 --lr-predictor 5e-4 --weight-decay 0.05
    --warmup-frac 0.05 --grad-clip 1.0 --num-workers 3 --gpu "$gpu"
    --projection-norm batchnorm --anticollapse sigreg
    --observable-head cl_future --observable-head-weight 0.01 --observable-head-deltas 0
    --partition v2p1 --split "$SPLIT" --omega-pipeline-manifest "$MANIFEST"
    --wandb-mode offline --log-every 50 --diagnostic-every 200 --checkpoint-every "$iters"
    --wake-observable-type patch_signed_spectrum --lambda-wake 1.00
    --wake-loss smooth_l1 --wake-loss-beta 0.5 --wake-head-hidden 128
    --d "$d" --seed "$seed" --predictor-cond-dim 0
    --predictive-weight 0.0)

echo "[s29][gpu$gpu] START $tag (iters=$iters) at $(date -Iseconds)"
nice -n 10 python -u -m src.training.train_jepa "${common[@]}" \
    --tag-suffix "s29_${tag}" --output-dir "$out" > "$out/train.log" 2>&1
rc=$?
echo "[s29][gpu$gpu] DONE $tag rc=$rc at $(date -Iseconds)"
exit $rc
