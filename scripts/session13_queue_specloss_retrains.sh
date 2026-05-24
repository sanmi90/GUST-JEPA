#!/usr/bin/env bash
# Session 13 queue: launch SL-decoder retrains sequentially on a single GPU
# for the configs assigned to that GPU. Each retrain runs in the foreground
# so the queue blocks naturally between configs. Run on each GPU in parallel
# via two `bash ... &` invocations.
#
# Usage:
#   bash scripts/session13_queue_specloss_retrains.sh <gpu>
#
# GPU 0 queue (5 configs ~5h):
#   S12_C_lam200, S12_C_lam300, S12_C_lam500, S12_E_d64, S12_F_TC0p10
# GPU 1 queue (4 configs ~4h):
#   S12_D_coarse288, S12_D_coarse512, S12_F_TC0p01, S12_F_TC0p03

set -euo pipefail

GPU=${1:?gpu required (0|1)}

cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

if [[ "$GPU" == "0" ]]; then
    # S12_C_lam200 already trained to iter 26000; eval from iter 12000 ckpt.
    QUEUE=(
        "outputs/runs/session12/S12_C_lam300/encoder"
        "outputs/runs/session12/S12_C_lam500/encoder"
        "outputs/runs/session12/S12_E_d64/encoder"
        "outputs/runs/session12/S12_F_TC0p10/encoder"
    )
elif [[ "$GPU" == "1" ]]; then
    # S12_D_coarse288 already trained to iter 26000; eval from iter 12000 ckpt.
    QUEUE=(
        "outputs/runs/session12/S12_D_coarse512/encoder"
        "outputs/runs/session12/S12_F_TC0p01/encoder"
        "outputs/runs/session12/S12_F_TC0p03/encoder"
    )
else
    echo "error: gpu must be 0 or 1" >&2
    exit 2
fi

QLOG="outputs/runs/session13/queue_gpu${GPU}.log"
mkdir -p "$(dirname "$QLOG")"
echo "[s13-queue gpu=$GPU] queue size = ${#QUEUE[@]}" | tee "$QLOG"

for ENC_RUN in "${QUEUE[@]}"; do
    TAG=$(basename "$(dirname "$ENC_RUN")")
    DEC_DIR="${ENC_RUN}/decoder_specloss_recipe"
    LOG="${ENC_RUN}/../decoder_specloss_launch.log"
    mkdir -p "$DEC_DIR"
    echo "[s13-queue gpu=$GPU] $(date -Iseconds) starting $TAG -> $DEC_DIR" | tee -a "$QLOG"

    # Foreground call so the queue blocks until this retrain finishes.
    python -u scripts/session9_train_decoder.py \
        --encoder-run "$ENC_RUN" \
        --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \
        --decoder-type lapfilm \
        --decoder-upsample pixelshuffle \
        --decoder-loss region_pyr_specloss \
        --lambda-region 1.0 --lambda-pyramid 0.4 \
        --lambda-gradient 1.0 --lambda-spectral-amp 1.0 \
        --lambda-enstrophy 0.02 --lambda-circulation 0.01 \
        --spectral-window hann --spectral-wake-only \
        --max-iters 12000 \
        --B 16 --T 32 --seed 42 \
        --gpu "$GPU" \
        --output-dir "$DEC_DIR" \
        --eval-every 2000 --checkpoint-every 2000 --log-every 200 \
        2>&1 | tee "$LOG"

    echo "[s13-queue gpu=$GPU] $(date -Iseconds) finished $TAG" | tee -a "$QLOG"
done
echo "[s13-queue gpu=$GPU] $(date -Iseconds) ALL DONE" | tee -a "$QLOG"
