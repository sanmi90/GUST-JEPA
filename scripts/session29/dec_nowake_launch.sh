#!/usr/bin/env bash
# Train the SL visualisation decoder on the FROZEN no-wake-head predictive
# encoder (ctrl_pred_vit_nowake), with the IDENTICAL recipe and budget as the
# production T9 decoders (scripts/session28/t9_runner.sh): LapFiLM + pixelshuffle
# + region_pyr_specloss + spectral-wake-only + enstrophy/circulation terms.
#
# Purpose: answer "JEPA-no-wake-head encoder + decoder-with-wake-loss" --
# does a wake-region-spectral-loss decoder on a frozen latent that does NOT
# carry the wake (repr. wake R^2 = 0.11) recover the wake in the DECODED field,
# and how does its time-averaged SSIM compare to the wake-supervised JEPA
# decoder (test_b SSIM 0.50)? The encoder is frozen, so the latent readability
# cannot change; this isolates what the decoder alone can do.
#
# Shared-workstation hygiene: OMP=8, num-workers 3, taskset confines the caller
# to cores 0-15 so >=64 cores stay free for asolera (SOD2D). RTX 6000 only.
set -uo pipefail
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export OMP_WAIT_POLICY=PASSIVE

SEED="${1:-s0}"                       # ctrl_pred_vit_nowake seed dir suffix
GPU="${2:-0}"                         # RTX 6000 index (0 or 1)
ENC="outputs/runs/session28/ctrl_pred_vit_nowake_${SEED}/encoder"
OUT="outputs/runs/session28/dec_nowake_${SEED}"
SPLIT="configs/splits/split_v2p1.json"
MANIFEST="outputs/data_pipeline/v2p1/manifest.json"

SL_FLAGS=(--omega-pipeline-manifest "$MANIFEST" --split "$SPLIT" --partition v2p1
    --decoder-type lapfilm --decoder-upsample pixelshuffle
    --decoder-loss region_pyr_specloss
    --lambda-region 1.0 --lambda-pyramid 0.4
    --lambda-gradient 1.0 --lambda-spectral-amp 1.0
    --lambda-enstrophy 0.02 --lambda-circulation 0.01
    --spectral-window hann --spectral-wake-only
    --max-iters 30000 --B 16 --T 32 --seed 42 --gpu "$GPU" --num-workers 3
    --eval-every 2000 --checkpoint-every 2000 --log-every 200)

if [[ -f "$OUT/decoder_iter030000.pt" ]]; then
    echo "[dec_nowake] SKIP $OUT (final checkpoint exists)"; exit 0
fi
mkdir -p "$OUT"
echo "[dec_nowake] START $OUT from $ENC at $(date -Iseconds)"
python -u scripts/session9_train_decoder.py \
    --encoder-run "$ENC" "${SL_FLAGS[@]}" \
    --output-dir "$OUT" > "$OUT/train.log" 2>&1
echo "[dec_nowake] DONE $OUT rc=$? at $(date -Iseconds)"
