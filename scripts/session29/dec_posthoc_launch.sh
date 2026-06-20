#!/usr/bin/env bash
# Post-hoc T9 SL decoder on a frozen latent set (--latents-npz mode), IDENTICAL
# recipe/budget to the production decoders (scripts/session28/t9_runner.sh) and to
# dec_posthoc_fukami_d64, so the resulting test_b SSIM is comparable to JEPA's
# 0.504 and Fukami's 0.380. Used here to score field reconstruction of the
# matched-head AE (ctrl_recon) and regAE latents on the same convention.
# RTX 6000 only; caller caps CPU (taskset -c 0-15).
set -uo pipefail
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}" WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE
LATDIR="${1:?latents dir}"; TAG="${2:?tag}"; GPU="${3:-0}"
OUT="outputs/runs/session29/$TAG"
SPLIT="configs/splits/split_v2p1.json"; MANIFEST="outputs/data_pipeline/v2p1/manifest.json"
SL_FLAGS=(--omega-pipeline-manifest "$MANIFEST" --split "$SPLIT" --partition v2p1
    --decoder-type lapfilm --decoder-upsample pixelshuffle
    --decoder-loss region_pyr_specloss
    --lambda-region 1.0 --lambda-pyramid 0.4 --lambda-gradient 1.0 --lambda-spectral-amp 1.0
    --lambda-enstrophy 0.02 --lambda-circulation 0.01
    --spectral-window hann --spectral-wake-only
    --max-iters 30000 --B 16 --T 32 --seed 42 --gpu "$GPU" --num-workers 3
    --eval-every 2000 --checkpoint-every 2000 --log-every 200)
if [[ -f "$OUT/decoder_iter030000.pt" ]]; then echo "[posthoc] SKIP $TAG"; exit 0; fi
mkdir -p "$OUT"
echo "[posthoc] START $TAG from $LATDIR gpu=$GPU at $(date -Iseconds)"
python -u scripts/session9_train_decoder.py --latents-npz "$LATDIR" "${SL_FLAGS[@]}" \
    --output-dir "$OUT" > "$OUT/train.log" 2>&1
echo "[posthoc] DONE $TAG rc=$? at $(date -Iseconds)"
