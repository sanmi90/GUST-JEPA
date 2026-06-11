#!/usr/bin/env bash
# T9 follow-on (decoders; master plan A1 row T9): waits for the GPU-0 queue to
# complete, then trains the T9 decoder set on that card, 2-packed
# (Fukami-class packing rule). Recipe = session13 SL decoder
# (LapFiLM + pixelshuffle + region_pyr_specloss, D98/D99) on v2p1, identical
# recipe and budget across families (decode-fairness mandate, referee M2/C5):
#   block 1: production tf-no-c + lstm-no-c decoders (via frozen encoders);
#   block 2: matched POST-HOC decoders on frozen Fukami d=64 latents and POD
#            d=64 coefficients (via --latents-npz; fukami latents are encoded
#            here once the fukami_d64_s42 cell lands on the other card).
# Idempotent: skips any decoder whose final checkpoint exists.
set -uo pipefail
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
GPU=0
LOG=outputs/runs/session28/queue_gpu0.log
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

sl_decoder() {  # sl_decoder <tag> <source-flag> <source-path>
    local tag=$1 src_flag=$2 src=$3
    local out="outputs/runs/session28/$tag"
    if [[ -f "$out/decoder_iter030000.pt" ]]; then
        echo "[t9] SKIP $tag"; return 0
    fi
    mkdir -p "$out"
    echo "[t9] START $tag at $(date -Iseconds)"
    python -u scripts/session9_train_decoder.py \
        "$src_flag" "$src" "${SL_FLAGS[@]}" \
        --output-dir "$out" > "$out/train.log" 2>&1
    echo "[t9] DONE $tag rc=$? at $(date -Iseconds)"
}

echo "[t9] waiting for the gpu0 queue to complete ($LOG)"
while ! grep -q "QUEUE COMPLETE" "$LOG" 2>/dev/null; do sleep 300; done
echo "[t9] gpu0 free at $(date -Iseconds); block 1: production SL decoders (2-pack)"

sl_decoder dec_jepa_tf_noc_d64_s42 --encoder-run outputs/runs/session28/jepa_tf_noc_d64_s42/encoder &
P1=$!
sl_decoder dec_jepa_lstm_noc_d64_s42 --encoder-run outputs/runs/session28/jepa_lstm_noc_d64_s42/encoder &
P2=$!
wait $P1; wait $P2

echo "[t9] block 2: matched post-hoc decoders (fukami/pod, --latents-npz)"
FUK_CKPT=outputs/runs/session28/fukami_d64_s42/checkpoint_iter020000.pt
echo "[t9] waiting for $FUK_CKPT (gpu1 queue trains it)"
while [[ ! -f "$FUK_CKPT" ]]; do sleep 300; done

FUK_LAT=outputs/session28/latents/fukami_d64_s42
if [[ ! -f "$FUK_LAT/test_b.npz" ]]; then
    echo "[t9] encoding fukami_d64_s42 latents"
    python scripts/session18/encode_baseline_latents.py \
        --baseline fukami --d 64 --checkpoint "$FUK_CKPT" \
        --partition v2p1 --split "$SPLIT" --pipeline-manifest "$MANIFEST" \
        --splits train test_a test_b test_c \
        --gpu "$GPU" --output-dir "$FUK_LAT" \
        > "$FUK_LAT.encode.log" 2>&1 || { echo "[t9] FATAL: fukami encode failed"; exit 1; }
fi

sl_decoder dec_posthoc_fukami_d64 --latents-npz "$FUK_LAT" &
P3=$!
sl_decoder dec_posthoc_pod_d64 --latents-npz outputs/session28/latents/pod_d64 &
P4=$!
wait $P3; wait $P4
echo "[t9] T9 COMPLETE at $(date -Iseconds)"
