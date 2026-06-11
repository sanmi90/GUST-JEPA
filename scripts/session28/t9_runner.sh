#!/usr/bin/env bash
# T9 follow-on (decoders; master plan A1 row T9): waits for the GPU-0 queue to
# complete, then trains the two production SL decoders 2-packed on that card
# (Fukami-class packing rule). Recipe = session13 SL decoder
# (LapFiLM + pixelshuffle + region_pyr_specloss, D98/D99) on v2p1.
# The matched POST-HOC decoders on frozen Fukami d=64 latents and POD d=64
# coefficients additionally need the --latents-npz input mode in
# session9_train_decoder.py and the fukami_d64_s42 checkpoint; this runner
# logs NEXT for them if either is missing when the JEPA decoders finish.
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

sl_decoder() {  # sl_decoder <tag> <encoder_run_dir>
    local tag=$1 enc=$2
    local out="outputs/runs/session28/$tag"
    if compgen -G "$out/checkpoint_iter030000.pt" > /dev/null; then
        echo "[t9] SKIP $tag"; return 0
    fi
    mkdir -p "$out"
    echo "[t9] START $tag at $(date -Iseconds)"
    python -u scripts/session9_train_decoder.py \
        --encoder-run "$enc" \
        --omega-pipeline-manifest "$MANIFEST" \
        --split "$SPLIT" --partition v2p1 \
        --decoder-type lapfilm --decoder-upsample pixelshuffle \
        --decoder-loss region_pyr_specloss \
        --lambda-region 1.0 --lambda-pyramid 0.4 \
        --lambda-gradient 1.0 --lambda-spectral-amp 1.0 \
        --lambda-enstrophy 0.02 --lambda-circulation 0.01 \
        --spectral-window hann --spectral-wake-only \
        --max-iters 30000 --B 16 --T 32 --seed 42 --gpu "$GPU" \
        --num-workers 3 \
        --eval-every 2000 --checkpoint-every 2000 --log-every 200 \
        --output-dir "$out" > "$out/train.log" 2>&1
    echo "[t9] DONE $tag rc=$? at $(date -Iseconds)"
}

echo "[t9] waiting for the gpu0 queue to complete ($LOG)"
while ! grep -q "QUEUE COMPLETE" "$LOG" 2>/dev/null; do sleep 300; done
echo "[t9] gpu0 free at $(date -Iseconds); training production SL decoders (2-pack)"

sl_decoder dec_jepa_tf_noc_d64_s42 outputs/runs/session28/jepa_tf_noc_d64_s42/encoder &
P1=$!
sl_decoder dec_jepa_lstm_noc_d64_s42 outputs/runs/session28/jepa_lstm_noc_d64_s42/encoder &
P2=$!
wait $P1; wait $P2

if python - <<'EOF'
import sys
sys.exit(0 if "--latents-npz" in open("scripts/session9_train_decoder.py").read() else 1)
EOF
then
    echo "[t9] NEXT: launch matched post-hoc decoders (fukami/pod) via --latents-npz"
else
    echo "[t9] NEXT: --latents-npz mode not landed yet; fukami/pod post-hoc decoders pending"
fi
echo "[t9] JEPA decoder block complete at $(date -Iseconds)"
