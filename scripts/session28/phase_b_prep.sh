#!/usr/bin/env bash
# Session 28 -> 29 pipelining (Phase B0 prep, user-approved 2026-06-11):
# everything here is unblocked by wave-1 convergence and runs WHILE the
# training queue drains the remaining waves.
#   1. POD bases d in {16, 32, 64} on v2p1 (CPU only; ~1 h total).
#   2. Latents for the six frozen T1/T2 encoders (light RTX 6000 inference,
#      --gpu 1; minutes per encoder).
#   3. DNS physical metrics on v2p1 (CPU).
#   4. Per-frame probe targets on v2p1, z_full mirrored from the seed-42
#      production latents (CPU; needs step 2).
# Idempotent: each step skips when its output exists. Hardware rule: GPU work
# stays on the RTX 6000s (CLAUDE.md); POD/exp2 are CPU and niced so the
# training dataloaders keep their cores.
set -uo pipefail
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

SPLIT="configs/splits/split_v2p1.json"
MANIFEST="outputs/data_pipeline/v2p1/manifest.json"
PARTITION="v2p1"
GPU=1
TAGS=(jepa_tf_noc_d64_s42 jepa_tf_noc_d64_s0 jepa_tf_noc_d64_s1 jepa_tf_noc_d64_s2
      jepa_lstm_noc_d64_s42 jepa_lstm_noc_d64_s0)

pod() {
    for D in 16 32 64; do
        local out="outputs/session28/pod/pod_d${D}"
        if [[ -f "$out/pod_basis.npz" ]]; then
            echo "[prep][pod] SKIP d=$D"; continue
        fi
        mkdir -p "$out"
        echo "[prep][pod] d=$D START $(date -Iseconds)"
        nice -n 10 python scripts/session11_pod_baseline.py \
            --d "$D" --partition "$PARTITION" --split "$SPLIT" \
            --omega-pipeline-manifest "$MANIFEST" --output-dir "$out" \
            > "$out/pod.log" 2>&1 || { echo "[prep][pod] d=$D FAILED"; return 1; }
        echo "[prep][pod] d=$D DONE $(date -Iseconds)"
    done
}

latents() {
    for tag in "${TAGS[@]}"; do
        local ckpt="outputs/runs/session28/$tag/encoder/checkpoint_iter020000.pt"
        local out="outputs/session28/latents/$tag"
        if [[ -f "$out/test_b.npz" ]]; then
            echo "[prep][latents] SKIP $tag"; continue
        fi
        if [[ ! -f "$ckpt" ]]; then
            echo "[prep][latents] WAITING-SKIP $tag (no final ckpt yet)"; continue
        fi
        mkdir -p "$out"
        echo "[prep][latents] $tag START $(date -Iseconds)"
        python scripts/session18/encode_baseline_latents.py \
            --baseline jepa --d 64 --checkpoint "$ckpt" \
            --partition "$PARTITION" --split "$SPLIT" \
            --pipeline-manifest "$MANIFEST" \
            --splits train test_a test_b test_c \
            --gpu "$GPU" --output-dir "$out" \
            > "$out/encode.log" 2>&1 || { echo "[prep][latents] $tag FAILED"; return 1; }
        echo "[prep][latents] $tag DONE $(date -Iseconds)"
    done
}

dns_metrics() {
    local out="outputs/session28/exp2"
    if [[ -f "$out/dns_physical_metrics.npz" ]]; then
        echo "[prep][dns] SKIP"; return 0
    fi
    mkdir -p "$out"
    echo "[prep][dns] START $(date -Iseconds)"
    nice -n 10 python scripts/session17/exp2_dns_physical_metrics.py \
        --split-manifest "$SPLIT" --partition "$PARTITION" \
        --omega-manifest "$MANIFEST" --out "$out" \
        > "$out/dns_metrics.log" 2>&1 || { echo "[prep][dns] FAILED"; return 1; }
    echo "[prep][dns] DONE $(date -Iseconds)"
}

targets() {
    local out="outputs/session28/exp2/per_frame_targets"
    if [[ -f "$out/test_b.npz" ]]; then
        echo "[prep][targets] SKIP"; return 0
    fi
    mkdir -p "$out"
    echo "[prep][targets] START $(date -Iseconds)"
    nice -n 10 python scripts/session16/exp2_build_targets.py \
        --split-manifest "$SPLIT" --partition "$PARTITION" \
        --latents-dir "outputs/session28/latents/jepa_tf_noc_d64_s42" \
        --out "$out" \
        > "$out/build_targets.log" 2>&1 || { echo "[prep][targets] FAILED"; return 1; }
    echo "[prep][targets] DONE $(date -Iseconds)"
}

pod & POD_PID=$!
latents & LAT_PID=$!
dns_metrics & DNS_PID=$!
rc=0
wait $POD_PID || rc=1
wait $DNS_PID || rc=1
wait $LAT_PID || { rc=1; echo "[prep] latents failed; skipping targets"; }
if [[ $rc -eq 0 || -f outputs/session28/latents/jepa_tf_noc_d64_s42/test_b.npz ]]; then
    targets || rc=1
fi
echo "[prep] PHASE B PREP COMPLETE rc=$rc at $(date -Iseconds)"
exit $rc
