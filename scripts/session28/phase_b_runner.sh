#!/usr/bin/env bash
# Phase B (Session 29) overnight chain on GPU 1 (author go: 2026-06-12 22:52).
# Waits for the T8 bvae production cells to finish (the bvae_sweep_runner's
# terminal log line), then:
#   stage 1: extract latents for EVERY remaining trained checkpoint
#            (B0 pre-flight completion; ~33 short encoder passes);
#   stage 2: train the B1 matched predictors, one per (family, d), on the
#            s42/canonical latents, per SESSION18_B1_PROTOCOL with the two
#            frozen v2p1 amendments: --cond-dim 0 (unconditioned mandate) and
#            --no-output-bn (D129). bvae family row = the FAITHFUL variant
#            (parallel to the Fukami B1 recipe); the matched-head variant is
#            kept for the GC dossier.
#   stage 3: full-context + markov rollouts for every matched predictor on
#            test_b and test_c (eval_baseline_rollouts.py; the protocol
#            selects z_full downstream; z_markov is retired but kept on disk).
# Thread caps + nice inherited from the 2026-06-12 hygiene block.
# Idempotent: every stage skips work whose outputs exist.
set -uo pipefail
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export OMP_WAIT_POLICY=PASSIVE
GPU=1
SPLIT="configs/splits/split_v2p1.json"
MANIFEST="outputs/data_pipeline/v2p1/manifest.json"
RUNS=outputs/runs/session28
LAT=outputs/session28/latents
PRED=outputs/session28/predictors
ROLL=outputs/session28/rollouts

echo "[phaseB] waiting for T8 completion (bvae_sweep_runner.log)"
while ! grep -q "T8 complete" outputs/runs/session28/bvae_sweep_runner.log 2>/dev/null; do sleep 300; done
echo "[phaseB] GPU $GPU free at $(date -Iseconds); stage 1: latents"

encode() {  # encode <baseline> <tag> <ckpt>
    local baseline=$1 tag=$2 ckpt=$3
    local out="$LAT/$tag"
    # d from the tag (..._dNN_...); control/budget tags carry no _dNN and are all d=64
    local d
    d=$(echo "$tag" | grep -oE '_d[0-9]+' | head -1 | tr -dc '0-9')
    d=${d:-64}
    if [[ -f "$out/test_b.npz" ]]; then echo "[phaseB][lat] SKIP $tag"; return 0; fi
    if [[ ! -f "$ckpt" ]]; then echo "[phaseB][lat] MISSING ckpt for $tag ($ckpt)"; return 1; fi
    mkdir -p "$out"
    nice -n 10 python scripts/session18/encode_baseline_latents.py \
        --baseline "$baseline" --d "$d" --checkpoint "$ckpt" \
        --partition v2p1 --split "$SPLIT" --pipeline-manifest "$MANIFEST" \
        --splits train test_a test_b test_c \
        --gpu "$GPU" --output-dir "$out" > "$out.encode.log" 2>&1 \
        || { echo "[phaseB][lat] FAILED $tag"; return 1; }
    echo "[phaseB][lat] DONE $tag"
}

rc=0
# JEPA-type checkpoints (d read from the checkpoint args; --d 0 is ignored by
# the jepa loader, which reconstructs from the blob)
for tag in jepa_lstm_noc_d64_s1 \
           jepa_tf_noc_d4_s42 jepa_tf_noc_d8_s42 \
           jepa_tf_noc_d16_s42 jepa_tf_noc_d16_s0 jepa_tf_noc_d16_s1 \
           jepa_tf_noc_d32_s42 jepa_tf_noc_d32_s0 jepa_tf_noc_d32_s1 \
           jepa_tf_cond_d64_s42 jepa_tf_cond_d64_s0 jepa_tf_cond_d64_s1 \
           ctrl_pred_cnn_s0 ctrl_pred_cnn_s1 ctrl_pred_cnn_s2 \
           ctrl_pred_vit_nowake_s0 ctrl_pred_vit_nowake_s1 ctrl_pred_vit_nowake_s2; do
    encode jepa "$tag" "$RUNS/$tag/encoder/checkpoint_iter020000.pt" || rc=1
done
# Fukami-type
for tag in fukami_d3_s42 fukami_d16_s42 fukami_d32_s42 \
           fukami_d64_s0 fukami_d64_s1 fukami_d64_s2 \
           fukami_budget_lr3e4_i20k \
           ctrl_recon_cnn_s0 ctrl_recon_cnn_s1 ctrl_recon_cnn_s2 \
           ctrl_recon_cnnvit_s0 ctrl_recon_cnnvit_s1 ctrl_recon_cnnvit_s2 \
           bvae_faith_d64_s42 bvae_faith_d64_s0 bvae_faith_d64_s1 \
           bvae_match_d64_s42 bvae_match_d64_s0 bvae_match_d64_s1; do
    encode fukami "$tag" "$RUNS/$tag/checkpoint_iter020000.pt" || rc=1
done
for tag in fukami_budget_lr1e3_i30k fukami_budget_lr3e4_i30k; do
    encode fukami "$tag" "$RUNS/$tag/checkpoint_iter030000.pt" || rc=1
done
echo "[phaseB] stage 1 done rc=$rc at $(date -Iseconds)"

echo "[phaseB] stage 2: B1 matched predictors (cond-dim 0, no-output-bn)"
declare -A PRED_LAT=(
    [jepa_d16]="$LAT/jepa_tf_noc_d16_s42"   [jepa_d32]="$LAT/jepa_tf_noc_d32_s42"
    [jepa_d64]="$LAT/jepa_tf_noc_d64_s42"
    [fukami_d3]="$LAT/fukami_d3_s42"        [fukami_d16]="$LAT/fukami_d16_s42"
    [fukami_d32]="$LAT/fukami_d32_s42"      [fukami_d64]="$LAT/fukami_d64_s42"
    [pod_d16]="$LAT/pod_d16" [pod_d32]="$LAT/pod_d32" [pod_d64]="$LAT/pod_d64"
    [bvae_d64]="$LAT/bvae_faith_d64_s42"
)
train_pred() {  # train_pred <key>
    local key=$1; local lat="${PRED_LAT[$key]}"
    local out="$PRED/$key"
    if [[ -f "$out/checkpoint_iter020000.pt" ]]; then echo "[phaseB][pred] SKIP $key"; return 0; fi
    if [[ ! -f "$lat/train.npz" ]]; then echo "[phaseB][pred] MISSING latents for $key"; return 1; fi
    mkdir -p "$out"
    nice -n 10 python scripts/session18/train_baseline_predictor.py \
        --latents-dir "$lat" --tag "s28_$key" --gpu "$GPU" --seed 42 \
        --cond-dim 0 --no-output-bn \
        --output-dir "$out" > "$out/train.log" 2>&1 \
        || { echo "[phaseB][pred] FAILED $key"; return 1; }
    echo "[phaseB][pred] DONE $key at $(date -Iseconds)"
}
KEYS=(jepa_d64 fukami_d64 pod_d64 bvae_d64 jepa_d32 fukami_d32 pod_d32 jepa_d16 fukami_d16 pod_d16 fukami_d3)
i=0
while [[ $i -lt ${#KEYS[@]} ]]; do
    pids=()
    for k in "${KEYS[@]:$i:3}"; do train_pred "$k" & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    i=$((i+3))
done
echo "[phaseB] stage 2 done rc=$rc at $(date -Iseconds)"

echo "[phaseB] stage 3: rollouts (full-context primary, markov retired-on-disk)"
for key in "${KEYS[@]}"; do
    out="$ROLL/$key"
    if [[ -f "$out/test_b.npz" ]]; then echo "[phaseB][roll] SKIP $key"; continue; fi
    mkdir -p "$out"
    nice -n 10 python scripts/session18/eval_baseline_rollouts.py \
        --latents-dir "${PRED_LAT[$key]}" \
        --predictor "$PRED/$key/checkpoint_iter020000.pt" \
        --tag "s28_$key" --gpu "$GPU" \
        --output-dir "$out" > "$out/eval.log" 2>&1 \
        || { echo "[phaseB][roll] FAILED $key"; rc=1; continue; }
    echo "[phaseB][roll] DONE $key"
done
echo "[phaseB] PHASE B GPU CHAIN COMPLETE rc=$rc at $(date -Iseconds)"
