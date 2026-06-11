#!/usr/bin/env bash
# Session 28 training-matrix queue (master plan A1, gate GA1).
#
# Two per-GPU queues of serial WAVES; each wave is a comma-separated list of
# cell tags run CONCURRENTLY on that card. Packing rules learned in session 20:
#   - JEPA cells (train_jepa): up to 3 concurrent per RTX 6000 card.
#   - Fukami/AE cells (carry a full-field decoder): at most 2 per card
#     (3-packs OOM-killed twice on 2026-05-29, rc=137).
# Queue discipline (plan A1): T1/T2 first, then T4/T5/T6, then T3/T7
# interleaved. T8 (beta-VAE) launches separately once the port lands;
# T9 (decoders) launches after the relevant encoders are frozen.
#
# Idempotent: _run_one.sh skips any cell whose final checkpoint exists, so the
# whole launcher can be re-run after an interruption.
#
# Usage:
#   bash scripts/session28/launch_queue.sh           # both cards, full queue
#   bash scripts/session28/launch_queue.sh 0         # only the gpu-0 queue
set -uo pipefail
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
REPO=$(pwd)
ONE="$REPO/scripts/session28/_run_one.sh"
ROOT="outputs/runs/session28"
mkdir -p "$ROOT"
ONLY="${1:-}"

# Pre-flight hard gates (cheap, fail fast).
[[ -f configs/splits/split_v2p1.json ]] || { echo "FATAL: split_v2p1.json missing"; exit 1; }
[[ -f outputs/data_pipeline/v2p1/manifest.json ]] || { echo "FATAL: v2p1 pipeline manifest missing"; exit 1; }
CACHE="${VORTEX_JEPA_CACHE:-${PREVENT_ROOT:-$HOME/PREVENT}/data/processed/vortex-jepa}"
[[ -e "$CACHE/v2p1" ]] || { echo "FATAL: cache partition symlink $CACHE/v2p1 missing (ln -s v1 $CACHE/v2p1)"; exit 1; }
[[ -f "$CACHE/v2p1/wake_observables/_train_stats.json" ]] || { echo "FATAL: wake observable train stats missing"; exit 1; }
grep -q '"v2p1"' "$CACHE/v2p1/wake_observables/_manifest.json" 2>/dev/null \
    || { echo "FATAL: wake observables not recomputed for v2p1 (run session11_precompute_wake_observables.py with the v2p1 split first)"; exit 1; }

# Waves per GPU. Edit here, not in _run_one.sh.
WAVES_GPU0=(
  "jepa_tf_noc_d64_s42,jepa_tf_noc_d64_s0,jepa_lstm_noc_d64_s42"
  "jepa_lstm_noc_d64_s1,jepa_tf_cond_d64_s42,jepa_tf_cond_d64_s0"
  "ctrl_pred_cnn_s2,ctrl_pred_vit_nowake_s0,ctrl_pred_vit_nowake_s1"
  "jepa_tf_noc_d16_s42,jepa_tf_noc_d16_s0,jepa_tf_noc_d16_s1"
  "fukami_lcurve_b0p005,fukami_lcurve_b0p01"
  "fukami_lcurve_b0p02,fukami_lcurve_b0p05"
  "fukami_lcurve_b0p1,fukami_d3_s42"
  "fukami_d16_s42,fukami_d32_s42"
  "ctrl_recon_cnn_s0,ctrl_recon_cnn_s1"
  "ctrl_recon_cnn_s2,fukami_budget_lr3e4_i20k"
  "fukami_budget_lr1e3_i30k,fukami_budget_lr3e4_i30k"
)
WAVES_GPU1=(
  "jepa_tf_noc_d64_s1,jepa_tf_noc_d64_s2,jepa_lstm_noc_d64_s0"
  "jepa_tf_cond_d64_s1,ctrl_pred_cnn_s0,ctrl_pred_cnn_s1"
  "ctrl_pred_vit_nowake_s2,jepa_tf_noc_d4_s42,jepa_tf_noc_d8_s42"
  "jepa_tf_noc_d32_s42,jepa_tf_noc_d32_s0,jepa_tf_noc_d32_s1"
  "fukami_d64_s42,fukami_d64_s0"
  "fukami_d64_s1,fukami_d64_s2"
  "ctrl_recon_cnnvit_s0,ctrl_recon_cnnvit_s1"
  "ctrl_recon_cnnvit_s2,bvae_lcurve_b5em4"
  "bvae_lcurve_b1em3,bvae_lcurve_b2p5em3"
  "bvae_lcurve_b5em3,bvae_lcurve_b1em2"
  "bvae_faith_d64_s42,bvae_faith_d64_s0"
  "bvae_faith_d64_s1,bvae_match_d64_s42"
  "bvae_match_d64_s0,bvae_match_d64_s1"
)
# T8 ordering note (2026-06-11): the bvae L-curve sweep waves precede the
# production bvae cells, and the production cells fail fast until
# outputs/session28/bvae_beta_pin.json exists (written by pick_bvae_beta.py
# from the sweep). The queue drivers launched at 10:40 on 2026-06-11 hold the
# PRE-sweep wave list in memory, so their bvae production cells fail fast
# harmlessly; scripts/session28/bvae_sweep_runner.sh waits for that queue to
# complete, re-invokes this launcher for GPU 1 (idempotent SKIPs), pins the
# elbow, and re-invokes once more for the production cells.

drain() {  # drain <gpu> <wave...>
    local gpu=$1; shift
    local wi=0
    for wave in "$@"; do
        wi=$((wi+1))
        IFS=',' read -ra cells <<< "$wave"
        echo "[queue][gpu$gpu] wave $wi/${#@}: ${wave} at $(date -Iseconds)"
        local pids=()
        for c in "${cells[@]}"; do
            bash "$ONE" "$c" "$gpu" &
            pids+=($!)
        done
        local rc=0
        for p in "${pids[@]}"; do wait "$p" || rc=1; done
        echo "[queue][gpu$gpu] wave $wi done (rc=$rc) at $(date -Iseconds)"
    done
    echo "[queue][gpu$gpu] QUEUE COMPLETE at $(date -Iseconds)"
}

if [[ -z "$ONLY" || "$ONLY" == "0" ]]; then
    drain 0 "${WAVES_GPU0[@]}" >> "$ROOT/queue_gpu0.log" 2>&1 &
    P0=$!
    echo "[queue] gpu0 driver pid $P0 -> $ROOT/queue_gpu0.log"
fi
if [[ -z "$ONLY" || "$ONLY" == "1" ]]; then
    drain 1 "${WAVES_GPU1[@]}" >> "$ROOT/queue_gpu1.log" 2>&1 &
    P1=$!
    echo "[queue] gpu1 driver pid $P1 -> $ROOT/queue_gpu1.log"
fi
wait
echo "[queue] ALL QUEUES COMPLETE at $(date -Iseconds)"
