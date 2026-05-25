#!/usr/bin/env bash
# Session 14 Thrust 6: extended-metrics eval for the 3 JEPA d=64 seeds (paired
# with their SL decoders) and the 3 Fukami d=32 seeds, then Welch t-test on
# the per-encounter metrics. Fires after the SL decoder queue drains.
#
# Steps:
#   1. Eval each JEPA seed: scripts/session10_evaluate.py on
#      (jepa_d64_seed*/encoder, decoder_specloss_recipe/decoder_iter012000.pt)
#   2. Eval each Fukami seed: TODO (Fukami AE needs its own eval path; the
#      training-time loss alone is not enough; would need to add a Fukami
#      reconstruction eval pass that produces SSIM + spectral metrics).
#      For now we just do the JEPA side and the comparison vs Fukami is via
#      training-loss only (covered in D105 partial).
#   3. Compute Welch t-test across the 3 JEPA seeds vs the production
#      S12_E_d64 + SL D99 numbers (one-sample t-test, n=3) on every
#      extended_metrics key.
#
# Usage:
#   bash scripts/session14_thrust6_eval_and_welch.sh <gpu>

set -euo pipefail

GPU=${1:?gpu required (0|1)}

cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

ROOT="outputs/runs/session14/thrust6"
LOG="${ROOT}/welch_eval.log"
mkdir -p "$(dirname "$LOG")"
echo "[s14-welch] starting JEPA-seed extended eval on gpu=$GPU" | tee "$LOG"

for seed in 0 1 2; do
    enc="${ROOT}/jepa_d64_seed${seed}/encoder"
    dec="${enc}/decoder_specloss_recipe"
    ckpt="${dec}/decoder_iter012000.pt"
    if [ ! -f "$ckpt" ]; then
        echo "[s14-welch] MISSING $ckpt; skipping seed $seed" | tee -a "$LOG"
        continue
    fi
    out_dir="${dec}/eval"
    mkdir -p "$out_dir"
    echo "[s14-welch] $(date -Iseconds) eval jepa_d64_seed${seed}" | tee -a "$LOG"
    python -u scripts/session10_evaluate.py \
        --encoder-run "$enc" \
        --decoder-run "$dec" \
        --decoder-checkpoint "$ckpt" \
        --gpu "$GPU" \
        --output-json "$out_dir/extended_metrics.json" \
        2>&1 | tee -a "$LOG"
done

echo "[s14-welch] $(date -Iseconds) running cross-seed Welch t-tests" | tee -a "$LOG"
python scripts/session14_thrust6_welch.py 2>&1 | tee -a "$LOG"

echo "[s14-welch] $(date -Iseconds) all done" | tee -a "$LOG"
