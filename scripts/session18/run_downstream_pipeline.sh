#!/usr/bin/env bash
# Session 18 B1 downstream pipeline: encode Fukami d=64 latents (if needed),
# run Markov-only + Full-context rollouts on all 7 baselines, fit per-baseline
# physical-metric probes, and build the headline comparison figure.
#
# Assumes upstream is complete:
#   outputs/session18/exp_b1/fukami_ae_d64/checkpoint_iter020000.pt
#   outputs/session18/exp_b1/predictor_{fukami_d3,fukami_d32,pod_d16,pod_d32,pod_d64,jepa_d64}/checkpoint_iter020000.pt
#
# Usage:
#   bash scripts/session18/run_downstream_pipeline.sh [gpu]
#     gpu: 0 (default) or 1.

set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
# shellcheck source=/dev/null
source "$REPO/.venv/bin/activate"
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

GPU="${1:-0}"

echo "==============================================================="
echo "Session 18 B1 downstream pipeline (gpu=$GPU)"
echo "==============================================================="

# 1. Encode Fukami d=64 latents (if missing)
if [[ ! -f "outputs/session18/exp_b1/latents_fukami_d64/train.npz" ]]; then
    if [[ -f "outputs/session18/exp_b1/fukami_ae_d64/checkpoint_iter020000.pt" ]]; then
        echo ">>> Encoding Fukami d=64 latents"
        python scripts/session18/encode_baseline_latents.py \
            --baseline fukami --d 64 \
            --checkpoint outputs/session18/exp_b1/fukami_ae_d64/checkpoint_iter020000.pt \
            --gpu "$GPU"
    else
        echo "WARNING: Fukami d=64 encoder checkpoint missing; skipping that baseline"
    fi
fi

# 2. Train Fukami d=64 predictor (if missing)
if [[ ! -f "outputs/session18/exp_b1/predictor_fukami_d64/checkpoint_iter020000.pt" ]]; then
    if [[ -f "outputs/session18/exp_b1/latents_fukami_d64/train.npz" ]]; then
        echo ">>> Training Fukami d=64 predictor"
        python scripts/session18/train_baseline_predictor.py \
            --latents-dir outputs/session18/exp_b1/latents_fukami_d64 \
            --tag fukami_d64 \
            --gpu "$GPU" \
            --seed 0 \
            --num-workers 0
    fi
fi

# 3. Rollouts on each baseline (Markov + Full)
declare -a tags=("fukami_d3" "fukami_d32" "fukami_d64" "pod_d16" "pod_d32" "pod_d64" "jepa_d64")
declare -a ds=(3 32 64 16 32 64 64)
declare -a kinds=("fukami" "fukami" "fukami" "pod" "pod" "pod" "jepa")

for i in "${!tags[@]}"; do
    tag="${tags[$i]}"
    pred="outputs/session18/exp_b1/predictor_${tag}/checkpoint_iter020000.pt"
    latents="outputs/session18/exp_b1/latents_${tag}"
    out="outputs/session18/exp_b1/rollouts_${tag}"
    if [[ ! -f "$pred" ]]; then
        echo "[skip $tag] predictor $pred missing"
        continue
    fi
    if [[ -f "$out/test_b.npz" && -f "$out/test_c.npz" ]]; then
        echo "[skip $tag] rollouts already present"
        continue
    fi
    echo ">>> Rollouts for $tag"
    python scripts/session18/eval_baseline_rollouts.py \
        --latents-dir "$latents" \
        --predictor "$pred" \
        --tag "$tag" \
        --gpu "$GPU"
done

# 4. Physical metrics
echo ">>> Physical metrics and bootstrap CIs"
declare -a present_tags=()
declare -a present_ds=()
declare -a present_kinds=()
for i in "${!tags[@]}"; do
    rollouts_dir="outputs/session18/exp_b1/rollouts_${tags[$i]}"
    if [[ -f "$rollouts_dir/test_b.npz" ]]; then
        present_tags+=("${tags[$i]}")
        present_ds+=("${ds[$i]}")
        present_kinds+=("${kinds[$i]}")
    fi
done
echo "Baselines with rollouts: ${present_tags[*]}"
python scripts/session18/physical_metrics_from_rollouts.py \
    --baselines "${present_tags[@]}" \
    --d-per-baseline "${present_ds[@]}" \
    --baseline-kind "${present_kinds[@]}"

# 5. Comparison figure
echo ">>> Comparison figure"
python scripts/session18/build_comparison_figure.py

echo "==============================================================="
echo "Session 18 B1 downstream pipeline complete"
echo "==============================================================="
echo "Headline table: outputs/session18/exp_b1/physical_closure_comparison.csv"
echo "Figure:         outputs/session18/figures/exp_b1_markov_closure_baselines.png"
