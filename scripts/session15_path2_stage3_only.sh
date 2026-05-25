#!/usr/bin/env bash
# Session 15 Path 2 Stage 3 only: assumes cache + manifest already built.
# Launches BOTH canonical (GPU 0) and reduced (GPU 1) variants in parallel.
# Each runs: encoder (5h) -> SL decoder (2h) -> extended-metrics eval (10 min).

set -euo pipefail
cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

ROOT="outputs/runs/session15/path2_meantrain"
mkdir -p "$ROOT"
LOG="$ROOT/stage3_relaunch.log"
echo "[s15-p2-relaunch] start at $(date -Iseconds)" | tee "$LOG"

launch_variant() {
    local variant=$1
    local gpu=$2
    local lambda_wake=$3
    local lambda_grad=$4
    local lambda_spec=$5
    local enc_dir="$ROOT/${variant}/encoder"
    local dec_dir="$enc_dir/decoder_specloss_recipe"
    local log="$ROOT/${variant}/launch.log"
    # Wipe stale crashed state
    rm -rf "$enc_dir"
    mkdir -p "$enc_dir" "$dec_dir"
    echo "[s15-p2-relaunch] $(date -Iseconds) launching $variant on gpu=$gpu" \
        "(lambda_wake=$lambda_wake gradient=$lambda_grad spectral_amp=$lambda_spec)" \
        | tee -a "$LOG"

    nohup bash -c "
        set -e
        # Use --partition v1_mean (which expects cache at PARENT/v1_mean/) instead
        # of overriding VORTEX_JEPA_CACHE. configs/splits/split_v1_mean.json is a
        # symlink to split_v1.json; wake_observables precomputed at
        # \$PREVENT_ROOT/data/processed/vortex-jepa/v1_mean/wake_observables/
        python -u -m src.training.train_jepa \\
            --partition v1_mean --all-train --max-iters 20000 --seed 42 \\
            --d 64 --B 16 --T 32 --H-roll 8 \\
            --lambda-sigreg 0.01 \\
            --lr-encoder 1.5e-4 --lr-predictor 5e-4 \\
            --weight-decay 0.05 --warmup-frac 0.05 \\
            --num-workers 4 --gpu $gpu \\
            --output-dir $enc_dir \\
            --log-every 50 --diagnostic-every 500 --checkpoint-every 2000 \\
            --projection-norm batchnorm --anticollapse sigreg \\
            --tag-suffix path2_${variant} \\
            --observable-head cl_future --observable-head-weight 0.01 \\
            --observable-head-deltas 0 \\
            --wake-observable-type patch_signed_spectrum --lambda-wake $lambda_wake \\
            --wake-loss smooth_l1 --wake-loss-beta 0.5 --wake-head-hidden 128 \\
            --omega-pipeline-manifest outputs/data_pipeline/v1_mean/manifest.json \\
            --wandb-mode offline 2>&1
        python -u scripts/session9_train_decoder.py \\
            --encoder-run $enc_dir \\
            --omega-pipeline-manifest outputs/data_pipeline/v1_mean/manifest.json \\
            --decoder-type lapfilm --decoder-upsample pixelshuffle \\
            --decoder-loss region_pyr_specloss \\
            --lambda-region 1.0 --lambda-pyramid 0.4 \\
            --lambda-gradient $lambda_grad --lambda-spectral-amp $lambda_spec \\
            --lambda-enstrophy 0.02 --lambda-circulation 0.01 \\
            --spectral-window hann --spectral-wake-only \\
            --max-iters 12000 --B 16 --T 32 --seed 42 \\
            --gpu $gpu --output-dir $dec_dir \\
            --eval-every 2000 --checkpoint-every 2000 --log-every 200 2>&1
        python -u scripts/session10_evaluate.py \\
            --encoder-run $enc_dir \\
            --decoder-run $dec_dir \\
            --decoder-checkpoint $dec_dir/decoder_iter012000.pt \\
            --gpu $gpu \\
            --output-json $dec_dir/eval/extended_metrics.json 2>&1
        echo \"[s15-p2-${variant}] all stages done at \$(date -Iseconds)\"
    " > "$log" 2>&1 &
    echo "[s15-p2-relaunch] $(date -Iseconds) $variant PID=$!" | tee -a "$LOG"
}

launch_variant "canonical"  0  1.0 1.0 1.0
launch_variant "reduced"    1  0.3 0.3 0.3

echo "[s15-p2-relaunch] both variants launched at $(date -Iseconds)" | tee -a "$LOG"
