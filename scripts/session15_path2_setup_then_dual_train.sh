#!/usr/bin/env bash
# Session 15 Path 2: build v1_mean cache + pipeline once, then launch two
# JEPA d=64 + SL trainings in parallel on GPU 0 and GPU 1.
#
#   Variant A (canonical):  same loss weights as slice production
#     lambda_wake=1.0, lambda_gradient=1.0, lambda_spectral_amp=1.0
#
#   Variant B (reduced):    weights motivated by the spanwise-mean
#                           data's reduced 3D structure
#     lambda_wake=0.3, lambda_gradient=0.3, lambda_spectral_amp=0.3
#
# Both train E d=64 (5h) + SL decoder (2h) + eval (10 min) end-to-end.
# Total wall ~8 h with the two GPUs in parallel.

set -euo pipefail
cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

ROOT="outputs/runs/session15/path2_meantrain"
mkdir -p "$ROOT"
LOG="$ROOT/setup.log"
echo "[s15-p2] start at $(date -Iseconds)" | tee "$LOG"

# Stage 1: build v1_mean cache (one-shot, ~30-60 min I/O)
echo "[s15-p2] Stage 1: building v1_mean cache" | tee -a "$LOG"
python scripts/build_omega_mean_cache.py --split configs/splits/split_v1.json \
    2>&1 | tee -a "$LOG"

# Stage 2: build v1_mean pipeline manifest (one-shot)
echo "[s15-p2] Stage 2: building v1_mean pipeline manifest" | tee -a "$LOG"
python scripts/build_omega_mean_pipeline.py --split configs/splits/split_v1.json \
    2>&1 | tee -a "$LOG"

# Stage 3: launch BOTH variants in parallel
echo "[s15-p2] Stage 3: launching dual variant training on GPUs 0+1" | tee -a "$LOG"

launch_variant() {
    local variant=$1
    local gpu=$2
    local lambda_wake=$3
    local lambda_grad=$4
    local lambda_spec=$5
    local enc_dir="$ROOT/${variant}/encoder"
    local dec_dir="$enc_dir/decoder_specloss_recipe"
    local log="$ROOT/${variant}/launch.log"
    mkdir -p "$enc_dir" "$dec_dir"
    echo "[s15-p2] $(date -Iseconds) launching $variant on gpu=$gpu" \
        "(lambda_wake=$lambda_wake gradient=$lambda_grad spectral_amp=$lambda_spec)" \
        | tee -a "$LOG"

    # Chain encoder -> decoder -> eval in a single background subshell.
    nohup bash -c "
        set -e
        export VORTEX_JEPA_CACHE=\"\$PREVENT_ROOT/data/processed/vortex-jepa/v1_mean\"
        # ENCODER (5h)
        python -u -m src.training.train_jepa \\
            --all-train --max-iters 20000 --seed 42 \\
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
        # SL DECODER (2h)
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
        # EXTENDED METRICS EVAL (10 min)
        python -u scripts/session10_evaluate.py \\
            --encoder-run $enc_dir \\
            --decoder-run $dec_dir \\
            --decoder-checkpoint $dec_dir/decoder_iter012000.pt \\
            --gpu $gpu \\
            --output-json $dec_dir/eval/extended_metrics.json 2>&1
        echo \"[s15-p2-${variant}] all stages done at \$(date -Iseconds)\"
    " > "$log" 2>&1 &
    echo "[s15-p2] $(date -Iseconds) $variant PID=$!" | tee -a "$LOG"
}

launch_variant "canonical"  0  1.0 1.0 1.0
launch_variant "reduced"    1  0.3 0.3 0.3

echo "[s15-p2] both variants launched at $(date -Iseconds); see $ROOT for logs" \
    | tee -a "$LOG"
