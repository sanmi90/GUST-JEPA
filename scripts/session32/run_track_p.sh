#!/bin/bash
# SESSION 32 Track P -- pooled canonical matrix on v2.2 (D227, full 6-model matrix).
# 5 kit rows via train_canonical (latent:pooled) + bvae via the native beta-VAE path
# (session9_train_fukami --vae). jepa_wake_pool is REUSED from Session 31 jepa_pool
# (D226), not retrained here. Budget = 10000 iters to match the spatial canonical spine.
#
# Two RTX 6000 cards, 2-at-a-time. Core-split 0-7 / 8-15, OMP=4, num-workers 3 so
# >=64 cores stay free for asolera. Offline W&B, group partition_v2p2, tier=pooled.
set -u
cd /home/carlos/GUST-JEPA
source .venv/bin/activate
export PREVENT_ROOT=$HOME/PREVENT WANDB_PROJECT=vortex-jepa OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

RUNS=outputs/runs/session32
PIPE=outputs/data_pipeline/v2p2/manifest.json
SPLIT=configs/splits/split_v2p2.json

run_kit() {  # model gpu cores
  local m=$1 gpu=$2 cores=$3
  mkdir -p "$RUNS/$m"
  taskset -c "$cores" python -m src.training.train_canonical \
    --config configs/ablation/"$m".yaml \
    --partition v2p2 --pipeline-manifest "$PIPE" \
    --gpu "$gpu" --seed 0 --max-iters 10000 --num-workers 3 \
    --diagnostic-every 1000 --checkpoint-every 2500 --log-every 200 \
    --wandb-mode offline \
    --out "$RUNS/$m" > "$RUNS/$m/train.log" 2>&1 || echo "[track-p] FAILED: $m" >> "$RUNS/failures.log"
}

run_bvae() {  # gpu cores
  local gpu=$1 cores=$2
  mkdir -p "$RUNS/bvae"
  taskset -c "$cores" python -u scripts/session9_train_fukami.py \
    --all-train --max-iters 10000 --B 16 --T 32 --gpu "$gpu" \
    --partition v2p2 --split "$SPLIT" --omega-pipeline-manifest "$PIPE" \
    --recon-loss-type mse \
    --observable-head cl_future --observable-head-deltas 0 --lambda-lift 0.01 \
    --wake-observable-type patch_signed_spectrum --lambda-wake 1.0 \
    --wake-loss smooth_l1 --wake-loss-beta 0.5 --wake-head-hidden 128 \
    --vae --beta 0.0025 --beta-warmup-frac 0.02 --encoder cnn --d 32 \
    --seed 0 --num-workers 3 --wandb-mode offline \
    --log-every 200 --diagnostic-every 2000 --checkpoint-every 10000 \
    --output-dir "$RUNS/bvae" > "$RUNS/bvae/train.log" 2>&1 || echo "[track-p] FAILED: bvae" >> "$RUNS/failures.log"
}

echo "[track-p] wave 1: jepa_nowake_pool (gpu0) + ae_wake_pool (gpu1) @ $(date -Iseconds)"
run_kit jepa_nowake_pool 0 0-7 &
run_kit ae_wake_pool     1 8-15 &
wait

echo "[track-p] wave 2: ae_nowake_pool (gpu0) + supervised_only_pool (gpu1) @ $(date -Iseconds)"
run_kit ae_nowake_pool       0 0-7 &
run_kit supervised_only_pool 1 8-15 &
wait

echo "[track-p] wave 3: regAE_pool (gpu0) + bvae (gpu1) @ $(date -Iseconds)"
run_kit regAE_pool 0 0-7 &
run_bvae           1 8-15 &
wait

echo "[track-p] complete @ $(date -Iseconds)"
