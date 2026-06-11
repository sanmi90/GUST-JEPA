#!/usr/bin/env bash
# Session 28 training matrix: run ONE cell on ONE RTX 6000 card.
# Usage: _run_one.sh <cell_tag> <gpu_index>
#   cell_tag: one of the tags defined in the case statement below (master plan A1).
#   gpu_index: 0 or 1 (0-indexed selector into the RTX 6000 subset; D40 semantics).
#
# Every run: split v2p1, partition v2p1, omega pipeline v2p1, unconditioned
# (--predictor-cond-dim 0) except the jepa_tf_cond_* AD2 reference cells.
# Idempotent: skips if the final checkpoint exists. Logs to
# outputs/runs/session28/<tag>/train.log (fukami/bvae) or <tag>/encoder/train.log (jepa).
set -uo pipefail
cell=$1; gpu=$2
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
REPO=$(pwd)
source "$REPO/.venv/bin/activate"
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT="outputs/runs/session28"
SPLIT="configs/splits/split_v2p1.json"
MANIFEST="outputs/data_pipeline/v2p1/manifest.json"
PARTITION="v2p1"
NW=3   # 3 concurrent runs/card -> keep dataloader CPU pressure sane (session20 practice)

# ---------- shared flag groups ----------
jepa_common=(--all-train --max-iters 20000 --B 16 --T 32 --H-roll 8
    --lambda-sigreg 0.01 --lr-encoder 1.5e-4 --lr-predictor 5e-4 --weight-decay 0.05
    --warmup-frac 0.05 --grad-clip 1.0 --num-workers "$NW" --gpu "$gpu"
    --projection-norm batchnorm --anticollapse sigreg
    --observable-head cl_future --observable-head-weight 0.01 --observable-head-deltas 0
    --partition "$PARTITION" --split "$SPLIT" --omega-pipeline-manifest "$MANIFEST"
    --wandb-mode offline --log-every 200 --diagnostic-every 2000 --checkpoint-every 10000)
wake_on=(--wake-observable-type patch_signed_spectrum --lambda-wake 1.00
    --wake-loss smooth_l1 --wake-loss-beta 0.5 --wake-head-hidden 128)
lstm_pred=(--predictor-type lstm --predictor-hidden 256 --predictor-layers 3)

# Fukami B1 production recipe (SESSION18_B1_PROTOCOL as executed by
# scripts/session18/train_fukami_baselines.sh: charbonnier + ReLU + GroupNorm +
# future-lift {8,16,24} at weight 1.0).
fuk_b1=(--all-train --max-iters 20000 --B 16 --T 32 --gpu "$gpu"
    --partition "$PARTITION" --split "$SPLIT" --omega-pipeline-manifest "$MANIFEST"
    --recon-loss-type charbonnier --charbonnier-epsilon 0.05 --activation relu
    --observable-head cl_future --observable-head-deltas 8 16 24
    --observable-head-weight 1.0 --lambda-recon 1.0 --lambda-lift 1.0
    --lr 1e-3 --weight-decay 0.0 --warmup-frac 0.05 --grad-clip 1.0
    --num-workers "$NW" --wandb-mode offline)

# Fukami matched-head 2x2 control recipe (session20 _train_one.sh, T4 cells).
fuk_matched=(--all-train --max-iters 20000 --B 16 --T 32 --gpu "$gpu"
    --partition "$PARTITION" --split "$SPLIT" --omega-pipeline-manifest "$MANIFEST"
    --recon-loss-type mse
    --observable-head cl_future --observable-head-deltas 0 --lambda-lift 0.01
    --wake-observable-type patch_signed_spectrum --lambda-wake 1.0
    --wake-loss smooth_l1 --wake-loss-beta 0.5 --wake-head-hidden 128
    --num-workers "$NW" --wandb-mode offline
    --log-every 200 --diagnostic-every 2000 --checkpoint-every 10000)

run_jepa() {  # run_jepa <tag> <extra flags...>
    local tag=$1; shift
    local out="$ROOT/$tag/encoder"
    if [[ -f "$out/checkpoint_iter020000.pt" ]]; then
        echo "[s28][gpu$gpu] SKIP $tag (ckpt exists)"; return 0
    fi
    mkdir -p "$out"
    echo "[s28][gpu$gpu] START $tag at $(date -Iseconds)"
    python -u -m src.training.train_jepa "${jepa_common[@]}" "$@" \
        --tag-suffix "s28_${tag}" --output-dir "$out" > "$out/train.log" 2>&1
    local rc=$?
    echo "[s28][gpu$gpu] DONE $tag rc=$rc at $(date -Iseconds)"
    return $rc
}

run_fukami() {  # run_fukami <tag> <recipe-array-name> <extra flags...>
    local tag=$1; local recipe=$2; shift 2
    local out="$ROOT/$tag"
    local final_ckpt="checkpoint_iter020000.pt"
    [[ "$tag" == *i30k* ]] && final_ckpt="checkpoint_iter030000.pt"
    if [[ -f "$out/$final_ckpt" ]]; then
        echo "[s28][gpu$gpu] SKIP $tag (ckpt exists)"; return 0
    fi
    mkdir -p "$out"
    local -n rec="$recipe"
    echo "[s28][gpu$gpu] START $tag at $(date -Iseconds)"
    python -u scripts/session9_train_fukami.py "${rec[@]}" "$@" \
        --tag-suffix "s28_${tag}" --output-dir "$out" > "$out/train.log" 2>&1
    local rc=$?
    echo "[s28][gpu$gpu] DONE $tag rc=$rc at $(date -Iseconds)"
    return $rc
}

case "$cell" in
  # ---- T1 lead family: JEPA tf-no-c d=64 ----
  jepa_tf_noc_d64_s42) run_jepa "$cell" "${wake_on[@]}" --predictor-cond-dim 0 --d 64 --seed 42 ;;
  jepa_tf_noc_d64_s0)  run_jepa "$cell" "${wake_on[@]}" --predictor-cond-dim 0 --d 64 --seed 0 ;;
  jepa_tf_noc_d64_s1)  run_jepa "$cell" "${wake_on[@]}" --predictor-cond-dim 0 --d 64 --seed 1 ;;
  jepa_tf_noc_d64_s2)  run_jepa "$cell" "${wake_on[@]}" --predictor-cond-dim 0 --d 64 --seed 2 ;;

  # ---- T2 lstm-no-c d=64 ----
  jepa_lstm_noc_d64_s42) run_jepa "$cell" "${wake_on[@]}" "${lstm_pred[@]}" --predictor-cond-dim 0 --d 64 --seed 42 ;;
  jepa_lstm_noc_d64_s0)  run_jepa "$cell" "${wake_on[@]}" "${lstm_pred[@]}" --predictor-cond-dim 0 --d 64 --seed 0 ;;
  jepa_lstm_noc_d64_s1)  run_jepa "$cell" "${wake_on[@]}" "${lstm_pred[@]}" --predictor-cond-dim 0 --d 64 --seed 1 ;;

  # ---- T3 capacity ladder tf-no-c ----
  jepa_tf_noc_d4_s42)  run_jepa "$cell" "${wake_on[@]}" --predictor-cond-dim 0 --d 4  --seed 42 ;;
  jepa_tf_noc_d8_s42)  run_jepa "$cell" "${wake_on[@]}" --predictor-cond-dim 0 --d 8  --seed 42 ;;
  jepa_tf_noc_d16_s42) run_jepa "$cell" "${wake_on[@]}" --predictor-cond-dim 0 --d 16 --seed 42 ;;
  jepa_tf_noc_d16_s0)  run_jepa "$cell" "${wake_on[@]}" --predictor-cond-dim 0 --d 16 --seed 0 ;;
  jepa_tf_noc_d16_s1)  run_jepa "$cell" "${wake_on[@]}" --predictor-cond-dim 0 --d 16 --seed 1 ;;
  jepa_tf_noc_d32_s42) run_jepa "$cell" "${wake_on[@]}" --predictor-cond-dim 0 --d 32 --seed 42 ;;
  jepa_tf_noc_d32_s0)  run_jepa "$cell" "${wake_on[@]}" --predictor-cond-dim 0 --d 32 --seed 0 ;;
  jepa_tf_noc_d32_s1)  run_jepa "$cell" "${wake_on[@]}" --predictor-cond-dim 0 --d 32 --seed 1 ;;

  # ---- T4 2x2 control cells (predictive CNN+ViT cell = T1 seeds 0/1/2, PF-A2 reuse) ----
  ctrl_pred_cnn_s0) run_jepa "$cell" "${wake_on[@]}" --predictor-cond-dim 0 --encoder cnn_only --d 64 --seed 0 ;;
  ctrl_pred_cnn_s1) run_jepa "$cell" "${wake_on[@]}" --predictor-cond-dim 0 --encoder cnn_only --d 64 --seed 1 ;;
  ctrl_pred_cnn_s2) run_jepa "$cell" "${wake_on[@]}" --predictor-cond-dim 0 --encoder cnn_only --d 64 --seed 2 ;;
  ctrl_recon_cnnvit_s0) run_fukami "$cell" fuk_matched --encoder cnn_vit --d 64 --seed 0 ;;
  ctrl_recon_cnnvit_s1) run_fukami "$cell" fuk_matched --encoder cnn_vit --d 64 --seed 1 ;;
  ctrl_recon_cnnvit_s2) run_fukami "$cell" fuk_matched --encoder cnn_vit --d 64 --seed 2 ;;
  ctrl_recon_cnn_s0) run_fukami "$cell" fuk_matched --encoder cnn --d 64 --seed 0 ;;
  ctrl_recon_cnn_s1) run_fukami "$cell" fuk_matched --encoder cnn --d 64 --seed 1 ;;
  ctrl_recon_cnn_s2) run_fukami "$cell" fuk_matched --encoder cnn --d 64 --seed 2 ;;

  # ---- T5 wake-head-removed (lift-only) ----
  ctrl_pred_vit_nowake_s0) run_jepa "$cell" --predictor-cond-dim 0 --encoder hybrid --lambda-wake 0.0 --d 64 --seed 0 ;;
  ctrl_pred_vit_nowake_s1) run_jepa "$cell" --predictor-cond-dim 0 --encoder hybrid --lambda-wake 0.0 --d 64 --seed 1 ;;
  ctrl_pred_vit_nowake_s2) run_jepa "$cell" --predictor-cond-dim 0 --encoder hybrid --lambda-wake 0.0 --d 64 --seed 2 ;;

  # ---- T6 conditioned reference (AD2; the ONLY conditioned cells) ----
  jepa_tf_cond_d64_s42) run_jepa "$cell" "${wake_on[@]}" --predictor-cond-dim 3 --d 64 --seed 42 ;;
  jepa_tf_cond_d64_s0)  run_jepa "$cell" "${wake_on[@]}" --predictor-cond-dim 3 --d 64 --seed 0 ;;
  jepa_tf_cond_d64_s1)  run_jepa "$cell" "${wake_on[@]}" --predictor-cond-dim 3 --d 64 --seed 1 ;;

  # ---- T7 Fukami AE: B1 production rows ----
  fukami_d3_s42)  run_fukami "$cell" fuk_b1 --d 3  --seed 42 ;;
  fukami_d16_s42) run_fukami "$cell" fuk_b1 --d 16 --seed 42 ;;
  fukami_d32_s42) run_fukami "$cell" fuk_b1 --d 32 --seed 42 ;;
  fukami_d64_s42) run_fukami "$cell" fuk_b1 --d 64 --seed 42 ;;
  fukami_d64_s0)  run_fukami "$cell" fuk_b1 --d 64 --seed 0 ;;
  fukami_d64_s1)  run_fukami "$cell" fuk_b1 --d 64 --seed 1 ;;
  fukami_d64_s2)  run_fukami "$cell" fuk_b1 --d 64 --seed 2 ;;

  # ---- T7 L-curve sweep (paper-faithful objective: mse, current-frame C_L, lambda_lift = beta) ----
  fukami_lcurve_b0p005) run_fukami "$cell" fuk_b1 --d 3 --seed 42 --recon-loss-type mse --observable-head-deltas 0 --lambda-lift 0.005 ;;
  fukami_lcurve_b0p01)  run_fukami "$cell" fuk_b1 --d 3 --seed 42 --recon-loss-type mse --observable-head-deltas 0 --lambda-lift 0.01 ;;
  fukami_lcurve_b0p02)  run_fukami "$cell" fuk_b1 --d 3 --seed 42 --recon-loss-type mse --observable-head-deltas 0 --lambda-lift 0.02 ;;
  fukami_lcurve_b0p05)  run_fukami "$cell" fuk_b1 --d 3 --seed 42 --recon-loss-type mse --observable-head-deltas 0 --lambda-lift 0.05 ;;
  fukami_lcurve_b0p1)   run_fukami "$cell" fuk_b1 --d 3 --seed 42 --recon-loss-type mse --observable-head-deltas 0 --lambda-lift 0.1 ;;

  # ---- T7 budget-parity sweep at d=64 (closes M2(ii); prod (1e-3, 20k) is the 4th point) ----
  fukami_budget_lr1e3_i30k) run_fukami "$cell" fuk_b1 --d 64 --seed 42 --max-iters 30000 ;;
  fukami_budget_lr3e4_i30k) run_fukami "$cell" fuk_b1 --d 64 --seed 42 --lr 3e-4 --max-iters 30000 ;;
  fukami_budget_lr3e4_i20k) run_fukami "$cell" fuk_b1 --d 64 --seed 42 --lr 3e-4 ;;

  # ---- T8 beta-VAE (AD3; src/baselines/solera_rico.py). BETA: verify the value
  # against the Nat. Commun. 2024 recipe (literature check L5) BEFORE launching
  # these cells; 1e-3 is the placeholder default. faithful = lift head only
  # (parallel to the Fukami B1 recipe); matched = lift + wake (parallel to T4). ----
  bvae_faith_d64_s42) run_fukami "$cell" fuk_b1 --vae --beta 1e-3 --recon-loss-type mse --d 64 --seed 42 ;;
  bvae_faith_d64_s0)  run_fukami "$cell" fuk_b1 --vae --beta 1e-3 --recon-loss-type mse --d 64 --seed 0 ;;
  bvae_faith_d64_s1)  run_fukami "$cell" fuk_b1 --vae --beta 1e-3 --recon-loss-type mse --d 64 --seed 1 ;;
  bvae_match_d64_s42) run_fukami "$cell" fuk_matched --vae --beta 1e-3 --encoder cnn --d 64 --seed 42 ;;
  bvae_match_d64_s0)  run_fukami "$cell" fuk_matched --vae --beta 1e-3 --encoder cnn --d 64 --seed 0 ;;
  bvae_match_d64_s1)  run_fukami "$cell" fuk_matched --vae --beta 1e-3 --encoder cnn --d 64 --seed 1 ;;

  *) echo "[s28] unknown cell '$cell'"; exit 2 ;;
esac
