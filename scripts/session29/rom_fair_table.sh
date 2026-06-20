#!/usr/bin/env bash
# FAIR ROM field-forecast table (Solera-Rico style): decode each model's ROLLED
# latent at horizon h and score SSIM/MSE vs the true DNS field, on BOTH test_b
# (in-envelope) and test_c (OOD, G=+4). All four models share the SAME matched
# predictor recipe so only the latent + decoder vary (apples-to-apples ROM).
#
#   jepa        per-frame JEPA (paper model, lift+wake heads)   matched predictor
#   regae       reconstructive observable-augmented AE          matched predictor
#   bvae_match  beta-VAE + matched lift+wake heads (fuk_matched) matched predictor
#   vjepa_best  FULL V-JEPA 2.1: dense ctx (lam 0.5) + deep-SS  matched predictor
#               (n_levels 4) + lift+wake heads  == "all the tricks"
#
# Representative seed s0 (firm seeds later). RTX 6000 only via --gpu {0,1} ->
# require_rtx6000; CPU-capped (taskset 0-31 leaves >=64 cores for asolera);
# idempotent (skips any step whose output already exists).
set -uo pipefail
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}" WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SPLIT="configs/splits/split_v2p1.json"; MAN="outputs/data_pipeline/v2p1/manifest.json"
RUN=outputs/runs/session29/vjepa; LAT=outputs/session28/latents
PRED=outputs/session29/lowd_predictors; ROLL=outputs/session29/lowd_rollouts
DEC=outputs/runs/session29; ROM=outputs/session29/rom_field_forecast
LOCK=outputs/runs/session29/rom_fair_table.lock
[ -f "$LOCK" ] && { echo "[rft] lock exists; exit"; exit 0; }
mkdir -p "$ROM"; echo "$$ $(date -Iseconds)" > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
echo "[rft] ===== FAIR ROM TABLE START $(date -Iseconds) ====="

# --- Phase 0: fill missing rollouts/latents (cheap, 2-packed) -----------------
# 0a) regae: re-roll its existing matched predictor over BOTH splits (cnn_vit_s0
#     has test_c; the prior rollout only wrote test_b).
regae_reroll(){ local g=$1
  [ -f "$ROLL/regae_d64/test_c.npz" ] && { echo "[rft] skip regae reroll"; return 0; }
  echo "[rft] regae re-roll (tb+tc) gpu$g"
  taskset -c 0-31 nice -n 10 python scripts/session18/eval_baseline_rollouts.py \
    --latents-dir "$LAT/regae/cnn_vit_s0" \
    --predictor "$PRED/regae_d64/checkpoint_iter020000.pt" \
    --tag s29_regae_d64 --gpu "$g" --output-dir "$ROLL/regae_d64" \
    >"$ROLL/regae_d64/reroll.log" 2>&1 || echo "[rft] regae reroll FAIL"; }
# 0b) vjepa_best: extract FINE latents (all splits) with the n_levels-aware loader.
vbest_extract(){ local g=$1; local o=$LAT/vjepa_best_fine_s0
  [ -f "$o/test_c.npz" ] && { echo "[rft] skip vbest extract"; return 0; }
  mkdir -p "$o"; echo "[rft] vjepa_best extract (all splits) gpu$g"
  taskset -c 0-31 python scripts/session18/encode_baseline_latents.py --baseline vjepa --d 64 \
    --vjepa-eval-stride 8 --vjepa-eval-interp linear \
    --checkpoint "$RUN/vjepa_best_s0/checkpoint_iter020000.pt" --partition v2p1 --split "$SPLIT" \
    --pipeline-manifest "$MAN" --splits train test_a test_b test_c --gpu "$g" \
    --output-dir "$o" >"$o/encode.log" 2>&1 || echo "[rft] vbest extract FAIL"; }

echo "[rft] PHASE 0: regae reroll (gpu0) + vjepa_best extract (gpu1)"
regae_reroll 0 & vbest_extract 1 & wait

# --- Phase 1: matched predictors for the two new models (cheap, 2-packed) ------
mk_pred(){ local latd=$1 tag=$2 g=$3; local p=$PRED/$tag
  [ -f "$p/checkpoint_iter020000.pt" ] && { echo "[rft] skip pred $tag"; return 0; }
  mkdir -p "$p"; echo "[rft] predictor $tag gpu$g"
  taskset -c 0-31 nice -n 10 python scripts/session18/train_baseline_predictor.py \
    --latents-dir "$latd" --tag s29_$tag --gpu "$g" --seed 42 --cond-dim 0 --no-output-bn \
    --output-dir "$p" >"$p/train.log" 2>&1 || echo "[rft] pred $tag FAIL"; }
echo "[rft] PHASE 1: predictors (vjepa_best gpu0 + bvae_match gpu1)"
mk_pred "$LAT/vjepa_best_fine_s0" vjepa_best_matched_s0 0 &
mk_pred "$LAT/bvae_match_d64_s0"  bvae_match_matched_s0 1 & wait

# --- Phase 2: rollouts for the two new models (cheap, tb+tc, 2-packed) ---------
mk_roll(){ local latd=$1 tag=$2 g=$3; local p=$PRED/$tag r=$ROLL/$tag
  [ -f "$r/test_c.npz" ] && { echo "[rft] skip roll $tag"; return 0; }
  mkdir -p "$r"; echo "[rft] rollout $tag (tb+tc) gpu$g"
  taskset -c 0-31 nice -n 10 python scripts/session18/eval_baseline_rollouts.py \
    --latents-dir "$latd" --predictor "$p/checkpoint_iter020000.pt" \
    --tag s29_$tag --gpu "$g" --output-dir "$r" >"$r/eval.log" 2>&1 || echo "[rft] roll $tag FAIL"; }
echo "[rft] PHASE 2: rollouts (vjepa_best gpu0 + bvae_match gpu1)"
mk_roll "$LAT/vjepa_best_fine_s0" vjepa_best_matched_s0 0 &
mk_roll "$LAT/bvae_match_d64_s0"  bvae_match_matched_s0 1 & wait

# --- Phase 3: T9 SL decoders for the two new latent spaces (LONG, 2-packed) ----
mk_dec(){ local latd=$1 tag=$2 g=$3
  [ -f "$DEC/$tag/decoder_iter030000.pt" ] && { echo "[rft] skip dec $tag"; return 0; }
  echo "[rft] decoder $tag gpu$g $(date -Iseconds)"
  taskset -c 0-31 bash scripts/session29/dec_posthoc_launch.sh "$latd" "$tag" "$g"; }
echo "[rft] PHASE 3: decoders (vjepa_best gpu0 + bvae_match gpu1)"
mk_dec "$LAT/vjepa_best_fine_s0" dec_vjepa_best_s0 0 &
mk_dec "$LAT/bvae_match_d64_s0"  dec_bvae_match_s0 1 & wait

# --- Phase 4: ROM field-forecast evals (cheap, both splits) -------------------
# rom <decoder> <rollout-tag> <latents-tag> <gpu>  -> tb json + tc json
rom(){ local dec=$1 roll=$2 lat=$3 g=$4
  for sp in test_b test_c; do
    local out="$ROM/$roll.json"; [ "$sp" = test_c ] && out="$ROM/$roll.$sp.json"
    [ -f "$out" ] && { echo "[rft] skip rom $roll $sp"; continue; }
    [ -f "$ROLL/$roll/$sp.npz" ] || { echo "[rft] no $sp rollout for $roll, skip"; continue; }
    echo "[rft] ROM $roll $sp"
    taskset -c 0-31 python scripts/session29/rom_field_forecast.py \
      --decoder "$dec" --rollout "$roll" --latents "$lat" --split "$sp" --gpu "$g" \
      >"$ROM/$roll.$sp.log" 2>&1 || echo "[rft] ROM $roll $sp FAIL"
  done; }
echo "[rft] PHASE 4: ROM evals (serialized on gpu0)"
rom "$DEC/dec_jepa_d64_s0/decoder_iter030000.pt"        jepa_matched_d64_s0   jepa_tf_noc_d64_s0   0
rom "$DEC/dec_posthoc_regae_d64/decoder_iter030000.pt"  regae_d64             regae/cnn_vit_s0     0
rom "$DEC/dec_bvae_match_s0/decoder_iter030000.pt"      bvae_match_matched_s0 bvae_match_d64_s0    0
rom "$DEC/dec_vjepa_best_s0/decoder_iter030000.pt"      vjepa_best_matched_s0 vjepa_best_fine_s0   0

echo "[rft] ===== FAIR ROM TABLE COMPLETE $(date -Iseconds) ====="
echo "[rft] summary table:"
taskset -c 0-31 python scripts/session29/rom_table_print.py 2>&1 || echo "[rft] (printer not yet present)"
