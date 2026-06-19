#!/usr/bin/env bash
# "Best JEPA" = FULL V-JEPA 2.1 (dense context loss lam_ctx 0.5 + deep
# self-supervision n_levels 4) + the lift(C_L)+wake observable heads (same as the
# per-frame JEPA). Goal: beat regAE on forecast (wake+C_L) / SSIM / readability.
# Per seed {0,1,2}: train -> fine extract -> matched predictor + rollout.
# Self-gates on the vjepa_heads band finishing (free the cards). RTX 6000 only;
# CPU-capped by caller (taskset 0-31); idempotent.
set -uo pipefail
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}" WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SPLIT="configs/splits/split_v2p1.json"; MAN="outputs/data_pipeline/v2p1/manifest.json"
RUN=outputs/runs/session29/vjepa; LAT=outputs/session28/latents
PRED=outputs/session29/lowd_predictors; ROLL=outputs/session29/lowd_rollouts
LOCK=outputs/runs/session29/vjepa_best.lock
[ -f "$LOCK" ] && { echo "[vb] lock exists; exit"; exit 0; }
mkdir -p "$RUN" outputs/runs/session29; echo "$$ $(date -Iseconds)" > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

echo "[vb] waiting for vjepa_heads band to finish (free cards) ..."
deadline=$((SECONDS+18000))
while [ -f outputs/runs/session29/vjepa_heads.lock ]; do
  sleep 60; [ "$SECONDS" -ge "$deadline" ] && { echo "[vb] WARN wait timed out; proceeding"; break; }
done
echo "[vb] cards free at $(date -Iseconds)"

vb_train(){ local s=$1 g=$2; local out=$RUN/vjepa_best_s${s}
  [ -f "$out/checkpoint_iter020000.pt" ] && { echo "[vb] skip train s$s"; return 0; }
  mkdir -p "$out"; echo "[vb] train BEST s$s gpu$g $(date -Iseconds)"
  nice -n 10 python -u -m src.training.train_vjepa --all-train --max-iters 20000 \
    --B 16 --T 32 --hidden 384 --depth 8 --pred-depth 6 --mask-ratio 0.8 \
    --lam-ctx 0.5 --lam-ctx-warmup-frac 0.25 --n-levels 4 \
    --observable-head cl_future --observable-head-weight 0.01 --observable-head-deltas 0 \
    --wake-observable-type patch_signed_spectrum --lambda-wake 1.0 --wake-dim 80 \
    --partition v2p1 --split "$SPLIT" --omega-pipeline-manifest "$MAN" \
    --num-workers 3 --wandb-mode offline --log-every 200 --diagnostic-every 2000 \
    --checkpoint-every 10000 --seed "$s" --gpu "$g" \
    --tag-suffix s29_vjepa_best_s${s} --output-dir "$out" >"$out/train.log" 2>&1; }
vb_lat(){ local s=$1 g=$2; local o=$LAT/vjepa_best_fine_s${s}
  [ -f "$o/test_b.npz" ] && return 0; mkdir -p "$o"; echo "[vb] fine extract s$s"
  taskset -c 0-31 python scripts/session18/encode_baseline_latents.py --baseline vjepa --d 64 \
    --vjepa-eval-stride 8 --vjepa-eval-interp linear \
    --checkpoint "$RUN/vjepa_best_s${s}/checkpoint_iter020000.pt" --partition v2p1 --split "$SPLIT" \
    --pipeline-manifest "$MAN" --splits train test_a test_b test_c --gpu "$g" --output-dir "$o" >"$o/encode.log" 2>&1 \
    || echo "[vb] extract FAIL s$s"; }
vb_matched(){ local s=$1 g=$2; local latd=$LAT/vjepa_best_fine_s${s} tag=vjepa_best_matched_s${s}
  local p=$PRED/$tag r=$ROLL/$tag
  if [ ! -f "$p/checkpoint_iter020000.pt" ]; then mkdir -p "$p"; echo "[vb] pred s$s"
    nice -n 10 python scripts/session18/train_baseline_predictor.py --latents-dir "$latd" \
      --tag s29_$tag --gpu "$g" --seed 42 --cond-dim 0 --no-output-bn --output-dir "$p" >"$p/train.log" 2>&1; fi
  if [ ! -f "$r/test_b.npz" ]; then mkdir -p "$r"; echo "[vb] roll s$s"
    nice -n 10 python scripts/session18/eval_baseline_rollouts.py --latents-dir "$latd" \
      --predictor "$p/checkpoint_iter020000.pt" --tag s29_$tag --gpu "$g" --output-dir "$r" >"$r/eval.log" 2>&1; fi; }
vb_dec(){ local s=$1 g=$2; local tag=dec_vjepa_best_s${s}
  [ -f "outputs/runs/session29/$tag/decoder_iter030000.pt" ] && return 0
  echo "[vb] decoder s$s"; taskset -c 0-31 bash scripts/session29/dec_posthoc_launch.sh "$LAT/vjepa_best_fine_s${s}" "$tag" "$g"; }

echo "[vb] ===== BEST (full 2.1 + heads) band START $(date -Iseconds) ====="
echo "[vb] PHASE A: train (2-packed)"
vb_train 0 0 & vb_train 1 1 & wait
vb_train 2 0 & wait
echo "[vb] PHASE B: fine extract (all splits, for forecast + SSIM)"
for s in 0 1 2; do vb_lat $s 0; done
echo "[vb] PHASE C: matched predictor + rollout (2-packed)"
vb_matched 0 0 & vb_matched 1 1 & wait
vb_matched 2 0 & wait
echo "[vb] PHASE D: T9 decoders for SSIM (2-packed)"
vb_dec 0 0 & vb_dec 1 1 & wait
vb_dec 2 0 & wait
echo "[vb] ===== BEST band COMPLETE $(date -Iseconds) ====="
