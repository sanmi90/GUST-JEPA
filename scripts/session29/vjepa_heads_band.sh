#!/usr/bin/env bash
# Step (1): V-JEPA + lift(C_L)+wake heads = the PROPER both-supervised fair test.
# Plain masked V-JEPA objective (lam_ctx=0) + the SAME observable heads as the
# per-frame JEPA (lift cl_future delta0 weight 0.01, wake patch_signed_spectrum
# lambda 1.0). Per seed {0,1,2}: train -> extract FINE latent -> matched predictor
# + rollout. Then compare forecast (wake + C_L) to jepa-own/jepa_matched and to
# the no-head V-JEPA (vjepa_fine), isolating OBJECTIVE at matched supervision.
# Self-gates on the dense SSIM decoders finishing (vjepa_dense_decode.lock) to
# avoid GPU contention. RTX 6000 only; CPU-capped by caller (taskset 0-31).
set -uo pipefail
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}" WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SPLIT="configs/splits/split_v2p1.json"; MAN="outputs/data_pipeline/v2p1/manifest.json"
RUN=outputs/runs/session29/vjepa; LAT=outputs/session28/latents
PRED=outputs/session29/lowd_predictors; ROLL=outputs/session29/lowd_rollouts
LOCK=outputs/runs/session29/vjepa_heads.lock
[ -f "$LOCK" ] && { echo "[vh] lock exists; exit"; exit 0; }
mkdir -p "$RUN" outputs/runs/session29; echo "$$ $(date -Iseconds)" > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

echo "[vh] waiting for dense SSIM decoders to finish (free the cards) ..."
deadline=$((SECONDS+18000))
while [ -f outputs/runs/session29/vjepa_dense_decode.lock ]; do
  sleep 60; [ "$SECONDS" -ge "$deadline" ] && { echo "[vh] WARN wait timed out; proceeding"; break; }
done
echo "[vh] cards free at $(date -Iseconds)"

vh_train(){ local s=$1 g=$2; local out=$RUN/vjepa_heads_s${s}
  [ -f "$out/checkpoint_iter020000.pt" ] && { echo "[vh] skip train s$s"; return 0; }
  mkdir -p "$out"; echo "[vh] train heads s$s gpu$g $(date -Iseconds)"
  nice -n 10 python -u -m src.training.train_vjepa --all-train --max-iters 20000 \
    --B 16 --T 32 --hidden 384 --depth 8 --pred-depth 6 --mask-ratio 0.8 --lam-ctx 0.0 \
    --observable-head cl_future --observable-head-weight 0.01 --observable-head-deltas 0 \
    --wake-observable-type patch_signed_spectrum --lambda-wake 1.0 --wake-dim 80 \
    --partition v2p1 --split "$SPLIT" --omega-pipeline-manifest "$MAN" \
    --num-workers 3 --wandb-mode offline --log-every 200 --diagnostic-every 2000 \
    --checkpoint-every 10000 --seed "$s" --gpu "$g" \
    --tag-suffix s29_vjepa_heads_s${s} --output-dir "$out" >"$out/train.log" 2>&1; }
vh_lat(){ local s=$1 g=$2; local o=$LAT/vjepa_heads_fine_s${s}
  [ -f "$o/test_b.npz" ] && return 0; mkdir -p "$o"; echo "[vh] fine extract s$s"
  taskset -c 0-31 python scripts/session18/encode_baseline_latents.py --baseline vjepa --d 64 \
    --vjepa-eval-stride 8 --vjepa-eval-interp linear \
    --checkpoint "$RUN/vjepa_heads_s${s}/checkpoint_iter020000.pt" --partition v2p1 --split "$SPLIT" \
    --pipeline-manifest "$MAN" --splits train test_b --gpu "$g" --output-dir "$o" >"$o/encode.log" 2>&1 \
    || echo "[vh] extract FAIL s$s"; }
vh_matched(){ local s=$1 g=$2; local latd=$LAT/vjepa_heads_fine_s${s} tag=vjepa_heads_matched_s${s}
  local p=$PRED/$tag r=$ROLL/$tag
  if [ ! -f "$p/checkpoint_iter020000.pt" ]; then mkdir -p "$p"; echo "[vh] pred s$s"
    nice -n 10 python scripts/session18/train_baseline_predictor.py --latents-dir "$latd" \
      --tag s29_$tag --gpu "$g" --seed 42 --cond-dim 0 --no-output-bn --output-dir "$p" >"$p/train.log" 2>&1; fi
  if [ ! -f "$r/test_b.npz" ]; then mkdir -p "$r"; echo "[vh] roll s$s"
    nice -n 10 python scripts/session18/eval_baseline_rollouts.py --latents-dir "$latd" \
      --predictor "$p/checkpoint_iter020000.pt" --tag s29_$tag --gpu "$g" --output-dir "$r" >"$r/eval.log" 2>&1; fi; }

echo "[vh] ===== V-JEPA+heads band START $(date -Iseconds) ====="
echo "[vh] PHASE A: train (2-packed)"
vh_train 0 0 & vh_train 1 1 & wait
vh_train 2 0 & wait
echo "[vh] PHASE B: fine extract"
for s in 0 1 2; do vh_lat $s 0; done
echo "[vh] PHASE C: matched predictor + rollout (2-packed)"
vh_matched 0 0 & vh_matched 1 1 & wait
vh_matched 2 0 & wait
echo "[vh] ===== V-JEPA+heads band COMPLETE $(date -Iseconds) ====="
