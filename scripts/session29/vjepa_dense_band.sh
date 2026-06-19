#!/usr/bin/env bash
# V-JEPA dense-loss variant band (V-JEPA 2.1 context-token loss, --lam-ctx 0.5
# + warmup). Tests whether forcing dense local grounding fixes V-JEPA's poor
# short-horizon wake forecast. Per seed {0,1,2}: train dense encoder -> extract
# the FINE eval latent (overlapping clips stride8 + linear interp, the fair
# forecast eval) -> matched predictor + rollout. Then compare the forecast to
# plain vjepa_fine (-0.08->+0.51) and jepa_matched (+0.88->+0.53).
# RTX 6000 only; CPU-capped by caller (taskset 0-31, >=64 cores free); idempotent.
set -uo pipefail
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}" WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SPLIT="configs/splits/split_v2p1.json"; MAN="outputs/data_pipeline/v2p1/manifest.json"
RUN=outputs/runs/session29/vjepa; LAT=outputs/session28/latents
PRED=outputs/session29/lowd_predictors; ROLL=outputs/session29/lowd_rollouts
LOCK=outputs/runs/session29/vjepa_dense.lock
[ -f "$LOCK" ] && { echo "[vd] lock exists; exit"; exit 0; }
mkdir -p "$RUN" outputs/runs/session29; echo "$$ $(date -Iseconds)" > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

vd_train(){ local s=$1 g=$2; local out=$RUN/vjepa_dense_s${s}
  [ -f "$out/checkpoint_iter020000.pt" ] && { echo "[vd] skip train s$s"; return 0; }
  mkdir -p "$out"; echo "[vd] train dense s$s gpu$g $(date -Iseconds)"
  nice -n 10 python -u -m src.training.train_vjepa --all-train --max-iters 20000 \
    --B 16 --T 32 --hidden 384 --depth 8 --pred-depth 6 --mask-ratio 0.8 \
    --lam-ctx 0.5 --lam-ctx-warmup-frac 0.25 \
    --partition v2p1 --split "$SPLIT" --omega-pipeline-manifest "$MAN" \
    --num-workers 3 --wandb-mode offline --log-every 200 --diagnostic-every 2000 \
    --checkpoint-every 10000 --seed "$s" --gpu "$g" \
    --tag-suffix s29_vjepa_dense_s${s} --output-dir "$out" >"$out/train.log" 2>&1; }
vd_lat(){ local s=$1 g=$2; local o=$LAT/vjepa_dense_fine_s${s}
  [ -f "$o/test_b.npz" ] && return 0; mkdir -p "$o"; echo "[vd] fine extract s$s"
  taskset -c 0-31 python scripts/session18/encode_baseline_latents.py --baseline vjepa --d 64 \
    --vjepa-eval-stride 8 --vjepa-eval-interp linear \
    --checkpoint "$RUN/vjepa_dense_s${s}/checkpoint_iter020000.pt" --partition v2p1 --split "$SPLIT" \
    --pipeline-manifest "$MAN" --splits train test_b --gpu "$g" --output-dir "$o" >"$o/encode.log" 2>&1 \
    || echo "[vd] extract FAIL s$s"; }
vd_matched(){ local s=$1 g=$2; local latd=$LAT/vjepa_dense_fine_s${s} tag=vjepa_dense_matched_s${s}
  local p=$PRED/$tag r=$ROLL/$tag
  if [ ! -f "$p/checkpoint_iter020000.pt" ]; then mkdir -p "$p"; echo "[vd] pred s$s"
    nice -n 10 python scripts/session18/train_baseline_predictor.py --latents-dir "$latd" \
      --tag s29_$tag --gpu "$g" --seed 42 --cond-dim 0 --no-output-bn --output-dir "$p" >"$p/train.log" 2>&1; fi
  if [ ! -f "$r/test_b.npz" ]; then mkdir -p "$r"; echo "[vd] roll s$s"
    nice -n 10 python scripts/session18/eval_baseline_rollouts.py --latents-dir "$latd" \
      --predictor "$p/checkpoint_iter020000.pt" --tag s29_$tag --gpu "$g" --output-dir "$r" >"$r/eval.log" 2>&1; fi; }

echo "[vd] ===== V-JEPA dense band START $(date -Iseconds) ====="
echo "[vd] PHASE A: train dense encoders (2-packed)"
vd_train 0 0 & vd_train 1 1 & wait
vd_train 2 0 & wait
echo "[vd] PHASE B: fine extract"
for s in 0 1 2; do vd_lat $s 0; done
echo "[vd] PHASE C: matched predictor + rollout (2-packed)"
vd_matched 0 0 & vd_matched 1 1 & wait
vd_matched 2 0 & wait
echo "[vd] ===== dense band COMPLETE $(date -Iseconds) ====="
