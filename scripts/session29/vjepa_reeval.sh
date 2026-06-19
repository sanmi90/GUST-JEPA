#!/usr/bin/env bash
# Fair V-JEPA forecast re-eval. Fixes TWO confounds in the first comparison:
#  (1) coarse d=64 eval pooling -> finer-temporal latent (overlapping clips
#      stride 8 + linear interp): --vjepa-eval-stride 8 --vjepa-eval-interp linear
#      -> latents outputs/session28/latents/vjepa_fine_s{0,1,2}.
#  (2) matched-vs-own predictor -> also build a JEPA-MATCHED forecast (matched
#      predictor on the per-frame jepa_tf_noc_d64 latents) so the forecast is
#      SAME-PREDICTOR across vjepa_fine / jepa_matched / regAE-matched (JEPA-own
#      kept only as the upper reference).
# RTX 6000 only (--gpu -> require_rtx6000); CPU-capped by caller (taskset 0-31);
# idempotent; 2-packed. Recompute the forecast band after with m_lowd_forecast.
set -uo pipefail
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}" WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SPLIT="configs/splits/split_v2p1.json"; MAN="outputs/data_pipeline/v2p1/manifest.json"
RUN=outputs/runs/session29/vjepa; LAT=outputs/session28/latents
PRED=outputs/session29/lowd_predictors; ROLL=outputs/session29/lowd_rollouts
LOCK=outputs/runs/session29/vjepa_reeval.lock
[ -f "$LOCK" ] && { echo "[re] lock exists; exit"; exit 0; }
mkdir -p outputs/runs/session29; echo "$$ $(date -Iseconds)" > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

# (1) finer V-JEPA latents
vj_fine_lat(){ local s=$1 g=$2; local o=$LAT/vjepa_fine_s${s}
  [ -f "$o/test_b.npz" ] && { echo "[re] skip fine-lat s$s"; return 0; }
  mkdir -p "$o"; echo "[re] fine extract s$s (stride8 linear)"
  taskset -c 0-31 python scripts/session18/encode_baseline_latents.py --baseline vjepa --d 64 \
    --vjepa-eval-stride 8 --vjepa-eval-interp linear \
    --checkpoint "$RUN/vjepa_s${s}/checkpoint_iter020000.pt" --partition v2p1 --split "$SPLIT" \
    --pipeline-manifest "$MAN" --splits train test_b --gpu "$g" --output-dir "$o" >"$o/encode.log" 2>&1 \
    || echo "[re] fine extract FAIL s$s"; }

# matched predictor + rollout for an arbitrary latent dir -> roll tag
matched(){ local latd=$1 tag=$2 g=$3; local p=$PRED/$tag r=$ROLL/$tag
  if [ ! -f "$p/checkpoint_iter020000.pt" ]; then mkdir -p "$p"; echo "[re] pred $tag"
    nice -n 10 python scripts/session18/train_baseline_predictor.py --latents-dir "$latd" \
      --tag s29_$tag --gpu "$g" --seed 42 --cond-dim 0 --no-output-bn --output-dir "$p" >"$p/train.log" 2>&1; fi
  if [ ! -f "$r/test_b.npz" ]; then mkdir -p "$r"; echo "[re] roll $tag"
    nice -n 10 python scripts/session18/eval_baseline_rollouts.py --latents-dir "$latd" \
      --predictor "$p/checkpoint_iter020000.pt" --tag s29_$tag --gpu "$g" --output-dir "$r" >"$r/eval.log" 2>&1; fi; }

echo "[re] ===== V-JEPA fair re-eval START $(date -Iseconds) ====="
echo "[re] PHASE 1: finer V-JEPA latents (2-packed)"
vj_fine_lat 0 0 & vj_fine_lat 1 1 & wait
vj_fine_lat 2 0 & wait
echo "[re] PHASE 2: matched predictors + rollouts (vjepa_fine + jepa_matched), 2-packed"
matched "$LAT/vjepa_fine_s0" vjepa_fine_matched_s0 0 & matched "$LAT/vjepa_fine_s1" vjepa_fine_matched_s1 1 & wait
matched "$LAT/vjepa_fine_s2" vjepa_fine_matched_s2 0 & matched "$LAT/jepa_tf_noc_d64_s0" jepa_matched_d64_s0 1 & wait
matched "$LAT/jepa_tf_noc_d64_s1" jepa_matched_d64_s1 0 & matched "$LAT/jepa_tf_noc_d64_s2" jepa_matched_d64_s2 1 & wait
echo "[re] ===== re-eval COMPLETE $(date -Iseconds) ====="
