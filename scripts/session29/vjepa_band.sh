#!/usr/bin/env bash
# V-JEPA band (variant B): train faithful V-JEPA at d=64 (pooled->PCA64), seeds
# {0,1,2}, then extract latents -> matched predictor + rollout -> T9 decoder, so
# the result lands in the same six-metric comparison as JEPA/ST/regAE. A 50-iter
# GPU smoke gate first validates train_vjepa + the vjepa extract path end-to-end
# (deferred from the unit-tested CPU build) before the full 3x20k run.
# RTX 6000 only (require_rtx6000 via --gpu); L40S forbidden. CPU-capped by the
# caller (taskset -c 0-31 -> >=64 cores free for asolera). Idempotent. 2-packed.
set -uo pipefail
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}" WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SPLIT="configs/splits/split_v2p1.json"; MAN="outputs/data_pipeline/v2p1/manifest.json"
RUN=outputs/runs/session29/vjepa; LAT=outputs/session28/latents
PRED=outputs/session29/lowd_predictors; ROLL=outputs/session29/lowd_rollouts; DEC=outputs/runs/session29
LOG=outputs/runs/session29/vjepa_band.log
LOCK=outputs/runs/session29/vjepa_band.lock
[ -f "$LOCK" ] && { echo "[vj] lock $LOCK exists; exit"; exit 0; }
mkdir -p "$RUN" outputs/runs/session29; echo "$$ $(date -Iseconds)" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

vj_train(){ local s=$1 g=$2; local out=$RUN/vjepa_s${s}
  [ -f "$out/checkpoint_iter020000.pt" ] && { echo "[vj] skip train s$s"; return 0; }
  mkdir -p "$out"; echo "[vj] train vjepa_s$s gpu$g $(date -Iseconds)"
  nice -n 10 python -u -m src.training.train_vjepa --all-train --max-iters 20000 \
    --B 16 --T 32 --hidden 384 --depth 8 --pred-depth 6 --mask-ratio 0.8 \
    --partition v2p1 --split "$SPLIT" --omega-pipeline-manifest "$MAN" \
    --num-workers 3 --wandb-mode offline --log-every 200 --diagnostic-every 2000 \
    --checkpoint-every 10000 --seed "$s" --gpu "$g" \
    --tag-suffix s29_vjepa_s${s} --output-dir "$out" >"$out/train.log" 2>&1; }
vj_lat(){ local s=$1 g=$2; local o=$LAT/vjepa_s${s}
  [ -f "$o/test_b.npz" ] && return 0; mkdir -p "$o"; echo "[vj] extract vjepa_s$s"
  taskset -c 0-31 python scripts/session18/encode_baseline_latents.py --baseline vjepa --d 64 \
    --checkpoint "$RUN/vjepa_s${s}/checkpoint_iter020000.pt" --partition v2p1 --split "$SPLIT" \
    --pipeline-manifest "$MAN" --splits train test_a test_b test_c --gpu "$g" \
    --output-dir "$o" >"$o/encode.log" 2>&1 || echo "[vj] extract FAIL s$s"; }
vj_pr(){ local s=$1 g=$2; local lat=$LAT/vjepa_s${s} p=$PRED/vjepa_s${s} r=$ROLL/vjepa_matched_s${s}
  if [ ! -f "$p/checkpoint_iter020000.pt" ]; then mkdir -p "$p"; echo "[vj] pred vjepa_s$s"
    nice -n 10 python scripts/session18/train_baseline_predictor.py --latents-dir "$lat" \
      --tag s29_vjepa_s${s} --gpu "$g" --seed 42 --cond-dim 0 --no-output-bn \
      --output-dir "$p" >"$p/train.log" 2>&1; fi
  if [ ! -f "$r/test_b.npz" ]; then mkdir -p "$r"; echo "[vj] roll vjepa_s$s"
    nice -n 10 python scripts/session18/eval_baseline_rollouts.py --latents-dir "$lat" \
      --predictor "$p/checkpoint_iter020000.pt" --tag s29_vjepa_s${s} --gpu "$g" \
      --output-dir "$r" >"$r/eval.log" 2>&1; fi; }
vj_dec(){ local s=$1 g=$2; local tag=dec_vjepa_s${s}
  [ -f "$DEC/$tag/decoder_iter030000.pt" ] && { echo "[vj] skip dec s$s"; return 0; }
  echo "[vj] decoder s$s"; taskset -c 0-31 bash scripts/session29/dec_posthoc_launch.sh "$LAT/vjepa_s${s}" "$tag" "$g"; }

smoke(){
  local sm=$RUN/vjepa_smoke; rm -rf "$sm" "$LAT/vjepa_smoke"; mkdir -p "$sm"
  echo "[vj] SMOKE train 50 iters $(date -Iseconds)"
  nice -n 10 python -u -m src.training.train_vjepa --all-train --max-iters 50 \
    --B 16 --T 32 --hidden 384 --depth 8 --pred-depth 6 --mask-ratio 0.8 \
    --partition v2p1 --split "$SPLIT" --omega-pipeline-manifest "$MAN" \
    --num-workers 3 --wandb-mode offline --log-every 10 --diagnostic-every 50 \
    --checkpoint-every 50 --seed 0 --gpu 0 --tag-suffix s29_vjepa_smoke \
    --output-dir "$sm" >"$sm/train.log" 2>&1
  [ -f "$sm/checkpoint_iter000050.pt" ] || { echo "[vj] SMOKE FAIL: no ckpt (see $sm/train.log)"; return 1; }
  grep -q 'token_std' "$sm/train.log" && grep 'PR=' "$sm/train.log" | tail -1
  taskset -c 0-31 python scripts/session18/encode_baseline_latents.py --baseline vjepa --d 64 \
    --checkpoint "$sm/checkpoint_iter000050.pt" --partition v2p1 --split "$SPLIT" \
    --pipeline-manifest "$MAN" --splits train test_b --gpu 0 --output-dir "$LAT/vjepa_smoke" \
    >"$sm/encode.log" 2>&1 || { echo "[vj] SMOKE FAIL: extract (see $sm/encode.log)"; return 1; }
  python -c "import numpy as np; z=np.load('$LAT/vjepa_smoke/test_b.npz')['z_full']; assert z.shape[1:]==(120,64), z.shape; print('[vj] SMOKE PASS z_full',z.shape)" || return 1; }

echo "[vj] ===== V-JEPA band START $(date -Iseconds) ====="
echo "[vj] ===== SMOKE gate ====="
if ! smoke; then echo "[vj] ABORT: smoke failed; not launching the band."; exit 1; fi
echo "[vj] ===== PHASE A: train (2-packed) ====="
vj_train 0 0 & vj_train 1 1 & wait
vj_train 2 0 & wait
echo "[vj] ===== PHASE B: extract latents ====="
for s in 0 1 2; do vj_lat $s 0; done
echo "[vj] ===== PHASE C: matched predictor + rollout (2-packed) ====="
vj_pr 0 0 & vj_pr 1 1 & wait
vj_pr 2 0 & wait
echo "[vj] ===== PHASE D: T9 decoders (2-packed) ====="
vj_dec 0 0 & vj_dec 1 1 & wait
vj_dec 2 0 & wait
echo "[vj] ===== V-JEPA band COMPLETE $(date -Iseconds) ====="
