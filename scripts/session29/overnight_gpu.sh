#!/usr/bin/env bash
# Overnight autonomous GPU pipeline (user asleep 2026-06-18). Self-gates on the
# d32 band finishing, then runs, in order:
#   PHASE 1  deferred d64 SSIM decoders (dec_jepa_d64_s2, dec_regae_d64_s1/s2)
#   PHASE 2  ST smoke gate (200-iter st_hybrid -> extract -> roll; abort band on fail)
#   PHASE 3  full ST band: train d{64,16} seeds, extract latents, JEPA-own rolls,
#            T9 decoders.
# Verbatim jepa_common/wake_on from scripts/session28/_run_one.sh so the ST runs
# are the proven jepa_tf_noc recipe + (--encoder st_hybrid --temporal-kernel 3).
# RTX 6000 only (--gpu 0/1 -> require_rtx6000; roll uses raw cuda:2/cuda:3 which
# are the RTX cards in torch's device order, per d16_then_d32.sh). CPU-capped by
# the launcher (taskset -c 0-31 -> >=64 cores free for asolera). Idempotent +
# lockfile-guarded. Launch: taskset -c 0-31 bash scripts/session29/overnight_gpu.sh > LOG 2>&1 &
set -uo pipefail
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
REPO=$(pwd)
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}" WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SPLIT="configs/splits/split_v2p1.json"; MAN="outputs/data_pipeline/v2p1/manifest.json"
RUN=outputs/runs/session29/st; LAT=outputs/session28/latents; ROLL=outputs/session29/lowd_rollouts
DEC=outputs/runs/session29
LOCK=outputs/runs/session29/overnight.lock

[ -f "$LOCK" ] && { echo "[ov] lock $LOCK exists; another run owns this. exit."; exit 0; }
mkdir -p outputs/runs/session29; echo "$$ $(date -Iseconds)" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

jepa_common=(--all-train --max-iters 20000 --B 16 --T 32 --H-roll 8
    --lambda-sigreg 0.01 --lr-encoder 1.5e-4 --lr-predictor 5e-4 --weight-decay 0.05
    --warmup-frac 0.05 --grad-clip 1.0 --num-workers 3
    --projection-norm batchnorm --anticollapse sigreg
    --observable-head cl_future --observable-head-weight 0.01 --observable-head-deltas 0
    --partition v2p1 --split "$SPLIT" --omega-pipeline-manifest "$MAN"
    --wandb-mode offline --log-every 200 --diagnostic-every 2000 --checkpoint-every 10000)
wake_on=(--wake-observable-type patch_signed_spectrum --lambda-wake 1.00
    --wake-loss smooth_l1 --wake-loss-beta 0.5 --wake-head-hidden 128)
dev(){ [ "$1" -eq 0 ] && echo cuda:2 || echo cuda:3; }  # RTX 6000s in torch order

dec_existing(){ local latdir=$1 tag=$2 g=$3
    [ -f "$DEC/$tag/decoder_iter030000.pt" ] && { echo "[ov] skip dec $tag"; return 0; }
    [ -f "$latdir/test_b.npz" ] || { echo "[ov] MISSING latents $latdir; skip $tag"; return 0; }
    taskset -c 0-31 bash scripts/session29/dec_posthoc_launch.sh "$latdir" "$tag" "$g"; }

run_st(){ local d=$1 s=$2 g=$3; local out=$RUN/st_d${d}_s${s}/encoder
    [ -f "$out/checkpoint_iter020000.pt" ] && { echo "[ov] skip enc st_d${d}_s${s}"; return 0; }
    mkdir -p "$out"; echo "[ov] enc st_d${d}_s${s} gpu$g $(date -Iseconds)"
    nice -n 10 python -u -m src.training.train_jepa "${jepa_common[@]}" "${wake_on[@]}" \
        --encoder st_hybrid --temporal-kernel 3 --predictor-cond-dim 0 --d "$d" --seed "$s" --gpu "$g" \
        --tag-suffix s29_st_d${d}_s${s} --output-dir "$out" >"$out/train.log" 2>&1; }
st_lat(){ local d=$1 s=$2 g=$3; local o=$LAT/st_d${d}_s${s}
    [ -f "$o/test_b.npz" ] && return 0; mkdir -p "$o"; echo "[ov] extract st_d${d}_s${s}"
    taskset -c 0-31 python scripts/session18/encode_baseline_latents.py --baseline jepa --d "$d" \
        --checkpoint "$RUN/st_d${d}_s${s}/encoder/checkpoint_iter020000.pt" --partition v2p1 \
        --split "$SPLIT" --pipeline-manifest "$MAN" --splits train test_a test_b test_c \
        --gpu "$g" --output-dir "$o" >"$o/encode.log" 2>&1 || true; }
st_roll(){ local d=$1 s=$2 g=$3; local r=$ROLL/st_own_d${d}_s${s}
    [ -f "$r/test_b.npz" ] && return 0
    local ck=$RUN/st_d${d}_s${s}/encoder/checkpoint_iter020000.pt
    [ -f "$ck" ] || { echo "[ov] no enc st_d${d}_s${s}"; return 0; }; mkdir -p "$r"; echo "[ov] roll st_d${d}_s${s}"
    nice -n 10 python scripts/session29/roll_own_predictor.py "$ck" "$LAT/st_d${d}_s${s}/test_b.npz" \
        "$r" --device "$(dev $g)" >"$r.own.log" 2>&1 || true; }
st_dec(){ local d=$1 s=$2 g=$3; dec_existing "$LAT/st_d${d}_s${s}" "dec_st_d${d}_s${s}" "$g"; }

wait_d32(){ local deadline=$((SECONDS+9000))  # <=2.5h
    echo "[ov] waiting for d32 chain to free the cards..."
    while :; do
        if [ -f "$ROLL/jepa_own_d32_s42/test_b.npz" ] && ! pgrep -f 'd16_then_d32' >/dev/null 2>&1; then
            echo "[ov] d32 chain complete at $(date -Iseconds)"; return 0; fi
        [ "$SECONDS" -ge "$deadline" ] && { echo "[ov] WARN d32 wait timed out; proceeding $(date -Iseconds)"; return 0; }
        sleep 60
    done; }

smoke(){
    local sm=outputs/runs/session29/st_smoke
    rm -rf "$sm" "$LAT/st_smoke" "$ROLL/st_smoke_own"; mkdir -p "$sm/encoder"
    echo "[ov] SMOKE train (200 iters) $(date -Iseconds)"
    nice -n 10 python -u -m src.training.train_jepa "${jepa_common[@]}" "${wake_on[@]}" \
        --encoder st_hybrid --temporal-kernel 3 --predictor-cond-dim 0 --d 16 --seed 0 --gpu 0 \
        --max-iters 200 --checkpoint-every 200 --diagnostic-every 200 \
        --tag-suffix s29_st_smoke --output-dir "$sm/encoder" >"$sm/encoder/train.log" 2>&1
    [ -f "$sm/encoder/checkpoint_iter000200.pt" ] || { echo "[ov] SMOKE FAIL: no checkpoint (see $sm/encoder/train.log)"; return 1; }
    taskset -c 0-31 python scripts/session18/encode_baseline_latents.py --baseline jepa --d 16 \
        --checkpoint "$sm/encoder/checkpoint_iter000200.pt" --partition v2p1 --split "$SPLIT" \
        --pipeline-manifest "$MAN" --splits test_b --gpu 0 --output-dir "$LAT/st_smoke" >"$sm/extract.log" 2>&1 \
        || { echo "[ov] SMOKE FAIL: extract"; return 1; }
    [ -f "$LAT/st_smoke/test_b.npz" ] || { echo "[ov] SMOKE FAIL: no latents"; return 1; }
    nice -n 10 python scripts/session29/roll_own_predictor.py "$sm/encoder/checkpoint_iter000200.pt" \
        "$LAT/st_smoke/test_b.npz" "$ROLL/st_smoke_own" --device cuda:2 >"$sm/roll.log" 2>&1 \
        || { echo "[ov] SMOKE FAIL: roll"; return 1; }
    [ -f "$ROLL/st_smoke_own/test_b.npz" ] || { echo "[ov] SMOKE FAIL: no rollout"; return 1; }
    echo "[ov] SMOKE PASS $(date -Iseconds)"; return 0; }

echo "[ov] ===== overnight pipeline START $(date -Iseconds) (pid $$) ====="
wait_d32

# Reordered 2026-06-19 00:20: ST band is the priority and the long pole, so it
# runs FIRST (smoke -> encoders -> latents -> rolls gives forecast/drift/probe/PR
# by morning). ST decoders then the deferred d64 SSIM remainder run LAST (SSIM is
# secondary corroboration; the regae remainder is blocked by a pre-existing
# missing-latent gap anyway and will skip/fail harmlessly).
echo "[ov] ===== PHASE A: ST smoke gate ====="
if ! smoke; then echo "[ov] ABORT: ST smoke failed; not launching the band. $(date -Iseconds)"; exit 1; fi

echo "[ov] ===== PHASE B: ST encoders (2-packed) ====="
run_st 64 0 0 & run_st 64 1 1 & wait
run_st 64 2 0 & run_st 64 42 1 & wait
run_st 16 0 0 & run_st 16 1 1 & wait
run_st 16 42 0 & wait

echo "[ov] ===== PHASE C: ST latents + JEPA-own rollouts ====="
for s in 0 1 2 42; do st_lat 64 "$s" 0; done
for s in 0 1 42;   do st_lat 16 "$s" 0; done
st_roll 64 0 0 & st_roll 64 1 1 & wait
st_roll 64 2 0 & st_roll 64 42 1 & wait
st_roll 16 0 0 & st_roll 16 1 1 & wait
st_roll 16 42 0 & wait
echo "[ov] forecast/drift/probe/PR computable now (latents+rolls done) $(date -Iseconds)"

echo "[ov] ===== PHASE D: ST T9 decoders (2-packed) ====="
st_dec 64 0 0 & st_dec 64 1 1 & wait
st_dec 64 2 0 & st_dec 64 42 1 & wait
st_dec 16 0 0 & st_dec 16 1 1 & wait
st_dec 16 42 0 & wait

echo "[ov] ===== PHASE E: deferred d64 SSIM remainder (lowest priority) ====="
dec_existing "$LAT/jepa_tf_noc_d64_s2" dec_jepa_d64_s2  0 &
dec_existing "$LAT/regae/cnn_vit_s1"   dec_regae_d64_s1 1 & wait
dec_existing "$LAT/regae/cnn_vit_s2"   dec_regae_d64_s2 0 & wait
echo "[ov] ===== overnight pipeline COMPLETE $(date -Iseconds) ====="
