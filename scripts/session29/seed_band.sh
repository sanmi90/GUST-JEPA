#!/usr/bin/env bash
# 3-seed confirmation band: regAE (recon+anti-collapse) vs JEPA (predictive) at
# d=64 on BOTH axes -- forecast (matched predictor / JEPA-own rollout) and field
# SSIM (T9 SL decoder) -- across seeds s0,s1,s2 (JEPA s42 + regAE s0 already done).
# Idempotent (skips existing checkpoints/rollouts), 2-card packed, CPU-capped by
# the caller (taskset -c 0-15). RTX 6000 only. Single tracked job; reads back via
# m_lowd_forecast.py (forecast) and the decoder_summary.json files (SSIM).
set -uo pipefail
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}" WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE
LAT=outputs/session28/latents; PRED=outputs/session29/lowd_predictors; ROLL=outputs/session29/lowd_rollouts
jl(){ echo "$LAT/jepa_tf_noc_d64_s$1"; }; rl(){ echo "$LAT/regae/cnn_vit_s$1"; }
dev(){ [ "$1" -eq 0 ] && echo cuda:2 || echo cuda:3; }

regae_fc(){ local s=$1 g=$2; local lat=$(rl "$s") p=$PRED/regae_d64_s$s r=$ROLL/regae_d64_s$s
  if [ ! -f "$p/checkpoint_iter020000.pt" ]; then mkdir -p "$p"; echo "[band] regae pred s$s"
    nice -n 10 python scripts/session18/train_baseline_predictor.py --latents-dir "$lat" \
      --tag s29_regae_d64_s$s --gpu "$g" --seed 42 --cond-dim 0 --no-output-bn --output-dir "$p" >"$p/train.log" 2>&1; fi
  if [ ! -f "$r/test_b.npz" ]; then mkdir -p "$r"; echo "[band] regae roll s$s"
    nice -n 10 python scripts/session18/eval_baseline_rollouts.py --latents-dir "$lat" \
      --predictor "$p/checkpoint_iter020000.pt" --tag s29_regae_d64_s$s --gpu "$g" --output-dir "$r" >"$r/eval.log" 2>&1; fi
}
jepa_own(){ local s=$1 g=$2; local r=$ROLL/jepa_own_d64_s$s
  [ -f "$r/test_b.npz" ] && return 0; echo "[band] jepa-own roll s$s"
  nice -n 10 python scripts/session29/roll_own_predictor.py \
    outputs/runs/session28/jepa_tf_noc_d64_s$s/encoder/checkpoint_iter020000.pt \
    "$(jl $s)/test_b.npz" "$r" --device "$(dev $g)" >"$r.own.log" 2>&1 || true
}
decode(){ local latdir=$1 tag=$2 g=$3
  [ -f "outputs/runs/session29/$tag/decoder_iter030000.pt" ] && { echo "[band] skip dec $tag"; return 0; }
  echo "[band] decoder $tag"
  taskset -c 0-15 bash scripts/session29/dec_posthoc_launch.sh "$latdir" "$tag" "$g"
}

echo "[band] PHASE 1 forecast (parallel) at $(date -Iseconds)"
( regae_fc 1 0; jepa_own 0 0; jepa_own 2 0 ) &
( regae_fc 2 1; jepa_own 1 1 ) &
wait
echo "[band] PHASE 2 decoders, 2-card packed at $(date -Iseconds)"
decode "$(jl 0)" dec_jepa_d64_s0 0 & decode "$(jl 1)" dec_jepa_d64_s1 1 & wait
decode "$(jl 2)" dec_jepa_d64_s2 0 & decode "$(rl 1)" dec_regae_d64_s1 1 & wait
decode "$(rl 2)" dec_regae_d64_s2 0 & wait
echo "[band] COMPLETE at $(date -Iseconds)"
