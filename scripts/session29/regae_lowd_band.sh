#!/usr/bin/env bash
# regAE (recon + SIGReg anti-collapse + matched lift/wake heads, CNN+ViT) forecast
# band at d in {16,32}, seeds {0,1,2}: train encoder -> extract latents -> matched
# predictor -> rollout; plus JEPA-own rollouts at the same d/seeds. Both GPUs,
# 2-packed, idempotent, CPU-capped (taskset -c 0-15 by caller -> >=64 cores free
# for asolera even co-running with the SSIM band). RTX 6000 only.
set -uo pipefail
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}" WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE
SPLIT="configs/splits/split_v2p1.json"; MAN="outputs/data_pipeline/v2p1/manifest.json"
LAT=outputs/session28/latents; PRED=outputs/session29/lowd_predictors; ROLL=outputs/session29/lowd_rollouts
RG=outputs/runs/session29_8/regae
dev(){ [ "$1" -eq 0 ] && echo cuda:2 || echo cuda:3; }

regae_enc(){ local d=$1 s=$2 g=$3; local out=$RG/regae_cnn_vit_d${d}_s${s}
  [ -f "$out/checkpoint_iter020000.pt" ] && { echo "[rb] skip enc d$d s$s"; return 0; }
  mkdir -p "$out"; echo "[rb] enc d$d s$s gpu$g"
  nice -n 10 python -u scripts/session9_train_fukami.py --all-train --max-iters 20000 --B 16 --T 32 --gpu "$g" \
    --partition v2p1 --split "$SPLIT" --omega-pipeline-manifest "$MAN" --recon-loss-type mse \
    --observable-head cl_future --observable-head-deltas 0 --lambda-lift 0.01 \
    --wake-observable-type patch_signed_spectrum --lambda-wake 1.0 --wake-loss smooth_l1 --wake-loss-beta 0.5 \
    --wake-head-hidden 128 --encoder cnn_vit --d "$d" --lambda-sigreg 0.01 --seed "$s" \
    --num-workers 3 --wandb-mode offline --log-every 200 --diagnostic-every 2000 --checkpoint-every 10000 \
    --tag-suffix s29_regae_d${d}_s${s} --output-dir "$out" >"$out/train.log" 2>&1
}
regae_lat(){ local d=$1 s=$2 g=$3; local o=$LAT/regae/cnn_vit_d${d}_s${s}
  [ -f "$o/test_b.npz" ] && return 0; echo "[rb] extract d$d s$s"
  taskset -c 0-15 python scripts/session18/encode_baseline_latents.py --baseline fukami --d "$d" \
    --checkpoint "$RG/regae_cnn_vit_d${d}_s${s}/checkpoint_iter020000.pt" --partition v2p1 --split "$SPLIT" \
    --pipeline-manifest "$MAN" --splits train test_b test_c --gpu "$g" --output-dir "$o" >"$o.enc.log" 2>&1 || true
}
regae_pr(){ local d=$1 s=$2 g=$3; local lat=$LAT/regae/cnn_vit_d${d}_s${s} p=$PRED/regae_d${d}_s${s} r=$ROLL/regae_d${d}_s${s}
  if [ ! -f "$p/checkpoint_iter020000.pt" ]; then mkdir -p "$p"; echo "[rb] pred d$d s$s"
    nice -n 10 python scripts/session18/train_baseline_predictor.py --latents-dir "$lat" --tag s29_regae_d${d}_s${s} \
      --gpu "$g" --seed 42 --cond-dim 0 --no-output-bn --output-dir "$p" >"$p/train.log" 2>&1; fi
  if [ ! -f "$r/test_b.npz" ]; then mkdir -p "$r"; echo "[rb] roll d$d s$s"
    nice -n 10 python scripts/session18/eval_baseline_rollouts.py --latents-dir "$lat" \
      --predictor "$p/checkpoint_iter020000.pt" --tag s29_regae_d${d}_s${s} --gpu "$g" --output-dir "$r" >"$r/eval.log" 2>&1; fi
}
jepa_own(){ local d=$1 s=$2 g=$3; local r=$ROLL/jepa_own_d${d}_s${s}
  [ -f "$r/test_b.npz" ] && return 0
  local ck=outputs/runs/session28/jepa_tf_noc_d${d}_s${s}/encoder/checkpoint_iter020000.pt
  [ -f "$ck" ] || { echo "[rb] no jepa enc d$d s$s"; return 0; }
  echo "[rb] jepa-own d$d s$s"
  nice -n 10 python scripts/session29/roll_own_predictor.py "$ck" "$LAT/jepa_tf_noc_d${d}_s${s}/test_b.npz" "$r" --device "$(dev $g)" >"$r.own.log" 2>&1 || true
}

echo "[rb] PHASE A regAE encoders (d16,d32 x s0,s1,s2), 2-packed at $(date -Iseconds)"
regae_enc 16 0 0 & regae_enc 16 1 1 & wait
regae_enc 16 2 0 & regae_enc 32 0 1 & wait
regae_enc 32 1 0 & regae_enc 32 2 1 & wait
echo "[rb] PHASE B extract latents at $(date -Iseconds)"
for d in 16 32; do for s in 0 1 2; do regae_lat $d $s 0; done; done
echo "[rb] PHASE C matched predictors + rolls, 2-packed at $(date -Iseconds)"
regae_pr 16 0 0 & regae_pr 16 1 1 & wait
regae_pr 16 2 0 & regae_pr 32 0 1 & wait
regae_pr 32 1 0 & regae_pr 32 2 1 & wait
echo "[rb] PHASE D JEPA-own rolls (d16,d32 x s0,s1,s42)"
for d in 16 32; do for s in 0 1 42; do jepa_own $d $s 1; done; done
echo "[rb] COMPLETE at $(date -Iseconds)"
