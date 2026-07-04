#!/bin/bash
# SESSION 34 Track C conditioning-ablation training queue.
#
# The 2x2x2 conditioning cube over {L, W, N} on the pooled d=32 vector-predictor
# flagship pipeline (D250), plus the spec-exact AE objective anchors. Reused
# cells (CLW = session33/jepa_pool_vec s0-2, CL s0 = session33/jepa_nowake_pool_vec,
# AE-L s0 = session32/ae_nowake_pool, AE-LW = session32+33 ae_wake_pool) are NOT
# retrained; symlinks are laid by the eval chain.
#
# 25 runs, 10k iters each (~37 min/GPU). Work-stealing clone of
# scripts/session33/run_vec_queue.sh: one worker per GPU, atomic mkdir locks.
# PREREQUISITE: the nearbody observable cache must exist and its QC gate must
# have PASSED (scripts/session34/precompute_nearbody_observables.py).
# Usage:
#   scripts/session34/run_trackc_queue.sh <gpu 0|1> <cores e.g. 0-7>
set -u
cd /home/carlos/GUST-JEPA
source .venv/bin/activate
export PREVENT_ROOT=$HOME/PREVENT WANDB_PROJECT=vortex-jepa OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

GPU=$1
CORES=$2
RUNS=outputs/runs/session34
PIPE=outputs/data_pipeline/v2p2/manifest.json
LOCKS=$RUNS/.locks
mkdir -p "$RUNS" "$LOCKS"

# Fail fast if the nearbody QC gate has not passed.
GATE=$(python -c "import json,os;m=json.load(open(os.path.expanduser('$HOME/PREVENT/data/processed/vortex-jepa/v2p2/nearbody_observables/_manifest.json')));print(m['qc']['gate_pass'])" 2>/dev/null || echo missing)
if [ "$GATE" != "True" ]; then
  echo "[trackc gpu$GPU] ABORT: nearbody QC gate not passed (gate=$GATE). Run the precompute first."
  exit 2
fi

VEC="--predictor-class transformer"
# name|config|extra-flags (canonical trainer for all; AE anchors have no predictor)
QUEUE=(
  "jepa_pool_c0_s0|configs/ablation/jepa_pool_c0.yaml|$VEC --seed 0"
  "jepa_pool_c0_s1|configs/ablation/jepa_pool_c0.yaml|$VEC --seed 1"
  "jepa_pool_c0_s2|configs/ablation/jepa_pool_c0.yaml|$VEC --seed 2"
  "jepa_nowake_pool_vec_s1|configs/ablation/jepa_nowake_pool.yaml|$VEC --seed 1"
  "jepa_nowake_pool_vec_s2|configs/ablation/jepa_nowake_pool.yaml|$VEC --seed 2"
  "jepa_pool_w_s0|configs/ablation/jepa_pool_w.yaml|$VEC --seed 0"
  "jepa_pool_w_s1|configs/ablation/jepa_pool_w.yaml|$VEC --seed 1"
  "jepa_pool_w_s2|configs/ablation/jepa_pool_w.yaml|$VEC --seed 2"
  "jepa_pool_n_s0|configs/ablation/jepa_pool_n.yaml|$VEC --seed 0"
  "jepa_pool_n_s1|configs/ablation/jepa_pool_n.yaml|$VEC --seed 1"
  "jepa_pool_n_s2|configs/ablation/jepa_pool_n.yaml|$VEC --seed 2"
  "jepa_pool_ln_s0|configs/ablation/jepa_pool_ln.yaml|$VEC --seed 0"
  "jepa_pool_ln_s1|configs/ablation/jepa_pool_ln.yaml|$VEC --seed 1"
  "jepa_pool_ln_s2|configs/ablation/jepa_pool_ln.yaml|$VEC --seed 2"
  "jepa_pool_wn_s0|configs/ablation/jepa_pool_wn.yaml|$VEC --seed 0"
  "jepa_pool_wn_s1|configs/ablation/jepa_pool_wn.yaml|$VEC --seed 1"
  "jepa_pool_wn_s2|configs/ablation/jepa_pool_wn.yaml|$VEC --seed 2"
  "jepa_pool_lwn_s0|configs/ablation/jepa_pool_lwn.yaml|$VEC --seed 0"
  "jepa_pool_lwn_s1|configs/ablation/jepa_pool_lwn.yaml|$VEC --seed 1"
  "jepa_pool_lwn_s2|configs/ablation/jepa_pool_lwn.yaml|$VEC --seed 2"
  "ae_w_pool_s0|configs/ablation/ae_w_pool.yaml|--seed 0"
  "ae_w_pool_s1|configs/ablation/ae_w_pool.yaml|--seed 1"
  "ae_w_pool_s2|configs/ablation/ae_w_pool.yaml|--seed 2"
  "ae_nowake_pool_s1|configs/ablation/ae_nowake_pool.yaml|--seed 1"
  "ae_nowake_pool_s2|configs/ablation/ae_nowake_pool.yaml|--seed 2"
)

for item in "${QUEUE[@]}"; do
  IFS='|' read -r name cfg extra <<< "$item"
  if [ -f "$RUNS/$name/checkpoint_iter010000.pt" ]; then
    echo "[trackc gpu$GPU] skip (done): $name"
    continue
  fi
  if ! mkdir "$LOCKS/$name" 2>/dev/null; then
    echo "[trackc gpu$GPU] skip (claimed): $name"
    continue
  fi
  mkdir -p "$RUNS/$name"
  echo "[trackc gpu$GPU] start: $name @ $(date -Iseconds)"
  taskset -c "$CORES" python -m src.training.train_canonical \
    --config "$cfg" \
    --partition v2p2 --pipeline-manifest "$PIPE" \
    --gpu "$GPU" --max-iters 10000 --num-workers 3 \
    --diagnostic-every 1000 --checkpoint-every 2500 --log-every 200 \
    --wandb-mode offline \
    $extra \
    --out "$RUNS/$name" > "$RUNS/$name/train.log" 2>&1 \
    || echo "[trackc gpu$GPU] FAILED: $name" >> "$RUNS/failures.log"
  echo "[trackc gpu$GPU] done: $name @ $(date -Iseconds)"
done

echo "[trackc gpu$GPU] queue drained @ $(date -Iseconds)"
