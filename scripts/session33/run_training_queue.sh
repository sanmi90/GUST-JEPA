#!/bin/bash
# SESSION 33 training queue -- dimension plateau + min-d panel + 3-seed spine pass
# (SESSION_33_MANUSCRIPT_V3.md Section 11 items 2 and 8; HANDOFF D238).
#
# 11 runs, 10k iters each (~55 min/GPU). Work-stealing: each invocation is one
# worker pinned to one GPU; runs are claimed via atomic mkdir locks so a second
# worker can join later without double-running. Usage:
#   scripts/session33/run_training_queue.sh <gpu 0|1> <cores e.g. 0-7>
#
# Core budget per CLAUDE.md: OMP=4, num-workers 3, taskset core-split so >=64
# cores stay free for asolera. Offline W&B, group partition_v2p2.
set -u
cd /home/carlos/GUST-JEPA
source .venv/bin/activate
export PREVENT_ROOT=$HOME/PREVENT WANDB_PROJECT=vortex-jepa OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

GPU=$1
CORES=$2
RUNS=outputs/runs/session33
PIPE=outputs/data_pipeline/v2p2/manifest.json
LOCKS=$RUNS/.locks
mkdir -p "$RUNS" "$LOCKS"

# name|trainer|config|extra-flags
QUEUE=(
  "jepa_pool_d16|canonical|configs/ablation/jepa_pool.yaml|--d 16 --seed 0"
  "jepa_pool_d64|canonical|configs/ablation/jepa_pool.yaml|--d 64 --seed 0"
  "jepa_pool_s1|canonical|configs/ablation/jepa_pool.yaml|--seed 1"
  "jepa_pool_s2|canonical|configs/ablation/jepa_pool.yaml|--seed 2"
  "supervised_only_pool_s1|canonical|configs/ablation/supervised_only_pool.yaml|--seed 1"
  "supervised_only_pool_s2|canonical|configs/ablation/supervised_only_pool.yaml|--seed 2"
  "jepa_pool_d8|canonical|configs/ablation/jepa_pool.yaml|--d 8 --seed 0"
  "jepa_pool_d4|canonical|configs/ablation/jepa_pool.yaml|--d 4 --seed 0"
  "fukami_wake_d16|reference|configs/reference/fukami_wake.yaml|--d 16 --seed 0"
  "fukami_wake_d8|reference|configs/reference/fukami_wake.yaml|--d 8 --seed 0"
  "fukami_wake_d4|reference|configs/reference/fukami_wake.yaml|--d 4 --seed 0"
)

for item in "${QUEUE[@]}"; do
  IFS='|' read -r name trainer cfg extra <<< "$item"
  # already trained (checkpoint exists) -> skip
  if [ -f "$RUNS/$name/checkpoint_iter010000.pt" ]; then
    echo "[queue gpu$GPU] skip (done): $name"
    continue
  fi
  # atomic claim
  if ! mkdir "$LOCKS/$name" 2>/dev/null; then
    echo "[queue gpu$GPU] skip (claimed): $name"
    continue
  fi
  mkdir -p "$RUNS/$name"
  echo "[queue gpu$GPU] start: $name @ $(date -Iseconds)"
  if [ "$trainer" = "canonical" ]; then
    MOD=src.training.train_canonical
  else
    MOD=src.training.train_reference
  fi
  taskset -c "$CORES" python -m "$MOD" \
    --config "$cfg" \
    --partition v2p2 --pipeline-manifest "$PIPE" \
    --gpu "$GPU" --max-iters 10000 --num-workers 3 \
    --diagnostic-every 1000 --checkpoint-every 2500 --log-every 200 \
    --wandb-mode offline \
    $extra \
    --out "$RUNS/$name" > "$RUNS/$name/train.log" 2>&1 \
    || echo "[queue gpu$GPU] FAILED: $name" >> "$RUNS/failures.log"
  echo "[queue gpu$GPU] done: $name @ $(date -Iseconds)"
done

echo "[queue gpu$GPU] queue drained @ $(date -Iseconds)"
