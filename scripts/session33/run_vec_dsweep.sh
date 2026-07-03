#!/bin/bash
# SESSION 33 D250: vec dimension sweep to close the plateau + min-d split-brain.
# jepa_pool_vec at d in {4,8,16,64} (d=32 is the flagship, already trained), all
# with the native vector transformer predictor. Work-stealing, one worker per GPU.
# Usage: scripts/session33/run_vec_dsweep.sh <gpu 0|1> <cores e.g. 0-7>
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

QUEUE=(
  "jepa_pool_vec_d16|--d 16 --seed 0"
  "jepa_pool_vec_d64|--d 64 --seed 0"
  "jepa_pool_vec_d8|--d 8 --seed 0"
  "jepa_pool_vec_d4|--d 4 --seed 0"
)

for item in "${QUEUE[@]}"; do
  IFS='|' read -r name extra <<< "$item"
  if [ -f "$RUNS/$name/checkpoint_iter010000.pt" ]; then
    echo "[dsweep gpu$GPU] skip (done): $name"; continue
  fi
  if ! mkdir "$LOCKS/$name" 2>/dev/null; then
    echo "[dsweep gpu$GPU] skip (claimed): $name"; continue
  fi
  mkdir -p "$RUNS/$name"
  echo "[dsweep gpu$GPU] start: $name @ $(date -Iseconds)"
  taskset -c "$CORES" python -m src.training.train_canonical \
    --config configs/ablation/jepa_pool.yaml \
    --partition v2p2 --pipeline-manifest "$PIPE" \
    --predictor-class transformer \
    --gpu "$GPU" --max-iters 10000 --num-workers 3 \
    --diagnostic-every 1000 --checkpoint-every 2500 --log-every 200 \
    --wandb-mode offline \
    $extra \
    --out "$RUNS/$name" > "$RUNS/$name/train.log" 2>&1 \
    || echo "[dsweep gpu$GPU] FAILED: $name" >> "$RUNS/failures.log"
  echo "[dsweep gpu$GPU] done: $name @ $(date -Iseconds)"
done
echo "[dsweep gpu$GPU] queue drained @ $(date -Iseconds)"
