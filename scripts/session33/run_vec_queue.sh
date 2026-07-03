#!/bin/bash
# SESSION 33 D250 native-pooled-pipeline retrain queue.
#
# The flagship's training predictor becomes the v2.1 AutoregressivePredictor
# (cond_dim=0) rolling the (B, T, 32) pooled vector open-loop -- no tiling, no
# spatial ResUNet (user requirement, HANDOFF D250): the latent pipeline is now
# structurally identical to POD's / the AE's (32 numbers in, a vector dynamics
# model, 32 numbers out).
#
# 4 runs, 10k iters each (~55 min/GPU). Work-stealing clone of
# run_training_queue.sh: one worker per GPU, atomic mkdir locks. Usage:
#   scripts/session33/run_vec_queue.sh <gpu 0|1> <cores e.g. 0-7>
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

# name|config|extra-flags (all canonical trainer, all --predictor-class transformer)
QUEUE=(
  "jepa_pool_vec|configs/ablation/jepa_pool.yaml|--seed 0"
  "jepa_pool_vec_s1|configs/ablation/jepa_pool.yaml|--seed 1"
  "jepa_pool_vec_s2|configs/ablation/jepa_pool.yaml|--seed 2"
  "jepa_nowake_pool_vec|configs/ablation/jepa_nowake_pool.yaml|--seed 0"
)

for item in "${QUEUE[@]}"; do
  IFS='|' read -r name cfg extra <<< "$item"
  if [ -f "$RUNS/$name/checkpoint_iter010000.pt" ]; then
    echo "[vecq gpu$GPU] skip (done): $name"
    continue
  fi
  if ! mkdir "$LOCKS/$name" 2>/dev/null; then
    echo "[vecq gpu$GPU] skip (claimed): $name"
    continue
  fi
  mkdir -p "$RUNS/$name"
  echo "[vecq gpu$GPU] start: $name @ $(date -Iseconds)"
  taskset -c "$CORES" python -m src.training.train_canonical \
    --config "$cfg" \
    --partition v2p2 --pipeline-manifest "$PIPE" \
    --predictor-class transformer \
    --gpu "$GPU" --max-iters 10000 --num-workers 3 \
    --diagnostic-every 1000 --checkpoint-every 2500 --log-every 200 \
    --wandb-mode offline \
    $extra \
    --out "$RUNS/$name" > "$RUNS/$name/train.log" 2>&1 \
    || echo "[vecq gpu$GPU] FAILED: $name" >> "$RUNS/failures.log"
  echo "[vecq gpu$GPU] done: $name @ $(date -Iseconds)"
done

echo "[vecq gpu$GPU] queue drained @ $(date -Iseconds)"
