#!/bin/bash
# SESSION 31 Track C.2 -- train the remaining canonical spine models 2-at-a-time
# across the two RTX 6000 cards, after wave 1 (jepa_nowake, ae_nowake) frees them.
# Core-split (0-7 / 8-15), OMP=4, num-workers 3 so >=64 cores stay free for asolera.
set -u
cd /home/carlos/GUST-JEPA
source .venv/bin/activate
export PREVENT_ROOT=$HOME/PREVENT WANDB_PROJECT=vortex-jepa OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

run() {  # model gpu cores
  local m=$1 gpu=$2 cores=$3
  mkdir -p outputs/runs/session31/"$m"
  taskset -c "$cores" python -m src.training.train_canonical \
    --config configs/canonical/"$m".yaml \
    --partition v2p2 --pipeline-manifest outputs/data_pipeline/v2p2/manifest.json \
    --gpu "$gpu" --seed 0 --max-iters 10000 --num-workers 3 \
    --diagnostic-every 1000 --checkpoint-every 2500 --log-every 200 \
    --wandb-mode offline --out outputs/runs/session31/"$m" \
    > outputs/runs/session31/"$m"/train.log 2>&1
}

echo "[orch] waiting for wave 1 (jepa_nowake, ae_nowake) to finish..."
while pgrep -f "train_canonical.*configs/canonical/jepa_nowake.yaml" >/dev/null \
   || pgrep -f "train_canonical.*configs/canonical/ae_nowake.yaml" >/dev/null; do
  sleep 30
done

echo "[orch] wave 2: jepa_wake (gpu0) + ae_wake (gpu1)"
run jepa_wake 0 0-7 &
run ae_wake 1 8-15 &
wait

echo "[orch] wave 3: supervised_only (gpu0) + regAE (gpu1)"
run supervised_only 0 0-7 &
run regAE 1 8-15 &
wait

echo "[orch] canonical spine complete"
