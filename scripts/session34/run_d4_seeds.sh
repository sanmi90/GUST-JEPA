#!/bin/bash
# d=4 seed bands (Session 34): the low-d race claim needs 3 seeds per family.
# Chained behind the aerojepa weight sweep.
set -u
cd /home/carlos/GUST-JEPA
source .venv/bin/activate
export PREVENT_ROOT=$HOME/PREVENT WANDB_PROJECT=vortex-jepa OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
while [ ! -f outputs/session34/.sweep_gpu0_done ] || [ ! -f outputs/session34/.sweep_gpu1_done ]; do
  sleep 300
done
echo "[d4seeds] sweep done @ $(date -Iseconds); starting"
PIPE=outputs/data_pipeline/v2p2/manifest.json
R=outputs/runs/session34
run_canon() { # gpu cores name config seed extra
  taskset -c "$2" python -m src.training.train_canonical --config "$4" \
    --partition v2p2 --pipeline-manifest $PIPE --d 4 --gpu "$1" --seed "$5" \
    --max-iters 10000 --num-workers 3 --diagnostic-every 1000 \
    --checkpoint-every 2500 --log-every 200 --wandb-mode offline $6 \
    --out "$R/$3" > "$R/$3/train.log" 2>&1
}
( mkdir -p $R/jepa_pool_vec_d4_s1 $R/fukami_wake_d4_s1 $R/aerojepa_lift_d4_s1
  run_canon 0 0-7 jepa_pool_vec_d4_s1 configs/ablation/jepa_pool.yaml 1 "--predictor-class transformer"
  taskset -c 0-7 python -m src.training.train_reference --config configs/reference/fukami_wake.yaml \
    --partition v2p2 --pipeline-manifest $PIPE --d 4 --gpu 0 --seed 1 --max-iters 10000 \
    --num-workers 3 --wandb-mode offline --out $R/fukami_wake_d4_s1 > $R/fukami_wake_d4_s1/train.log 2>&1
  taskset -c 0-7 python -m scripts.session34.run_aerojepa_nolift --gpu 0 --r 4 --seed 1 \
    --lift-weight 1.0 --out $R/aerojepa_lift_d4_s1 > outputs/session34/aero_d4_s1.log 2>&1
  echo done > outputs/session34/.d4_gpu0_done ) &
( mkdir -p $R/jepa_pool_vec_d4_s2 $R/fukami_wake_d4_s2 $R/aerojepa_lift_d4_s2
  run_canon 1 8-15 jepa_pool_vec_d4_s2 configs/ablation/jepa_pool.yaml 2 "--predictor-class transformer"
  taskset -c 8-15 python -m src.training.train_reference --config configs/reference/fukami_wake.yaml \
    --partition v2p2 --pipeline-manifest $PIPE --d 4 --gpu 1 --seed 2 --max-iters 10000 \
    --num-workers 3 --wandb-mode offline --out $R/fukami_wake_d4_s2 > $R/fukami_wake_d4_s2/train.log 2>&1
  taskset -c 8-15 python -m scripts.session34.run_aerojepa_nolift --gpu 1 --r 4 --seed 2 \
    --lift-weight 1.0 --out $R/aerojepa_lift_d4_s2 > outputs/session34/aero_d4_s2.log 2>&1
  echo done > outputs/session34/.d4_gpu1_done ) &
wait
echo "[d4seeds] all done @ $(date -Iseconds)"
