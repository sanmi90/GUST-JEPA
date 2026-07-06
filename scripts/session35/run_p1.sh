#!/bin/bash
# SESSION 35 P1 gap runs (T1-T4 compute + T3 filter/stream seeds).
#
# Phase A (one worker per RTX 6000, ~90 min/GPU):
#   gpu0: jepa_pool_ln_rexpred_s1 (T1) -> fukami_wake_d16_s1 (T4) -> rex2_cov s1 (T2)
#   gpu1: jepa_pool_ln_rexpred_s2 (T1) -> fukami_wake_d16_s2 (T4) -> rex2_cov s2 (T2)
# Phase B (parallel eval chains):
#   gpu0: trackc_encode (4 new runs) -> rexpred_band (T1 gate) -> da_fk16_seeds (T4 gate)
#   gpu1: rex_filter tuned seeds 1-4 (T3) -> rex_stream band4.0 seeds 1-2 +
#         band1.77 seeds 0-2, noise {0,0.05,0.1,0.2} (T3 amendment)
#
# Launch: nohup bash scripts/session35/run_p1.sh > outputs/session35/p1.log 2>&1 &
# (then disown; background Bash-tool tasks die with their wrapper, Session 34 lesson.)
set -u
cd /home/carlos/GUST-JEPA
source .venv/bin/activate
export PREVENT_ROOT=$HOME/PREVENT WANDB_PROJECT=vortex-jepa OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

R=outputs/runs/session34
S35=outputs/session35
PIPE=outputs/data_pipeline/v2p2/manifest.json
mkdir -p "$S35" "$R/jepa_pool_ln_rexpred_s1" "$R/jepa_pool_ln_rexpred_s2" \
         "$R/fukami_wake_d16_s1" "$R/fukami_wake_d16_s2"

echo "[p1] phase A start @ $(date -Iseconds)"

(
  taskset -c 0-7 python -m src.training.train_canonical \
    --config configs/ablation/jepa_pool_ln.yaml --predictor-class rex \
    --partition v2p2 --pipeline-manifest $PIPE --d 32 --gpu 0 --seed 1 \
    --max-iters 10000 --num-workers 3 --diagnostic-every 1000 \
    --checkpoint-every 2500 --log-every 200 --wandb-mode offline \
    --out $R/jepa_pool_ln_rexpred_s1 > $R/jepa_pool_ln_rexpred_s1/train.log 2>&1
  echo "[p1 gpu0] rexpred_s1 done @ $(date -Iseconds)"
  taskset -c 0-7 python -m src.training.train_reference \
    --config configs/reference/fukami_wake.yaml \
    --partition v2p2 --pipeline-manifest $PIPE --d 16 --gpu 0 --seed 1 \
    --max-iters 10000 --num-workers 3 --wandb-mode offline \
    --out $R/fukami_wake_d16_s1 > $R/fukami_wake_d16_s1/train.log 2>&1
  echo "[p1 gpu0] fukami_d16_s1 done @ $(date -Iseconds)"
  taskset -c 0-7 python -m scripts.session34.rex2_cov --gpu 0 --seed 1 \
    --out $S35/rex2_cov_s1.json > $S35/rex2_cov_s1.log 2>&1
  echo "[p1 gpu0] rex2_cov_s1 done @ $(date -Iseconds)"
  echo done > $S35/.p1_gpu0_done
) &

(
  taskset -c 8-15 python -m src.training.train_canonical \
    --config configs/ablation/jepa_pool_ln.yaml --predictor-class rex \
    --partition v2p2 --pipeline-manifest $PIPE --d 32 --gpu 1 --seed 2 \
    --max-iters 10000 --num-workers 3 --diagnostic-every 1000 \
    --checkpoint-every 2500 --log-every 200 --wandb-mode offline \
    --out $R/jepa_pool_ln_rexpred_s2 > $R/jepa_pool_ln_rexpred_s2/train.log 2>&1
  echo "[p1 gpu1] rexpred_s2 done @ $(date -Iseconds)"
  taskset -c 8-15 python -m src.training.train_reference \
    --config configs/reference/fukami_wake.yaml \
    --partition v2p2 --pipeline-manifest $PIPE --d 16 --gpu 1 --seed 2 \
    --max-iters 10000 --num-workers 3 --wandb-mode offline \
    --out $R/fukami_wake_d16_s2 > $R/fukami_wake_d16_s2/train.log 2>&1
  echo "[p1 gpu1] fukami_d16_s2 done @ $(date -Iseconds)"
  taskset -c 8-15 python -m scripts.session34.rex2_cov --gpu 1 --seed 2 \
    --out $S35/rex2_cov_s2.json > $S35/rex2_cov_s2.log 2>&1
  echo "[p1 gpu1] rex2_cov_s2 done @ $(date -Iseconds)"
  echo done > $S35/.p1_gpu1_done
) &

wait
echo "[p1] phase A done @ $(date -Iseconds)"

(
  taskset -c 0-7 python -m scripts.session34.trackc_encode --gpu 0 \
    --models jepa_pool_ln_rexpred_s1 jepa_pool_ln_rexpred_s2 \
             fukami_wake_d16_s1 fukami_wake_d16_s2 \
    > $S35/encode.log 2>&1
  echo "[p1 gpu0] encode done @ $(date -Iseconds)"
  taskset -c 0-7 python -m scripts.session35.rexpred_band \
    > $S35/rexpred_band.log 2>&1
  echo "[p1 gpu0] T1 band done @ $(date -Iseconds)"
  taskset -c 0-7 python -m scripts.session35.da_fk16_seeds --gpu 0 \
    > $S35/da_fk16.log 2>&1
  echo "[p1 gpu0] T4 fk16 DA done @ $(date -Iseconds)"
  echo done > $S35/.p1_evalA_done
) &

(
  for s in 1 2 3 4; do
    taskset -c 8-15 python -m scripts.session34.rex_filter --gpu 1 --tuned \
      --band-scale 1.77 --gamma-mode global --seed $s \
      --out $S35/rex_filter_tuned_s$s.json > $S35/rex_filter_tuned_s$s.log 2>&1
    echo "[p1 gpu1] rex_filter tuned s$s done @ $(date -Iseconds)"
  done
  for s in 1 2; do
    for nz in 0.0 0.05 0.1 0.2; do
      taskset -c 8-15 python -m scripts.session34.rex_stream --gpu 1 \
        --noise $nz --seed $s \
        --out $S35/rex_stream_noise${nz}_s$s.json \
        > $S35/rex_stream_noise${nz}_s$s.log 2>&1
    done
    echo "[p1 gpu1] rex_stream band4.0 seed $s done @ $(date -Iseconds)"
  done
  for s in 0 1 2; do
    for nz in 0.0 0.05 0.1 0.2; do
      taskset -c 8-15 python -m scripts.session34.rex_stream --gpu 1 \
        --noise $nz --seed $s --band-scale 1.77 \
        --out $S35/rex_stream_b177_noise${nz}_s$s.json \
        > $S35/rex_stream_b177_noise${nz}_s$s.log 2>&1
    done
    echo "[p1 gpu1] rex_stream band1.77 seed $s done @ $(date -Iseconds)"
  done
  echo done > $S35/.p1_evalB_done
) &

wait
echo "[p1] ALL P1 COMPUTE DONE @ $(date -Iseconds)"
echo done > $S35/.p1_done
