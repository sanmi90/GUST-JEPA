#!/bin/bash
# SESSION 33 overnight chain: wait for both training-queue workers to drain,
# then run the Q1/Q2 eval queue (GPU 0) and the three analysis scripts.
# Idempotent; safe to re-run. Logs to outputs/session33/overnight_chain.log.
set -u
cd /home/carlos/GUST-JEPA

G0=outputs/runs/session33_queue_gpu0.log
G1=outputs/runs/session33_queue_gpu1.log

echo "[chain] waiting for both queues to drain @ $(date -Iseconds)"
until grep -q "queue drained" "$G0" 2>/dev/null && grep -q "queue drained" "$G1" 2>/dev/null; do
  sleep 60
done
echo "[chain] queues drained @ $(date -Iseconds)"

if [ -s outputs/runs/session33/failures.log ]; then
  echo "[chain] TRAINING FAILURES DETECTED:"
  cat outputs/runs/session33/failures.log
  echo "[chain] continuing with the runs that exist; evals of missing models will fail loudly"
fi

echo "[chain] eval queue (gpu 0) @ $(date -Iseconds)"
scripts/session33/run_eval_queue.sh 0
echo "[chain] eval queue done @ $(date -Iseconds)"

source .venv/bin/activate
export PREVENT_ROOT=$HOME/PREVENT OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
for s in dim_plateau min_d_panel seed_band_v3; do
  echo "[chain] analysis: $s @ $(date -Iseconds)"
  taskset -c 16-23 python -m scripts.session33.$s || echo "[chain] ANALYSIS FAILED: $s"
done
echo "[chain] ALL DONE @ $(date -Iseconds)"
