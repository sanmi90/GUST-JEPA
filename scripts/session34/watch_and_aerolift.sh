#!/bin/bash
# Fires the AeroJEPA+lift arm on GPU 0 after the regular (no-lift) AeroJEPA arm
# finishes -- user-directed sequencing ("after regular Aerojepa").
set -u
cd /home/carlos/GUST-JEPA
source .venv/bin/activate
export PREVENT_ROOT=$HOME/PREVENT OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
while [ ! -f outputs/runs/session34/aerojepa_nolift_s0/summary.json ]; do
  sleep 120
done
echo "[aerolift-watch] regular AeroJEPA arm done @ $(date -Iseconds); starting lift variant"
taskset -c 0-15 python -m scripts.session34.run_aerojepa_nolift --gpu 0 \
  --lift-weight 1.0 \
  --out outputs/runs/session34/aerojepa_lift_s0 \
  > outputs/session34/aerojepa_lift.log 2>&1
echo "[aerolift-watch] lift arm done @ $(date -Iseconds) (exit $?)"
tail -3 outputs/runs/session34/aerojepa_lift_s0/train.log 2>/dev/null
