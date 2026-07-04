#!/bin/bash
# Fires the AeroJEPA-style full-encoder no-lift arm on GPU 0 once the ROM
# no-lift arm has finished (its summary.json exists), i.e. third in the
# overnight chain: Track C queue -> eval chain -> ROM arm -> this.
set -u
cd /home/carlos/GUST-JEPA
source .venv/bin/activate
export PREVENT_ROOT=$HOME/PREVENT OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
while [ ! -f outputs/runs/session34/rom_nolift_s0/summary.json ]; do
  sleep 120
done
echo "[aero-watch] ROM arm done @ $(date -Iseconds); starting AeroJEPA-style arm"
taskset -c 0-15 python -m scripts.session34.run_aerojepa_nolift --gpu 0 \
  > outputs/session34/aerojepa_nolift.log 2>&1
echo "[aero-watch] AeroJEPA arm done @ $(date -Iseconds) (exit $?)"
tail -3 outputs/runs/session34/aerojepa_nolift_s0/train.log 2>/dev/null
