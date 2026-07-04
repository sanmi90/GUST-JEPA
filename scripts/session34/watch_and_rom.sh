#!/bin/bash
# Fires the SIGReg-JEPA-ROM no-lift arm (full run, GPU 0) once the Track C
# eval chain has finished (complete OR aborted -- either way the GPUs are free).
set -u
cd /home/carlos/GUST-JEPA
source .venv/bin/activate
export PREVENT_ROOT=$HOME/PREVENT OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
while true; do
  if grep -qE "chain complete|ABORT" outputs/session34/eval_chain.log 2>/dev/null; then
    break
  fi
  sleep 120
done
echo "[rom-watch] eval chain finished @ $(date -Iseconds); starting ROM no-lift arm"
taskset -c 0-15 python -m scripts.session34.run_rom_nolift --gpu 0 \
  > outputs/session34/rom_nolift.log 2>&1
echo "[rom-watch] ROM arm done @ $(date -Iseconds) (exit $?)"
tail -3 outputs/runs/session34/rom_nolift_s0/train.log 2>/dev/null
