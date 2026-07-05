#!/bin/bash
# Loss-weight sweep for the aerojepa_lift arm (user-directed, Session 34).
# Waits for the low-d runs to drain both GPUs, then runs 4 configs (2/GPU).
# Axes: SIGReg weight (fix the PR-3.2 concentration) x lift weight.
set -u
cd /home/carlos/GUST-JEPA
source .venv/bin/activate
export PREVENT_ROOT=$HOME/PREVENT OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
while [ ! -f outputs/session34/.aero_lowd_gpu0_done ] || [ ! -f outputs/session34/.aero_lowd_gpu1_done ]; do
  sleep 300
done
echo "[sweep] low-d done @ $(date -Iseconds); starting weight sweep"
run() {  # gpu cores sig lift tag
  taskset -c "$2" python -m scripts.session34.run_aerojepa_nolift --gpu "$1" \
    --lift-weight "$4" --sig-weight "$3" \
    --out "outputs/runs/session34/aerojepa_lift_$5" \
    > "outputs/session34/aerojepa_sweep_$5.log" 2>&1
}
( run 0 0-7  0.5 1.0 sig05_lift10 ; run 0 0-7  0.1 3.0 sig01_lift30 ; echo done > outputs/session34/.sweep_gpu0_done ) &
( run 1 8-15 1.0 1.0 sig10_lift10 ; run 1 8-15 0.5 0.3 sig05_lift03 ; echo done > outputs/session34/.sweep_gpu1_done ) &
wait
echo "[sweep] all configs done @ $(date -Iseconds)"
for T in sig05_lift10 sig01_lift30 sig10_lift10 sig05_lift03; do
  grep "SUMMARY" "outputs/runs/session34/aerojepa_lift_$T/train.log" | sed "s/^/[$T] /"
done
