#!/bin/bash
# SESSION 35 T5 chain: waits for the P1 phase-B eval chain on GPU 0
# (.p1_evalA_done), then runs the full two-stage + NIS-band-tuning pipeline
# on GPU 0: data (encode test_a/test_c) -> tune (pre-registered band grid on
# test_a) -> frozen (one-shot test_b + test_c at c*, anchor at 1.77).
#
# Launch: nohup bash scripts/session35/run_t5.sh > outputs/session35/t5.log 2>&1 &
set -u
cd /home/carlos/GUST-JEPA
source .venv/bin/activate
export PREVENT_ROOT=$HOME/PREVENT WANDB_PROJECT=vortex-jepa OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
S35=outputs/session35

while [ ! -f $S35/.p1_evalA_done ]; do sleep 120; done
echo "[t5-chain] evalA done marker seen @ $(date -Iseconds); starting T5"

taskset -c 0-7 python -m scripts.session35.two_stage_envelope --gpu 0 --stage all \
  > $S35/two_stage_envelope.log 2>&1
rc=$?
echo "[t5-chain] two_stage_envelope exited rc=$rc @ $(date -Iseconds)"
if [ $rc -eq 0 ]; then
  echo done > $S35/.t5_done
else
  echo failed > $S35/.t5_failed
fi
