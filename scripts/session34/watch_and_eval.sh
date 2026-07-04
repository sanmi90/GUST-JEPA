#!/bin/bash
# Waits until every Track C queue run has its done-marker checkpoint, then
# fires the eval chain. Armed in the background right after the queue starts.
set -u
cd /home/carlos/GUST-JEPA
source .venv/bin/activate
while true; do
  MISSING=$(python -c "
from scripts.session34.trackc_cells import all_run_names, RUNS_BASE, CHECKPOINT, REUSED
new = [rn for rn in all_run_names() if rn not in REUSED]
print(sum(1 for rn in new if not (RUNS_BASE / rn / CHECKPOINT).exists()))
" 2>/dev/null || echo err)
  if [ "$MISSING" = "0" ]; then
    break
  fi
  if [ -s outputs/runs/session34/failures.log ]; then
    echo "[watch] training failures detected:"; cat outputs/runs/session34/failures.log
    # keep waiting for the rest; the eval chain aborts on missing checkpoints
  fi
  sleep 120
done
echo "[watch] all runs done @ $(date -Iseconds); firing eval chain"
bash scripts/session34/run_trackc_eval.sh
