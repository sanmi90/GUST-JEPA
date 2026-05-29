#!/usr/bin/env bash
# Sequential driver for Stage 7 (Session 17) experiments.
# Order follows RERUN_MANIFEST.md Stage 7 table.
# Each script's stdout/stderr appended to outputs/runs/stage7_driver.log.

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
source "$REPO/.venv/bin/activate"
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export VORTEX_JEPA_CACHE="${VORTEX_JEPA_CACHE:-$PREVENT_ROOT/data/processed/vortex-jepa}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

LOG="$REPO/outputs/runs/stage7_driver.log"
mkdir -p "$(dirname "$LOG")"
echo "[stage7] start at $(date -Iseconds)" | tee -a "$LOG"

SCRIPTS=(
    "scripts/session17/exp1a_projections.py"
    "scripts/session17/exp1b_trajectory_panel.py"
    "scripts/session17/exp1c_curvature.py"
    "scripts/session17/exp1c_extra_signatures.py"
    "scripts/session17/exp1d_cross_seed.py"
    "scripts/session17/exp1_day1_summary.py"
    "scripts/session17/exp2_rollouts_and_probes.py"
    "scripts/session17/exp2_aggregate.py"
    "scripts/session17/exp3a_param_recovery.py"
    "scripts/session17/exp3b_decay_fit.py"
    "scripts/session17/exp3c_cross_seed_transfer.py"
    "scripts/session17/exp3d_shap_decay.py"
    "scripts/session17/exp4_structures_shap.py"
    "scripts/session17/exp5_nonlinear.py"
    "scripts/session17/diagnostic_d_znorm.py"
)

ok=0; fail=0; skip=0
for s in "${SCRIPTS[@]}"; do
    if [[ ! -f "$s" ]]; then
        echo "[stage7] SKIP (missing): $s" | tee -a "$LOG"
        skip=$((skip+1))
        continue
    fi
    echo | tee -a "$LOG"
    echo "[stage7] >>> $s ($(date -Iseconds))" | tee -a "$LOG"
    t0=$(date +%s)
    if python -u "$s" >> "$LOG" 2>&1; then
        rc=0
    else
        rc=$?
    fi
    dt=$(($(date +%s) - t0))
    if [[ $rc -eq 0 ]]; then
        echo "[stage7] OK    $s  ${dt}s" | tee -a "$LOG"
        ok=$((ok+1))
    else
        echo "[stage7] FAIL  $s  rc=$rc  ${dt}s" | tee -a "$LOG"
        fail=$((fail+1))
    fi
done

echo | tee -a "$LOG"
echo "[stage7] end at $(date -Iseconds)  ok=$ok fail=$fail skip=$skip" | tee -a "$LOG"
