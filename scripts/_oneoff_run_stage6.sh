#!/usr/bin/env bash
# Sequential driver for Stage 6 (Session 16) experiments.
# Order follows RERUN_MANIFEST.md Stage 6 table.
# Each script's stdout/stderr appended to outputs/runs/stage6_driver.log.

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
source "$REPO/.venv/bin/activate"
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export VORTEX_JEPA_CACHE="${VORTEX_JEPA_CACHE:-$PREVENT_ROOT/data/processed/vortex-jepa}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

LOG="$REPO/outputs/runs/stage6_driver.log"
mkdir -p "$(dirname "$LOG")"
echo "[stage6] start at $(date -Iseconds)" | tee -a "$LOG"

SCRIPTS=(
    "scripts/session16/exp1a_pls_base.py"
    "scripts/session16/exp1a_pca_base.py"
    "scripts/session16/exp1a_diagnostics.py"
    "scripts/session16/exp1a_bis_nonlinear.py"
    "scripts/session16/exp1a_ter_followups.py"
    "scripts/session16/exp1b_decode_axes.py"
    "scripts/session16/exp1b_axis_summary.py"
    "scripts/session16/exp1c_seed_variance.py"
    "scripts/session16/exp1c_pairwise.py"
    "scripts/session16/exp4_markov_closure.py"
    "scripts/session16/exp4_cond_ablation.py"
    "scripts/session16/exp4_figure.py"
    "scripts/session16/exp2_probe_sweep.py"
    "scripts/session16/exp2_redo_probes.py"
    "scripts/session16/exp2_figure.py"
    "scripts/session16/exp3_shap.py"
    "scripts/session16/exp3_bootstrap.py"
    "scripts/session16/exp3_intervention.py"
    "scripts/session16/exp3_figure.py"
    "scripts/session16/exp3_figure_v2.py"
    "scripts/session16/exp3_shap_Y.py"
    "scripts/session16/exp3_shap_Y_figure.py"
)

ok=0; fail=0; skip=0
for s in "${SCRIPTS[@]}"; do
    if [[ ! -f "$s" ]]; then
        echo "[stage6] SKIP (missing): $s" | tee -a "$LOG"
        skip=$((skip+1))
        continue
    fi
    echo | tee -a "$LOG"
    echo "[stage6] >>> $s ($(date -Iseconds))" | tee -a "$LOG"
    t0=$(date +%s)
    if python -u "$s" >> "$LOG" 2>&1; then
        rc=0
    else
        rc=$?
    fi
    dt=$(($(date +%s) - t0))
    if [[ $rc -eq 0 ]]; then
        echo "[stage6] OK    $s  ${dt}s" | tee -a "$LOG"
        ok=$((ok+1))
    else
        echo "[stage6] FAIL  $s  rc=$rc  ${dt}s" | tee -a "$LOG"
        fail=$((fail+1))
    fi
done

echo | tee -a "$LOG"
echo "[stage6] end at $(date -Iseconds)  ok=$ok fail=$fail skip=$skip" | tee -a "$LOG"
