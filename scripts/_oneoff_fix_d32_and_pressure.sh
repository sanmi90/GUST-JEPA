#!/usr/bin/env bash
# Three-task chain:
#  A. Fix d=32 cascade failures (regenerate prerequisites, patch n_components,
#     re-run failed scripts) → completes outputs/session16_d32/ + session17_d32/
#  B. Baseline pressure observability for Fukami AE × 3 + POD × 3 + JEPA d=32
#     → outputs/session18/exp_b1_test3/baseline_pressure_observability.csv
#  C. Pre-impact forecast (advance-warning curve) for all baselines
#     → outputs/session18/exp_b1_test3/preimpact_forecast.csv

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
source "$REPO/.venv/bin/activate"
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export VORTEX_JEPA_CACHE="${VORTEX_JEPA_CACHE:-$PREVENT_ROOT/data/processed/vortex-jepa}"

LOG="$REPO/outputs/runs/d32_fix_and_pressure.log"
mkdir -p "$(dirname "$LOG")"
echo "[d32-fix-pressure] start at $(date -Iseconds)" | tee "$LOG"

TS=$(date +%Y%m%d_%H%M)
BACKUP_DIR="/tmp/d32_fix_scripts_backup_${TS}"
mkdir -p "$BACKUP_DIR"
cp -r scripts/session16 "$BACKUP_DIR/session16"
cp -r scripts/session17 "$BACKUP_DIR/session17"

# ============================================================
# TASK A: fix d=32 cascade
# ============================================================
echo | tee -a "$LOG"
echo "=== TASK A: fix d=32 cascade ===" | tee -a "$LOG"

# Stash d=64 outputs and restore d=32 outputs to canonical paths
mv outputs/session16 outputs/session16_d64_stash_${TS}
mv outputs/session17 outputs/session17_d64_stash_${TS}
mv outputs/session16_d32 outputs/session16
mv outputs/session17_d32 outputs/session17
echo "  swapped d=64 ↔ d=32 outputs" | tee -a "$LOG"

# Sed scripts to d=32 paths
do_sed () {
    local p_from="$1"; local p_to="$2"
    for f in scripts/session16/*.py scripts/session17/*.py; do
        [[ -f "$f" ]] || continue
        sed -i "s|${p_from}|${p_to}|g" "$f"
    done
}
do_sed "S12_E_d64" "S12_E_d32"
do_sed "jepa_d64_seed" "jepa_d32_seed"

# Patch exp1a_projections for d=32 (n_components must be <= 32)
sed -i 's|PCA(n_components=64)|PCA(n_components=min(64, X.shape[1]))|g' scripts/session17/exp1a_projections.py
sed -i 's|PLSRegression(n_components=64)|PLSRegression(n_components=min(64, X.shape[1]))|g' scripts/session17/exp1a_projections.py
echo "  patched exp1a_projections.py n_components" | tee -a "$LOG"

# Copy encoder-agnostic DNS metrics from the d=64 stash (it's identical for d=32)
mkdir -p outputs/session17/exp2
if [ -f "outputs/session17_d64_stash_${TS}/exp2/dns_physical_metrics.npz" ]; then
    cp "outputs/session17_d64_stash_${TS}/exp2/dns_physical_metrics.npz" \
       outputs/session17/exp2/dns_physical_metrics.npz
    echo "  copied DNS metrics (encoder-agnostic)" | tee -a "$LOG"
fi

# Regenerate per_frame_targets for d=32 (depends on d=32 latents)
echo "  regenerating per_frame_targets for d=32" | tee -a "$LOG"
mkdir -p outputs/session16/exp2
python -u scripts/session16/exp2_build_targets.py >> "$LOG" 2>&1 \
    || echo "  FAIL per_frame_targets build" | tee -a "$LOG"

# Re-run failed scripts in dependency order
echo "  re-running failed S16/S17 scripts" | tee -a "$LOG"
FAILED_S16=(
    scripts/session16/exp2_probe_sweep.py
    scripts/session16/exp2_redo_probes.py
    scripts/session16/exp2_figure.py
    scripts/session16/exp3_shap.py
    scripts/session16/exp3_bootstrap.py
    scripts/session16/exp3_intervention.py
    scripts/session16/exp3_figure.py
    scripts/session16/exp3_figure_v2.py
)
FAILED_S17=(
    scripts/session17/exp1a_projections.py
    scripts/session17/exp1b_trajectory_panel.py
    scripts/session17/exp1d_cross_seed.py
    scripts/session17/exp1_day1_summary.py
    scripts/session17/exp2_rollouts_and_probes.py
    scripts/session17/exp2_aggregate.py
    scripts/session17/exp4_structures_shap.py
    scripts/session17/exp5_nonlinear.py
)
for s in "${FAILED_S16[@]}" "${FAILED_S17[@]}"; do
    echo "  >>> $s" | tee -a "$LOG"
    t0=$(date +%s)
    if python -u "$s" >> "$LOG" 2>&1; then
        echo "  OK   $s  $(($(date +%s) - t0))s" | tee -a "$LOG"
    else
        echo "  FAIL $s  $(($(date +%s) - t0))s" | tee -a "$LOG"
    fi
done

# Restore script paths from backup BEFORE Task B (which needs originals)
rm -rf scripts/session16 scripts/session17
cp -r "$BACKUP_DIR/session16" scripts/session16
cp -r "$BACKUP_DIR/session17" scripts/session17

# Restore d=64 outputs
mv outputs/session16 outputs/session16_d32
mv outputs/session17 outputs/session17_d32
mv outputs/session16_d64_stash_${TS} outputs/session16
mv outputs/session17_d64_stash_${TS} outputs/session17
echo "[d32-fix-pressure] Task A done. d=32 outputs at outputs/session16_d32, outputs/session17_d32" | tee -a "$LOG"

# ============================================================
# TASK B: baseline pressure observability
# ============================================================
echo | tee -a "$LOG"
echo "=== TASK B: baseline pressure observability ===" | tee -a "$LOG"
python -u scripts/_oneoff_baseline_pressure_obs.py >> "$LOG" 2>&1 \
    || echo "  FAIL Task B" | tee -a "$LOG"

# ============================================================
# TASK C: pre-impact forecast (advance-warning curve)
# ============================================================
echo | tee -a "$LOG"
echo "=== TASK C: pre-impact forecast ===" | tee -a "$LOG"
python -u scripts/_oneoff_preimpact_forecast.py >> "$LOG" 2>&1 \
    || echo "  FAIL Task C" | tee -a "$LOG"

echo | tee -a "$LOG"
echo "[d32-fix-pressure] DONE at $(date -Iseconds)" | tee -a "$LOG"
