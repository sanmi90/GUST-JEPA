#!/usr/bin/env bash
# Session 8 finalizer (called after both Step 4 orchestrators complete).
# 1. Verify all expected grid checkpoints exist.
# 2. Run grid_analysis.py (analyses E1-E9 + E10 + Session 7 R3 = E5 + R1_S7).
# 3. Show the best_grid_point.json and champion_table.csv.
# 4. Launch Step 5 + 6 chain via orchestrate_session8_step56.sh.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

OUTROOT="outputs/runs/session8"
SUMMARY="$OUTROOT/finalize.log"
: > "$SUMMARY"

log() {
    echo "[$(date -Iseconds)] $*" | tee -a "$SUMMARY"
}

log "Session 8 finalizer starting"

# Check expected grid runs
EXPECTED_S8=(
    "run_e1_eta0p001_lam0p01"
    "run_e2_eta0p001_lam0p10"
    "run_e3_eta0p001_lam1p00"
    "run_e4_eta0p010_lam0p01"
    "run_e6_eta0p010_lam1p00"
    "run_e7_eta0p100_lam0p01"
    "run_e8_eta0p100_lam0p10"
    "run_e9_eta0p100_lam1p00"
    "run_e10_pldm_paper_tuned"
)
log "Checking expected Session 8 grid runs:"
ALL_PRESENT=1
for name in "${EXPECTED_S8[@]}"; do
    ckpt="$OUTROOT/$name/checkpoint_iter020000.pt"
    if [ -f "$ckpt" ]; then
        log "  OK $name"
    else
        log "  MISSING $name (no $ckpt)"
        ALL_PRESENT=0
    fi
done
if [ "$ALL_PRESENT" = "0" ]; then
    log "Some runs missing; proceeding anyway (grid_analysis.py skips missing)."
fi

# Run grid analysis
log "Running session8_grid_analysis.py"
python scripts/session8_grid_analysis.py \
    --gpu 1 \
    --output-dir outputs/runs/session8 \
    >"$OUTROOT/grid_analysis.log" 2>&1
RC=$?
log "grid_analysis exit=$RC"
if [ $RC -ne 0 ]; then
    log "Grid analysis failed; inspect $OUTROOT/grid_analysis.log"
    tail -30 "$OUTROOT/grid_analysis.log" | tee -a "$SUMMARY"
    exit $RC
fi

# Print best grid point and champion table
if [ -f "$OUTROOT/best_grid_point.json" ]; then
    log "Best grid point:"
    cat "$OUTROOT/best_grid_point.json" | tee -a "$SUMMARY"
fi
if [ -f "$OUTROOT/champion_table.csv" ]; then
    log "Champion table:"
    cat "$OUTROOT/champion_table.csv" | tee -a "$SUMMARY"
fi

log "Step 4 grid analysis complete. Launching Step 5 + 6 chain (~3h)."
exec bash scripts/orchestrate_session8_step56.sh
