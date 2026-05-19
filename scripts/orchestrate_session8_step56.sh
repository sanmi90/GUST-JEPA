#!/usr/bin/env bash
# Session 8 Steps 5 + 6 chain.
#
# Assumes:
# - Step 4 grid is complete and grid_analysis.py has been run.
# - outputs/runs/session8/best_grid_point.json exists with eta, lambda, run_dir.
#
# Sequence:
# 1. Read best (eta*, lambda*) from best_grid_point.json.
# 2. Step 5 d-sweep: launch d=8 on cuda:0 and d=16 on cuda:1 in parallel (1.5h).
# 3. After Step 5 done: run d_sweep_analysis.py.
# 4. Step 6 R0: launch on cuda:0 (lambda=0.1) (1.5h).
#    (If lambda* > 0.5 also launch a second R0 at lambda* on cuda:1 in parallel.)
# 5. After R0 done: print final status.
#
# Total wall-clock: ~3h.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

OUTROOT="outputs/runs/session8"
SUMMARY="$OUTROOT/orchestrate_step56.log"
: > "$SUMMARY"

log() {
    echo "[$(date -Iseconds)] $*" | tee -a "$SUMMARY"
}

BEST_JSON="$OUTROOT/best_grid_point.json"
if [ ! -f "$BEST_JSON" ]; then
    log "ERROR: $BEST_JSON not found. Run grid_analysis.py first."
    exit 1
fi

ETA_STAR=$(python -c "import json; print(json.load(open('$BEST_JSON'))['eta'])")
LAM_STAR=$(python -c "import json; print(json.load(open('$BEST_JSON'))['lambda'])")
BEST_RUN_DIR=$(python -c "import json; print(json.load(open('$BEST_JSON'))['run_dir'])")
BEST_CODE=$(python -c "import json; print(json.load(open('$BEST_JSON'))['code'])")
log "Best grid point: $BEST_CODE eta*=$ETA_STAR lambda*=$LAM_STAR dir=$BEST_RUN_DIR"

# Step 5: d-sweep (d=8 + d=16 in parallel)
log "Step 5: launching d-sweep"
bash scripts/launch_session8_step5_dsweep.sh "$ETA_STAR" "$LAM_STAR"
RC=$?
log "Step 5 d-sweep exit=$RC"
if [ $RC -ne 0 ]; then
    log "Step 5 had failures; investigate before proceeding"
    exit $RC
fi

# Step 5 analysis
log "Step 5 analysis: d_sweep_analysis.py"
python scripts/session8_d_sweep_analysis.py \
    --d32-run "$BEST_RUN_DIR" \
    --d16-run "outputs/runs/session8/run_d16_best" \
    --d8-run "outputs/runs/session8/run_d8_best" \
    --gpu 1 \
    --output "outputs/runs/session8/d_sweep.csv" \
    >"$OUTROOT/d_sweep_analysis.log" 2>&1
log "d_sweep_analysis exit=$?"

# Step 6: R0 control on cuda:0 (lambda=0.1). Run a second R0 at lambda* on
# cuda:1 in parallel only if lambda* > 0.5 (significantly different from default).
log "Step 6: launching R0 SIGReg-only control at lambda=0.1 on cuda:0"
bash scripts/launch_session8_step6_r0.sh 0.1 0 &
PID_R0_DEFAULT=$!
log "  R0 lambda=0.1 pid=$PID_R0_DEFAULT"

# Parallel R0 at lambda* on cuda:1 if interesting
RUN_LAM_STAR_R0=$(python -c "lam = $LAM_STAR; print(1 if lam > 0.5 or lam < 0.05 else 0)")
if [ "$RUN_LAM_STAR_R0" = "1" ]; then
    log "lambda*=$LAM_STAR is significantly different from 0.1; running second R0 at lambda* on cuda:1"
    bash scripts/launch_session8_step6_r0.sh "$LAM_STAR" 1 &
    PID_R0_STAR=$!
    log "  R0 lambda*=$LAM_STAR pid=$PID_R0_STAR"
fi

# Wait for R0 runs
wait "$PID_R0_DEFAULT"
RC_R0_DEFAULT=$?
log "R0 lambda=0.1 exit=$RC_R0_DEFAULT"
if [ "$RUN_LAM_STAR_R0" = "1" ]; then
    wait "$PID_R0_STAR"
    RC_R0_STAR=$?
    log "R0 lambda*=$LAM_STAR exit=$RC_R0_STAR"
fi

log "Step 5 + 6 chain complete"
exit 0
