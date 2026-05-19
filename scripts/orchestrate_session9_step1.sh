#!/usr/bin/env bash
# Session 9 Step 1 orchestrator: chain F1 -> F2 -> F3 -> analysis ->
# (F4, F5 once lambda* is identified) on cuda:0, while Step 3 ablations
# proceed in parallel on cuda:1 via orchestrate_session9_step3.sh.
#
# Sequence on cuda:0:
#   1) Launch F1, F2, F3 sequentially (each ~1.5h, total ~4.5h).
#   2) Run scripts/session9_bisection_analysis.py to identify lambda* from
#      {F1, F2, E4 anchor, F3, E5 anchor}.
#   3) Launch F4 (seed=42 at lambda*) on cuda:0.
#   4) Launch F5 (seed=123 at lambda*) on cuda:1 if Step 3 ablations have
#      finished there; otherwise sequence F5 after F4 on cuda:0.
#
# The card 1 orchestrator (orchestrate_session9_step3.sh) handles Step 3.
#
# Usage:
#   scripts/orchestrate_session9_step1.sh
#
# Args via env variables:
#   ANALYSIS_GPU=1  (use cuda:1 for analysis; default cuda:1 frees cuda:0 for F4 prep)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

ANALYSIS_GPU="${ANALYSIS_GPU:-1}"

OUTROOT="outputs/runs/session9"
mkdir -p "$OUTROOT"
SUMMARY="$OUTROOT/orchestrate_step1.log"
: > "$SUMMARY"

log() {
    echo "[$(date -Iseconds)] $*" | tee -a "$SUMMARY"
}

log "Session 9 Step 1 orchestrator starting"
log "  repo: $REPO_ROOT"
log "  git HEAD: $(git rev-parse HEAD)"
log "  ANALYSIS_GPU=cuda:$ANALYSIS_GPU"

# Step 1.1: launch F1, F2, F3 on cuda:0 sequentially.
log "Step 1.1: launching F1, F2, F3 on cuda:0"
bash scripts/launch_session9_step1_bisection.sh 0 F1,F2,F3
RC=$?
log "Step 1.1 finished with exit=$RC"
if [ $RC -ne 0 ]; then
    log "F1/F2/F3 had failures; check $OUTROOT/launch_step1_card0.log"
fi

# Step 1.2: analyse the bisection (with anchors E4 and E5 from disk).
log "Step 1.2: running bisection analysis"
ANALYSIS_LOG="$OUTROOT/bisection_analysis_seed0.log"
python scripts/session9_bisection_analysis.py \
    --gpu "$ANALYSIS_GPU" --iters 20000 \
    --output-dir "$OUTROOT" \
    >"$ANALYSIS_LOG" 2>&1
RC=$?
log "Bisection analysis exit=$RC"
if [ $RC -ne 0 ]; then
    log "Analysis failed; check $ANALYSIS_LOG"
    log "Aborting Step 1; F4/F5 not launched"
    exit $RC
fi

BEST_FILE="$OUTROOT/best_lambda_star.json"
if [ ! -f "$BEST_FILE" ]; then
    log "ERROR: $BEST_FILE not written"
    exit 2
fi

BEST_LAMBDA=$(python -c "import json; print(json.load(open('$BEST_FILE'))['best_lambda'])")
BEST_CODE=$(python -c "import json; print(json.load(open('$BEST_FILE'))['best_code'])")
BEST_DELTA=$(python -c "import json; print(json.load(open('$BEST_FILE'))['best_delta_test_b'])")
log "lambda* identified: $BEST_LAMBDA (code=$BEST_CODE, delta_test_b=$BEST_DELTA)"

# Step 1.3: launch F4 (seed=42) on cuda:0 at lambda*
log "Step 1.3: launching F4 seed=42 at lambda=$BEST_LAMBDA on cuda:0"
bash scripts/launch_session9_step1_bisection.sh 0 F4 "$BEST_LAMBDA" 42
RC=$?
log "F4 finished with exit=$RC"

# Step 1.4: launch F5 (seed=123) on cuda:0 at lambda*
log "Step 1.4: launching F5 seed=123 at lambda=$BEST_LAMBDA on cuda:0"
bash scripts/launch_session9_step1_bisection.sh 0 F5 "$BEST_LAMBDA" 123
RC=$?
log "F5 finished with exit=$RC"

# Step 1.5: re-run analysis with seed-variance evaluation.
log "Step 1.5: running bisection analysis with seed-variance"
VARIANCE_LOG="$OUTROOT/bisection_analysis_seed_variance.log"
python scripts/session9_bisection_analysis.py \
    --gpu "$ANALYSIS_GPU" --iters 20000 --seed-variance \
    --output-dir "$OUTROOT" \
    >"$VARIANCE_LOG" 2>&1
RC=$?
log "Seed-variance analysis exit=$RC"

log "Session 9 Step 1 orchestrator finished"
exit 0
