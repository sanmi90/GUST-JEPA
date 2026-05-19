#!/usr/bin/env bash
# Session 9 Step 3 orchestrator: cuda:1 chain for the ablation thin cut.
#
# Sequence on cuda:1:
#   1) A2 VICReg-only (1.5h, no lambda* dependency).
#   2) (Optional) A10 Solera-Rico beta-VAE (1.5h, no lambda* dep) -- only
#      if scripts/launch_session9_step3_ablations.sh A10 is callable
#      (i.e., src/baselines/solera_rico.py + train_baseline support landed).
#   3) (Optional) A11 Fukami observable-augmented AE (1.5h, no lambda* dep).
#   4) Wait for outputs/runs/session9/best_lambda_star.json (written by the
#      cuda:0 orchestrator after F3 completes the seed=0 set + analysis).
#   5) A7 no-scheduled-sampling at lambda* (1.5h).
#
# Codes to run are provided via the CODES env variable (default "A2,A7").
# Optional ablations A10/A11 are EXCLUDED by default to keep the wall-clock
# budget honest -- they require new src/baselines modules and a
# train_baseline dispatch; defer to Session 10 unless explicitly added.
#
# Usage:
#   scripts/orchestrate_session9_step3.sh
#   CODES="A2,A7" scripts/orchestrate_session9_step3.sh
#   CODES="A2,A10,A11,A7" scripts/orchestrate_session9_step3.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

CODES="${CODES:-A2,A7}"
OUTROOT="outputs/runs/session9"
mkdir -p "$OUTROOT"
SUMMARY="$OUTROOT/orchestrate_step3.log"
: > "$SUMMARY"

log() {
    echo "[$(date -Iseconds)] $*" | tee -a "$SUMMARY"
}

log "Session 9 Step 3 orchestrator starting on cuda:1"
log "  CODES: $CODES"
log "  repo: $REPO_ROOT"
log "  git HEAD: $(git rev-parse HEAD)"

# Split into pre-lambda (A2, A10, A11) and post-lambda (A7) phases.
PRE_LAMBDA=""
POST_LAMBDA=""
IFS=',' read -r -a CODE_LIST <<< "$CODES"
for code in "${CODE_LIST[@]}"; do
    case "$code" in
        A2|A10|A11)
            PRE_LAMBDA="${PRE_LAMBDA:+$PRE_LAMBDA,}$code"
            ;;
        A7)
            POST_LAMBDA="${POST_LAMBDA:+$POST_LAMBDA,}$code"
            ;;
        *)
            log "ERROR: unknown ablation code '$code'"
            ;;
    esac
done

# Phase 1: pre-lambda ablations (run immediately, lambda* not needed).
if [ -n "$PRE_LAMBDA" ]; then
    log "Phase 1: launching pre-lambda ablations $PRE_LAMBDA on cuda:1"
    bash scripts/launch_session9_step3_ablations.sh 1 "$PRE_LAMBDA"
    RC=$?
    log "Phase 1 finished with exit=$RC"
fi

# Phase 2: wait for lambda* to be identified, then run A7.
if [ -n "$POST_LAMBDA" ]; then
    BEST_FILE="$OUTROOT/best_lambda_star.json"
    log "Phase 2: waiting for $BEST_FILE (cuda:0 orchestrator writes it after F3 + analysis)"
    while [ ! -f "$BEST_FILE" ]; do
        sleep 60
    done
    BEST_LAMBDA=$(python -c "import json; print(json.load(open('$BEST_FILE'))['best_lambda'])")
    log "lambda* observed: $BEST_LAMBDA"
    log "Phase 2: launching post-lambda ablations $POST_LAMBDA at lambda*=$BEST_LAMBDA on cuda:1"
    bash scripts/launch_session9_step3_ablations.sh 1 "$POST_LAMBDA" "$BEST_LAMBDA"
    RC=$?
    log "Phase 2 finished with exit=$RC"
fi

log "Session 9 Step 3 orchestrator finished"
exit 0
