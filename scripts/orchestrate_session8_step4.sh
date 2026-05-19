#!/usr/bin/env bash
# Session 8 orchestrator for Steps 3 -> 4 chain.
#
# Waits for the named training run to write its iter-20000 checkpoint
# AND for its python process to exit. Then optionally runs a pass-criterion
# check (Step 3 R3-seed=42 only). Then sequentially launches a chain of
# Step 4 grid runs on the same card.
#
# Args:
#   $1: card index (0 or 1)
#   $2: name of the precursor run (relative dir under outputs/runs/session8/,
#       or "outputs/runs/session8/run_r3_seed42" for Step 3 on cuda:0)
#   $3: comma-separated run codes for the chain (e.g. "E1,E2,E3,E4")
#   $4 (optional): "verify" -- if present, run session8_eval_r3_seed42.py
#       on the precursor checkpoint and exit if the pass criterion fails.
#
# Example:
#   scripts/orchestrate_session8_step4.sh 0 run_r3_seed42 E1,E2,E3,E4 verify
#   scripts/orchestrate_session8_step4.sh 1 run_e6_eta0p010_lam1p00 E7,E8,E9,E10

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

CARD="${1:?usage: $0 <card> <precursor_run_dir_or_name> <codes> [verify]}"
PRE="${2:?usage: $0 <card> <precursor_run_dir_or_name> <codes> [verify]}"
CODES="${3:?usage: $0 <card> <precursor_run_dir_or_name> <codes> [verify]}"
VERIFY="${4:-no}"

OUTROOT="outputs/runs/session8"
PRE_DIR="$OUTROOT/$PRE"
if [ ! -d "$PRE_DIR" ]; then
    # Try as a relative path from repo root directly (for Session 7 dirs etc.)
    if [ -d "$PRE" ]; then
        PRE_DIR="$PRE"
    fi
fi
PRE_CKPT="$PRE_DIR/checkpoint_iter020000.pt"
SUMMARY="$OUTROOT/orchestrate_card${CARD}.log"
: > "$SUMMARY"

log() {
    echo "[$(date -Iseconds)] $*" | tee -a "$SUMMARY"
}

log "Session 8 orchestrator card $CARD"
log "  precursor: $PRE_DIR"
log "  codes:     $CODES"
log "  verify:    $VERIFY"

# Wait for the precursor's iter-20000 checkpoint and python process exit.
log "Polling for $PRE_CKPT and precursor python process exit (60s interval)"
while true; do
    if [ -f "$PRE_CKPT" ]; then
        # The precursor process is identified by its output-dir path on argv.
        # The exact match accommodates both 'session8/run_*' and 'session7/run_*'.
        if ! pgrep -f "$PRE_DIR" >/dev/null; then
            break
        fi
    fi
    sleep 60
done
log "Precursor finished. checkpoint present at $PRE_CKPT"

# Optional pass-criterion check
if [ "$VERIFY" = "verify" ]; then
    log "Running session8_eval_r3_seed42.py (pass bracket [+0.05, +0.25])"
    EVAL_LOG="$OUTROOT/eval_r3_seed42.log"
    python scripts/session8_eval_r3_seed42.py \
        --checkpoint "$PRE_CKPT" \
        --gpu "$CARD" \
        --bracket-min 0.05 --bracket-max 0.25 \
        --output "$OUTROOT/r3_seed42_eval.json" \
        >"$EVAL_LOG" 2>&1
    RC=$?
    log "eval exit=$RC"
    if [ $RC -ne 0 ]; then
        log "Pass criterion FAILED (or other error). NOT launching chain on card $CARD."
        log "--- last 30 lines of eval log ---"
        tail -n 30 "$EVAL_LOG" | tee -a "$SUMMARY"
        exit $RC
    fi
    log "Pass criterion PASSED. Continuing to chain."
fi

# Run the chain (delegate to launch_session8_step4_grid.sh).
bash scripts/launch_session8_step4_grid.sh "$CARD" "$CODES"
RC=$?
log "Chain finished, exit=$RC"
exit $RC
