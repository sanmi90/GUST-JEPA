#!/usr/bin/env bash
# Session 8 Step 5: latent-dimension sweep at the best (eta*, lambda*) point.
#
# Two new runs at d=8 (cuda:0) and d=16 (cuda:1) sharing the best
# (eta*, lambda*) from Step 4. d=32 reuses the Step 4 best-grid result.
# Wall-clock ~1.5h with two-card parallelism.
#
# Args:
#   $1: eta_star (e.g. 0.01)
#   $2: lambda_star (e.g. 0.1)
#
# Example:
#   scripts/launch_session8_step5_dsweep.sh 0.01 0.1

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
WANDB_MODE="${WANDB_MODE:-offline}"

ETA="${1:?usage: $0 <eta_star> <lambda_star>}"
LAM="${2:?usage: $0 <eta_star> <lambda_star>}"

OUTROOT="outputs/runs/session8"
mkdir -p "$OUTROOT"
SUMMARY="$OUTROOT/launch_step5_dsweep.log"
: > "$SUMMARY"

log() {
    echo "[$(date -Iseconds)] $*" | tee -a "$SUMMARY"
}

log "Session 8 Step 5 d-sweep launcher starting"
log "  eta*=$ETA lambda*=$LAM"
log "  repo: $REPO_ROOT"
log "  git HEAD: $(git rev-parse HEAD)"

launch_d() {
    local card=$1
    local d=$2
    local outdir="$OUTROOT/run_d${d}_best"
    mkdir -p "$outdir"
    local logfile="$outdir/train.log"
    log "BEGIN d=$d on cuda:$card eta=$ETA lambda=$LAM -> $outdir"
    python -m src.training.train_jepa \
        --gpu "$card" \
        --partition v1 --all-train --max-iters 20000 --seed 0 \
        --latent-dim "$d" \
        --observable-head cl_future --observable-head-weight "$ETA" \
        --observable-head-deltas 8 16 24 \
        --projection-norm batchnorm --anticollapse sigreg \
        --lambda-sigreg "$LAM" \
        --diagnostic-every 500 --checkpoint-every 2000 --log-every 50 \
        --output-dir "$outdir" \
        --wandb-mode "$WANDB_MODE" \
        --tag-suffix "run_d${d}_best_eta${ETA}_lam${LAM}" \
        >"$logfile" 2>&1 &
    local pid=$!
    log "  pid=$pid d=$d"
    echo "$pid"
}

PID_D8=$(launch_d 0 8)
PID_D16=$(launch_d 1 16)

log "Waiting for d=8 (pid=$PID_D8) and d=16 (pid=$PID_D16) to finish"

rc_d8=0; rc_d16=0
wait "$PID_D8" || rc_d8=$?
log "d=8 exit=$rc_d8"
wait "$PID_D16" || rc_d16=$?
log "d=16 exit=$rc_d16"

if [ $rc_d8 -ne 0 ] || [ $rc_d16 -ne 0 ]; then
    log "Step 5 had failures: d8=$rc_d8 d16=$rc_d16"
    exit 1
fi
log "Step 5 finished successfully"
exit 0
