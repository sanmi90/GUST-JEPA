#!/usr/bin/env bash
# Recovery launcher for Session 7 R3.
#
# The original launch_session7.sh had a `nohup ... & ; disown` bug that
# caused all three runs to start concurrently (R2 and R3 collided on
# cuda:3). We killed R3 and left R1+R2 running. This script polls for
# R2 to finish (its iter-20000 checkpoint appears and its python process
# exits) and then launches R3 cleanly on the second RTX 6000.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
WANDB_MODE="${WANDB_MODE:-offline}"

OUTROOT="outputs/runs/session7"
SUMMARY="$OUTROOT/launch.log"

R2_CKPT="$OUTROOT/run_r2_pldm_only_bn/checkpoint_iter020000.pt"
R3_OUTDIR="$OUTROOT/run_r3_sigreg_obs_bn"

log() {
    echo "[$(date -Iseconds)] $*" | tee -a "$SUMMARY"
}

log "Recovery R3 launcher waiting for R2 to finish"
log "  Polling for $R2_CKPT and R2 python process exit (poll every 60s)"
while true; do
    if [ -f "$R2_CKPT" ]; then
        if ! pgrep -f "outputs/runs/session7/run_r2_pldm_only_bn" >/dev/null; then
            break
        fi
    fi
    sleep 60
done
log "R2 final checkpoint present and R2 process exited. Claiming cuda:3 for R3."

mkdir -p "$R3_OUTDIR"
R3_LOG="$R3_OUTDIR/train.log"
log "BEGIN run_r3_sigreg_obs_bn (recovery)"
log "  log -> $R3_LOG"

python -m src.training.train_jepa \
    --gpu 1 \
    --partition v1 \
    --all-train \
    --max-iters 20000 \
    --seed 0 \
    --observable-head cl_future \
    --observable-head-weight 0.01 \
    --observable-head-deltas 8 16 24 \
    --projection-norm batchnorm \
    --anticollapse sigreg \
    --lambda-sigreg 0.1 \
    --log-every 50 \
    --diagnostic-every 500 \
    --checkpoint-every 2000 \
    --wandb-mode "$WANDB_MODE" \
    --output-dir "$R3_OUTDIR" \
    --tag-suffix run_r3_sigreg_obs_bn_seed0_full \
    >"$R3_LOG" 2>&1
R3_RC=$?
log "END run_r3_sigreg_obs_bn exit=$R3_RC"
if [ $R3_RC -ne 0 ]; then
    log "--- last 30 lines of $R3_LOG ---"
    tail -n 30 "$R3_LOG" | tee -a "$SUMMARY"
fi
exit "$R3_RC"
