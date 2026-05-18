#!/usr/bin/env bash
# Session 7 launcher: three production-scale 20k-iter runs on the full v1.2
# train partition (41 cases, 138 train encounters per D35), dual-card.
#
# R1: PLDM + OBS + BN, --gpu 0 (cuda:2 in unfiltered torch view)
# R2: PLDM only,  BN, --gpu 1 (cuda:3) -- concurrent with R1
# R3: SIGReg + OBS + BN, --gpu 1 -- sequential after R2 completes
#
# Total wall clock ~10 hours (5h R1 || R2 then 5h R3). Test B is the headline
# evaluation metric per the SESSION7_FULL_SCALE_HONEST.md plan.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
WANDB_MODE="${WANDB_MODE:-offline}"

OUTROOT="outputs/runs/session7"
mkdir -p "$OUTROOT"
SUMMARY="$OUTROOT/launch.log"
: > "$SUMMARY"

log() {
    echo "[$(date -Iseconds)] $*" | tee -a "$SUMMARY"
}

log "Session 7 launch starting"
log "  repo:           $REPO_ROOT"
log "  PREVENT_ROOT:   $PREVENT_ROOT"
log "  WANDB_PROJECT:  $WANDB_PROJECT"
log "  WANDB_MODE:     $WANDB_MODE"
log "  git HEAD:       $(git rev-parse HEAD)"
log "  branch:         $(git rev-parse --abbrev-ref HEAD)"

# Pre-flight: GPU visibility (the four other checks ran ahead of this script;
# this one is cheap enough to repeat as a final guard).
log "Pre-flight: GPU enumeration"
python -c "
import torch
n = torch.cuda.device_count()
rtx = [i for i in range(n) if 'RTX' in torch.cuda.get_device_name(i) and '6000' in torch.cuda.get_device_name(i)]
print(f'  device_count={n}, RTX 6000 indices={rtx}')
assert len(rtx) >= 2, f'need >=2 RTX 6000 cards; saw {len(rtx)}'
" 2>&1 | tee -a "$SUMMARY"

declare -A R_PID
declare -A R_LOG

launch_bg() {
    local name=$1
    local outdir="$OUTROOT/$name"
    mkdir -p "$outdir"
    local logfile="$outdir/train.log"
    R_LOG[$name]="$logfile"
    shift
    log "BEGIN $name: $*"
    log "    log -> $logfile"
    nohup python -m "$@" >"$logfile" 2>&1 &
    R_PID[$name]=$!
    disown
    log "    pid=${R_PID[$name]}"
}

wait_for() {
    local name=$1
    local pid=${R_PID[$name]}
    log "Waiting for $name (pid=$pid) to complete"
    if wait "$pid"; then
        log "END   $name pid=$pid exit=0"
    else
        local rc=$?
        log "END   $name pid=$pid exit=$rc (FAILED)"
        log "Last 20 lines of ${R_LOG[$name]}:"
        tail -n 20 "${R_LOG[$name]}" | sed 's/^/    /' | tee -a "$SUMMARY"
    fi
}

# R1: PLDM + OBS + BN on the first RTX 6000
launch_bg run_r1_pldm_obs_bn \
    src.training.train_baseline \
    --baseline pldm \
    --gpu 0 \
    --partition v1 \
    --all-train \
    --max-iters 20000 \
    --seed 0 \
    --observable-head cl_future \
    --observable-head-weight 0.01 \
    --observable-head-deltas 8 16 24 \
    --projection-norm batchnorm \
    --log-every 50 \
    --diagnostic-every 500 \
    --checkpoint-every 2000 \
    --wandb-mode "$WANDB_MODE" \
    --output-dir "$OUTROOT/run_r1_pldm_obs_bn" \
    --tag-suffix run_r1_pldm_obs_bn_seed0_full

# R2: PLDM only (no OBS) on the second RTX 6000, concurrent with R1
launch_bg run_r2_pldm_only_bn \
    src.training.train_baseline \
    --baseline pldm \
    --gpu 1 \
    --partition v1 \
    --all-train \
    --max-iters 20000 \
    --seed 0 \
    --observable-head none \
    --projection-norm batchnorm \
    --log-every 50 \
    --diagnostic-every 500 \
    --checkpoint-every 2000 \
    --wandb-mode "$WANDB_MODE" \
    --output-dir "$OUTROOT/run_r2_pldm_only_bn" \
    --tag-suffix run_r2_pldm_only_bn_seed0_full

# Wait for R2 to free cuda:3 before launching R3
wait_for run_r2_pldm_only_bn

# R3: SIGReg + OBS on the second RTX 6000, sequential after R2
launch_bg run_r3_sigreg_obs_bn \
    src.training.train_jepa \
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
    --output-dir "$OUTROOT/run_r3_sigreg_obs_bn" \
    --tag-suffix run_r3_sigreg_obs_bn_seed0_full

# Wait for R1 (may still be running) and R3
wait_for run_r1_pldm_obs_bn
wait_for run_r3_sigreg_obs_bn

log "Session 7 launcher finished"
log "Per-run final checkpoint expected at:"
for name in run_r1_pldm_obs_bn run_r2_pldm_only_bn run_r3_sigreg_obs_bn; do
    log "  $OUTROOT/$name/checkpoint_iter020000.pt"
done
