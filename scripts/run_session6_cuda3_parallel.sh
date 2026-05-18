#!/usr/bin/env bash
# Parallel chain on the second RTX 6000. Sets CUDA_VISIBLE_DEVICES so torch
# sees only torch index 3 (the second RTX 6000); require_rtx6000() then
# returns cuda:0 in the filtered view, which is the second physical card.
#
# Runs F-NC (cond_dim=0 path; exotic, so it goes first to surface any
# failure quickly) then F-OBS (observable head, eta=0.01).
#
# Companion to scripts/run_session6_cuda2_remainder.sh which finishes F-CD
# and F-S on the first RTX 6000 after F-L completes.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
WANDB_MODE="${WANDB_MODE:-offline}"
export CUDA_VISIBLE_DEVICES=3  # only the second RTX 6000 is visible to torch

OUTROOT="outputs/runs/session6"
SUMMARY="$OUTROOT/_session6_summary_cuda3.log"
: > "$SUMMARY"

declare -A EXITS

run_variant() {
    local name=$1; shift
    local outdir="$OUTROOT/$name"
    mkdir -p "$outdir"
    local logfile="$outdir/train.log"
    local t_start
    t_start=$(date -Iseconds)
    {
        echo
        echo "=========================================================="
        echo "BEGIN $name at $t_start (cuda:3 = torch cuda:0 under CUDA_VISIBLE_DEVICES=3)"
        echo "args: $*"
        echo "=========================================================="
    } | tee -a "$SUMMARY"
    set +e
    python -m src.training.train_jepa \
        --partition v1 \
        --max-iters 5000 \
        --seed 0 \
        --log-every 25 \
        --diagnostic-every 250 \
        --checkpoint-every 1000 \
        --wandb-mode "$WANDB_MODE" \
        --output-dir "$outdir" \
        "$@" \
        >"$logfile" 2>&1
    local rc=$?
    set -e
    EXITS[$name]=$rc
    local t_end
    t_end=$(date -Iseconds)
    {
        echo "END   $name at $t_end   exit=$rc"
        if [ $rc -ne 0 ]; then
            echo "--- last 20 lines of $logfile ---"
            tail -n 20 "$logfile"
        else
            tail -n 8 "$logfile" | sed 's/^/    /'
        fi
    } | tee -a "$SUMMARY"
}

# F-NC: predictor cond_dim = 0
run_variant run_f_nc \
    --cases-from configs/cases/smoke_5cases.yaml \
    --predictor-cond-dim 0 \
    --tag-suffix run_f_nc_seed0_cond0

# F-OBS: observable head with weight 0.01
run_variant run_f_obs \
    --cases-from configs/cases/smoke_5cases.yaml \
    --observable-head cl_future \
    --observable-head-weight 0.01 \
    --observable-head-deltas 8 16 24 \
    --tag-suffix run_f_obs_seed0_eta0p01

{
    echo
    echo "=========================================================="
    echo "cuda:3 chain done at $(date -Iseconds)"
    for key in run_f_nc run_f_obs; do
        echo "    $key  exit=${EXITS[$key]:-NA}"
    done
} | tee -a "$SUMMARY"

overall=0
for key in run_f_nc run_f_obs; do
    if [ "${EXITS[$key]:-1}" -ne 0 ]; then overall=1; fi
done
exit "$overall"
