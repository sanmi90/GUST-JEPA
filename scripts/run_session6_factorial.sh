#!/usr/bin/env bash
# Session 6 Step 3: run all five factorial single-axis variants.
#
# Each variant trains for 5000 iterations on the RTX 6000. Sequential
# execution (single GPU); estimated total wall clock 2.5 to 3 hours.
#
# Output layout: outputs/runs/session6/run_f_{l,cd,nc,s,obs}/...
# Per-variant stdout/stderr is captured to <output-dir>/train.log.
# A top-level summary at outputs/runs/session6/_session6_summary.log
# records start/end timestamps and exit codes for every variant.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
WANDB_MODE="${WANDB_MODE:-offline}"

OUTROOT="outputs/runs/session6"
mkdir -p "$OUTROOT"
SUMMARY="$OUTROOT/_session6_summary.log"
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
        echo "BEGIN $name at $t_start"
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

# F-L: longer sub-trajectory (L=64, H_roll=16)
run_variant run_f_l \
    --cases-from configs/cases/smoke_5cases.yaml \
    --sub-trajectory-length 64 \
    --rollout-horizon 16 \
    --tag-suffix run_f_l_seed0_L64

# F-CD: per-batch c-dropout 0.5
run_variant run_f_cd \
    --cases-from configs/cases/smoke_5cases.yaml \
    --c-dropout-prob 0.5 \
    --tag-suffix run_f_cd_seed0_p0p5

# F-NC: predictor with no c at all
run_variant run_f_nc \
    --cases-from configs/cases/smoke_5cases.yaml \
    --predictor-cond-dim 0 \
    --tag-suffix run_f_nc_seed0_cond0

# F-S: scale up to 24 cases
run_variant run_f_s \
    --cases-from configs/cases/smoke_24cases.yaml \
    --tag-suffix run_f_s_seed0_24cases

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
    echo "Session 6 factorial done at $(date -Iseconds)"
    for key in run_f_l run_f_cd run_f_nc run_f_s run_f_obs; do
        echo "    $key  exit=${EXITS[$key]:-NA}"
    done
} | tee -a "$SUMMARY"

# Top-level script exits 0 only if every variant exited 0.
overall=0
for key in run_f_l run_f_cd run_f_nc run_f_s run_f_obs; do
    if [ "${EXITS[$key]:-1}" -ne 0 ]; then
        overall=1
    fi
done
exit "$overall"
