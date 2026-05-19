#!/usr/bin/env bash
# Session 9 Step 1: lambda bisection at the production point.
#
# Bisection lambda values at (d=32, eta=0.01, OBS=cl_future, BN, SIGReg, seed=0):
#   F1  lambda = 0.001
#   F2  lambda = 0.003
#   F3  lambda = 0.03
# Anchors already on disk (NOT re-trained):
#   E4  lambda = 0.01  -> outputs/runs/session8/run_e4_eta0p010_lam0p01
#   E5  lambda = 0.1   -> outputs/runs/session7/run_r3_sigreg_obs_bn
#
# After F1, F2, F3 + the two anchors are available, scripts/session9_bisection_analysis.py
# identifies lambda* = argmax delta_test_b. Then this same launcher accepts the
# seed-variance codes F4 (seed=42) and F5 (seed=123) at the lambda found.
#
# Each run is 20k iters at ~1.5h on RTX 6000 Blackwell per D49.
#
# Two args:
#   $1: card index (0 or 1)
#   $2: comma-separated run codes (e.g., "F1,F2,F3" or "F4" or "F5")
#   $3 (optional): lambda override (only used for F4/F5 once lambda* is known)
#   $4 (optional): seed override (only used for F4/F5; default seed depends on code)
#
# Example:
#   scripts/launch_session9_step1_bisection.sh 0 F1,F2,F3
#   scripts/launch_session9_step1_bisection.sh 0 F4 0.01 42
#   scripts/launch_session9_step1_bisection.sh 1 F5 0.01 123

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
WANDB_MODE="${WANDB_MODE:-offline}"

CARD="${1:?usage: $0 <card 0|1> <run codes> [lambda_override] [seed_override]}"
RUN_CODES="${2:?usage: $0 <card 0|1> <run codes> [lambda_override] [seed_override]}"
LAMBDA_OVERRIDE="${3:-}"
SEED_OVERRIDE="${4:-}"

OUTROOT="outputs/runs/session9"
mkdir -p "$OUTROOT"
SUMMARY="$OUTROOT/launch_step1_card${CARD}.log"
: > "$SUMMARY"

log() {
    echo "[$(date -Iseconds)] $*" | tee -a "$SUMMARY"
}

log "Session 9 Step 1 bisection launcher starting on cuda:${CARD}"
log "  Runs: $RUN_CODES"
log "  lambda override: '${LAMBDA_OVERRIDE}' (empty = use code default)"
log "  seed override:   '${SEED_OVERRIDE}' (empty = use code default)"
log "  repo: $REPO_ROOT"
log "  git HEAD: $(git rev-parse HEAD)"

# Code -> (default lambda, default seed). Production (d=32, eta=0.01,
# OBS=cl_future at eta=0.01, BN, SIGReg) is fixed.
declare -A LAM SEED NAME
LAM[F1]=0.001;   SEED[F1]=0;   NAME[F1]="run_f1_lam0p001_seed0"
LAM[F2]=0.003;   SEED[F2]=0;   NAME[F2]="run_f2_lam0p003_seed0"
LAM[F3]=0.03;    SEED[F3]=0;   NAME[F3]="run_f3_lam0p030_seed0"
# F4 + F5: lambda* and seed get filled in at launch time.
LAM[F4]="${LAMBDA_OVERRIDE:-0.01}";  SEED[F4]="${SEED_OVERRIDE:-42}"
LAM[F5]="${LAMBDA_OVERRIDE:-0.01}";  SEED[F5]="${SEED_OVERRIDE:-123}"
# NAME[F4]/[F5] derived from lambda + seed below.

name_of() {
    local code="$1"
    if [ "$code" = "F4" ] || [ "$code" = "F5" ]; then
        local lam="${LAM[$code]}"
        local seed="${SEED[$code]}"
        local lam_tag
        lam_tag=$(printf '%s' "$lam" | tr '.' 'p')
        echo "run_${code,,}_lam${lam_tag}_seed${seed}"
    else
        echo "${NAME[$code]}"
    fi
}

launch_sigreg_obs() {
    local code=$1
    local lam="${LAM[$code]}"
    local seed="${SEED[$code]}"
    local outdir="$OUTROOT/$(name_of "$code")"
    mkdir -p "$outdir"
    local logfile="$outdir/train.log"
    log "BEGIN $code SIGReg+OBS+BN d=32 eta=0.01 lambda=$lam seed=$seed -> $outdir"
    python -m src.training.train_jepa \
        --gpu "$CARD" \
        --partition v1 --all-train --max-iters 20000 --seed "$seed" \
        --latent-dim 32 \
        --observable-head cl_future --observable-head-weight 0.01 \
        --observable-head-deltas 8 16 24 \
        --projection-norm batchnorm --anticollapse sigreg \
        --lambda-sigreg "$lam" \
        --diagnostic-every 500 --checkpoint-every 2000 --log-every 50 \
        --output-dir "$outdir" \
        --wandb-mode "$WANDB_MODE" \
        --tag-suffix "$(name_of "$code")_bisection" \
        >"$logfile" 2>&1
    local rc=$?
    log "END   $code exit=$rc"
    if [ $rc -ne 0 ]; then
        log "--- last 30 lines of $logfile ---"
        tail -n 30 "$logfile" | tee -a "$SUMMARY"
    fi
    return $rc
}

IFS=',' read -r -a CODES <<< "$RUN_CODES"
overall_rc=0
for code in "${CODES[@]}"; do
    if [ -z "${LAM[$code]:-}" ]; then
        log "ERROR: unknown run code '$code'"
        overall_rc=2
        continue
    fi
    if ! launch_sigreg_obs "$code"; then
        log "$code failed; continuing to next run"
        overall_rc=1
    fi
done

log "Step 1 launcher (card $CARD) finished with overall exit=$overall_rc"
exit $overall_rc
