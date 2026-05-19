#!/usr/bin/env bash
# Session 8 Step 4: 2D (eta x lambda) SIGReg grid + E10 PLDM reference.
#
# cuda:0 sequence: E1, E2, E3, E4 (4 SIGReg runs, ~6h total)
# cuda:1 sequence: E6, E7, E8, E9, E10 (4 SIGReg + 1 PLDM, ~7.5h)
# E5 reuses the Session 7 R3 anchor (already trained).
#
# Each run is 20k iters at 1.5h on RTX 6000 Blackwell per D49.
# Runs after Step 3 (R3-seed=42) has passed its [+0.05, +0.25] criterion.
#
# Two args:
#   $1: card index (0 or 1)
#   $2: comma-separated run codes (e.g., "E1,E2,E3,E4")
#
# Example:
#   scripts/launch_session8_step4_grid.sh 0 E1,E2,E3,E4
#   scripts/launch_session8_step4_grid.sh 1 E6,E7,E8,E9,E10

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
WANDB_MODE="${WANDB_MODE:-offline}"

CARD="${1:?usage: $0 <card 0|1> <run codes>}"
RUN_CODES="${2:?usage: $0 <card 0|1> <run codes>}"

OUTROOT="outputs/runs/session8"
mkdir -p "$OUTROOT"
SUMMARY="$OUTROOT/launch_step4_card${CARD}.log"
: > "$SUMMARY"

log() {
    echo "[$(date -Iseconds)] $*" | tee -a "$SUMMARY"
}

log "Session 8 Step 4 grid launcher starting on cuda:${CARD}"
log "  Runs: $RUN_CODES"
log "  repo: $REPO_ROOT"
log "  git HEAD: $(git rev-parse HEAD)"

# Grid mapping: SIGReg+OBS runs. eta = observable-head-weight; lambda = lambda-sigreg.
declare -A ETA LAM TYPE
# (eta=0.001 row)
ETA[E1]=0.001; LAM[E1]=0.01;  TYPE[E1]=sigreg
ETA[E2]=0.001; LAM[E2]=0.1;   TYPE[E2]=sigreg
ETA[E3]=0.001; LAM[E3]=1.0;   TYPE[E3]=sigreg
# (eta=0.01 row; E5 is the Session 7 R3 anchor, not re-run here)
ETA[E4]=0.01;  LAM[E4]=0.01;  TYPE[E4]=sigreg
ETA[E6]=0.01;  LAM[E6]=1.0;   TYPE[E6]=sigreg
# (eta=0.1 row)
ETA[E7]=0.1;   LAM[E7]=0.01;  TYPE[E7]=sigreg
ETA[E8]=0.1;   LAM[E8]=0.1;   TYPE[E8]=sigreg
ETA[E9]=0.1;   LAM[E9]=1.0;   TYPE[E9]=sigreg
# E10: PLDM+OBS+BN with paper-tuned Two-Rooms weights (arXiv:2502.14819 Appendix J.2)
TYPE[E10]=pldm

NAME_E1="run_e1_eta0p001_lam0p01"
NAME_E2="run_e2_eta0p001_lam0p10"
NAME_E3="run_e3_eta0p001_lam1p00"
NAME_E4="run_e4_eta0p010_lam0p01"
NAME_E6="run_e6_eta0p010_lam1p00"
NAME_E7="run_e7_eta0p100_lam0p01"
NAME_E8="run_e8_eta0p100_lam0p10"
NAME_E9="run_e9_eta0p100_lam1p00"
NAME_E10="run_e10_pldm_paper_tuned"

name_of() {
    local code="$1"
    case "$code" in
        E1)  echo "$NAME_E1" ;;
        E2)  echo "$NAME_E2" ;;
        E3)  echo "$NAME_E3" ;;
        E4)  echo "$NAME_E4" ;;
        E6)  echo "$NAME_E6" ;;
        E7)  echo "$NAME_E7" ;;
        E8)  echo "$NAME_E8" ;;
        E9)  echo "$NAME_E9" ;;
        E10) echo "$NAME_E10" ;;
        *)   echo ""; return 1 ;;
    esac
}

launch_sigreg() {
    local code=$1
    local outdir="$OUTROOT/$(name_of "$code")"
    mkdir -p "$outdir"
    local logfile="$outdir/train.log"
    local eta="${ETA[$code]}"
    local lam="${LAM[$code]}"
    log "BEGIN $code SIGReg+OBS+BN eta=$eta lambda=$lam -> $outdir"
    python -m src.training.train_jepa \
        --gpu "$CARD" \
        --partition v1 --all-train --max-iters 20000 --seed 0 \
        --observable-head cl_future --observable-head-weight "$eta" \
        --observable-head-deltas 8 16 24 \
        --projection-norm batchnorm --anticollapse sigreg \
        --lambda-sigreg "$lam" \
        --diagnostic-every 500 --checkpoint-every 2000 --log-every 50 \
        --output-dir "$outdir" \
        --wandb-mode "$WANDB_MODE" \
        --tag-suffix "$(name_of "$code")_step4" \
        >"$logfile" 2>&1
    local rc=$?
    log "END   $code exit=$rc"
    if [ $rc -ne 0 ]; then
        log "--- last 30 lines of $logfile ---"
        tail -n 30 "$logfile" | tee -a "$SUMMARY"
    fi
    return $rc
}

launch_pldm_e10() {
    local outdir="$OUTROOT/$NAME_E10"
    mkdir -p "$outdir"
    local logfile="$outdir/train.log"
    log "BEGIN E10 PLDM+OBS+BN paper-tuned (alpha=4.0, beta=6.9, delta=0.75, omega=0.0) -> $outdir"
    python -m src.training.train_baseline \
        --baseline pldm \
        --gpu "$CARD" \
        --partition v1 --all-train --max-iters 20000 --seed 0 \
        --lambda-var 4.0 --lambda-cov 6.9 --lambda-time-sim 0.75 --lambda-idm 0.0 \
        --observable-head cl_future --observable-head-weight 0.01 \
        --observable-head-deltas 8 16 24 \
        --projection-norm batchnorm \
        --diagnostic-every 500 --checkpoint-every 2000 --log-every 50 \
        --output-dir "$outdir" \
        --wandb-mode "$WANDB_MODE" \
        --tag-suffix "${NAME_E10}_step4" \
        >"$logfile" 2>&1
    local rc=$?
    log "END   E10 exit=$rc"
    if [ $rc -ne 0 ]; then
        log "--- last 30 lines of $logfile ---"
        tail -n 30 "$logfile" | tee -a "$SUMMARY"
    fi
    return $rc
}

# Process each run code in order.
IFS=',' read -r -a CODES <<< "$RUN_CODES"
overall_rc=0
for code in "${CODES[@]}"; do
    if [ -z "${TYPE[$code]:-}" ]; then
        log "ERROR: unknown run code '$code'"
        overall_rc=2
        continue
    fi
    if [ "${TYPE[$code]}" = "sigreg" ]; then
        if ! launch_sigreg "$code"; then
            log "$code failed; continuing to next run"
            overall_rc=1
        fi
    elif [ "${TYPE[$code]}" = "pldm" ]; then
        if ! launch_pldm_e10; then
            log "E10 failed; continuing"
            overall_rc=1
        fi
    fi
done

log "Step 4 launcher (card $CARD) finished with overall exit=$overall_rc"
exit $overall_rc
