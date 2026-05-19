#!/usr/bin/env bash
# Session 9 Step 3: Section 7 ablation thin cut.
#
# Four ablations on the production (d=32, eta=0.01) config:
#   A2  VICReg-only at the VICReg-canonical (lambda_var=25, lambda_cov=25,
#       nu=1) weights. Independent of lambda* (uses VICReg's own weights).
#   A7  No scheduled sampling (H_roll = T = 32) at lambda*. Needs lambda*.
#   A10 Solera-Rico beta-VAE + transformer at d=32. New baseline module.
#   A11 Fukami observable-augmented autoencoder at d=32. New baseline module.
#
# Each run is 20k iters at ~1.5h on RTX 6000 Blackwell.
#
# Two args:
#   $1: card index (0 or 1)
#   $2: comma-separated ablation codes (e.g., "A2,A7" or "A10,A11")
#   $3 (optional): lambda* (only used for A7 once Step 1 finds it; required
#                  for A7, ignored for A2/A10/A11)
#
# Example:
#   scripts/launch_session9_step3_ablations.sh 1 A2,A7 0.01
#   scripts/launch_session9_step3_ablations.sh 1 A10,A11

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
WANDB_MODE="${WANDB_MODE:-offline}"

CARD="${1:?usage: $0 <card 0|1> <ablation codes> [lambda_star]}"
RUN_CODES="${2:?usage: $0 <card 0|1> <ablation codes> [lambda_star]}"
LAMBDA_STAR="${3:-0.01}"

OUTROOT="outputs/runs/session9"
mkdir -p "$OUTROOT"
SUMMARY="$OUTROOT/launch_step3_card${CARD}.log"
: > "$SUMMARY"

log() {
    echo "[$(date -Iseconds)] $*" | tee -a "$SUMMARY"
}

log "Session 9 Step 3 ablations launcher starting on cuda:${CARD}"
log "  Runs: $RUN_CODES"
log "  lambda*: $LAMBDA_STAR (used by A7 only)"
log "  repo: $REPO_ROOT"
log "  git HEAD: $(git rev-parse HEAD)"

NAME_A2="run_a2_vicreg_only"
NAME_A7="run_a7_no_scheduled_sampling"
NAME_A10="run_a10_solera_rico"
NAME_A11="run_a11_fukami_ae"

launch_a2_vicreg() {
    # A2: VICReg-only at the production point. Same encoder + predictor
    # + OBS head + scheduled sampling, but anti-collapse module is VICReg
    # with canonical weights (mu=25, lambda_var=25, nu=1 per D22).
    # The lambda-sigreg flag is ignored when --anticollapse vicreg.
    # The train_jepa.py CLI doesn't accept VICReg-specific weights; the
    # defaults inside src/models/vicreg.py are the Bardes et al. canonical
    # ones (25, 25, 1). See HANDOFF.md D22.
    local outdir="$OUTROOT/$NAME_A2"
    mkdir -p "$outdir"
    local logfile="$outdir/train.log"
    log "BEGIN A2 VICReg+OBS+BN d=32 eta=0.01 (canonical VICReg weights) -> $outdir"
    python -m src.training.train_jepa \
        --gpu "$CARD" \
        --partition v1 --all-train --max-iters 20000 --seed 0 \
        --latent-dim 32 \
        --observable-head cl_future --observable-head-weight 0.01 \
        --observable-head-deltas 8 16 24 \
        --projection-norm batchnorm --anticollapse vicreg \
        --diagnostic-every 500 --checkpoint-every 2000 --log-every 50 \
        --output-dir "$outdir" \
        --wandb-mode "$WANDB_MODE" \
        --tag-suffix "${NAME_A2}_section7" \
        >"$logfile" 2>&1
    local rc=$?
    log "END   A2 exit=$rc"
    if [ $rc -ne 0 ]; then
        log "--- last 30 lines of $logfile ---"
        tail -n 30 "$logfile" | tee -a "$SUMMARY"
    fi
    return $rc
}

launch_a7_no_ss() {
    # A7: no scheduled sampling (H_roll = T = 32). Same SIGReg+OBS+BN
    # config at lambda*; just H_roll is bumped from 8 to 32 so the
    # rollout runs to the end of the sub-trajectory.
    local outdir="$OUTROOT/$NAME_A7"
    mkdir -p "$outdir"
    local logfile="$outdir/train.log"
    log "BEGIN A7 SIGReg+OBS+BN no-SS (H_roll=32) lambda=$LAMBDA_STAR -> $outdir"
    python -m src.training.train_jepa \
        --gpu "$CARD" \
        --partition v1 --all-train --max-iters 20000 --seed 0 \
        --latent-dim 32 --H-roll 32 \
        --observable-head cl_future --observable-head-weight 0.01 \
        --observable-head-deltas 8 16 24 \
        --projection-norm batchnorm --anticollapse sigreg \
        --lambda-sigreg "$LAMBDA_STAR" \
        --diagnostic-every 500 --checkpoint-every 2000 --log-every 50 \
        --output-dir "$outdir" \
        --wandb-mode "$WANDB_MODE" \
        --tag-suffix "${NAME_A7}_section7" \
        >"$logfile" 2>&1
    local rc=$?
    log "END   A7 exit=$rc"
    if [ $rc -ne 0 ]; then
        log "--- last 30 lines of $logfile ---"
        tail -n 30 "$logfile" | tee -a "$SUMMARY"
    fi
    return $rc
}

launch_a10_solera() {
    # A10: Solera-Rico beta-VAE + transformer at d=32.
    local outdir="$OUTROOT/$NAME_A10"
    mkdir -p "$outdir"
    local logfile="$outdir/train.log"
    log "BEGIN A10 Solera-Rico beta-VAE + transformer d=32 -> $outdir"
    python -m src.training.train_baseline \
        --baseline solera_rico \
        --gpu "$CARD" \
        --partition v1 --all-train --max-iters 20000 --seed 0 \
        --latent-dim 32 --B 16 --T 32 \
        --observable-head cl_future --observable-head-weight 0.01 \
        --observable-head-deltas 8 16 24 \
        --projection-norm batchnorm \
        --diagnostic-every 500 --checkpoint-every 2000 --log-every 50 \
        --output-dir "$outdir" \
        --wandb-mode "$WANDB_MODE" \
        --tag-suffix "${NAME_A10}_section7" \
        >"$logfile" 2>&1
    local rc=$?
    log "END   A10 exit=$rc"
    if [ $rc -ne 0 ]; then
        log "--- last 30 lines of $logfile ---"
        tail -n 30 "$logfile" | tee -a "$SUMMARY"
    fi
    return $rc
}

launch_a11_fukami() {
    # A11: Fukami observable-augmented AE at d=32.
    local outdir="$OUTROOT/$NAME_A11"
    mkdir -p "$outdir"
    local logfile="$outdir/train.log"
    log "BEGIN A11 Fukami observable-augmented AE d=32 -> $outdir"
    python -m src.training.train_baseline \
        --baseline fukami_ae \
        --gpu "$CARD" \
        --partition v1 --all-train --max-iters 20000 --seed 0 \
        --latent-dim 32 --B 16 --T 32 \
        --observable-head cl_future --observable-head-weight 0.01 \
        --observable-head-deltas 8 16 24 \
        --projection-norm batchnorm \
        --diagnostic-every 500 --checkpoint-every 2000 --log-every 50 \
        --output-dir "$outdir" \
        --wandb-mode "$WANDB_MODE" \
        --tag-suffix "${NAME_A11}_section7" \
        >"$logfile" 2>&1
    local rc=$?
    log "END   A11 exit=$rc"
    if [ $rc -ne 0 ]; then
        log "--- last 30 lines of $logfile ---"
        tail -n 30 "$logfile" | tee -a "$SUMMARY"
    fi
    return $rc
}

IFS=',' read -r -a CODES <<< "$RUN_CODES"
overall_rc=0
for code in "${CODES[@]}"; do
    case "$code" in
        A2)  launch_a2_vicreg ;;
        A7)  launch_a7_no_ss ;;
        A10) launch_a10_solera ;;
        A11) launch_a11_fukami ;;
        *)   log "ERROR: unknown ablation code '$code'"; overall_rc=2; continue ;;
    esac
    rc=$?
    if [ $rc -ne 0 ]; then
        log "$code failed; continuing to next ablation"
        overall_rc=1
    fi
done

log "Step 3 launcher (card $CARD) finished with overall exit=$overall_rc"
exit $overall_rc
