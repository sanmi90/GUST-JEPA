#!/usr/bin/env bash
# Session 8 Step 6: R0 SIGReg-only control. Pure SIGReg + BN at full
# scale with no observable head. ~1.5h on cuda:0 per D49.
#
# Two args (both optional):
#   $1: lambda (default 0.1 per the Session 7 default)
#   $2: card index (default 0)
#
# Example:
#   scripts/launch_session8_step6_r0.sh 0.1 0
#   scripts/launch_session8_step6_r0.sh 1.0 0   # if Step 4 finds lambda*>0.1

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
WANDB_MODE="${WANDB_MODE:-offline}"

LAM="${1:-0.1}"
CARD="${2:-0}"

OUTROOT="outputs/runs/session8"
mkdir -p "$OUTROOT"

# Output dir encodes lambda so two R0 variants can coexist.
LAM_TAG=$(echo "$LAM" | sed 's/\./p/')
OUTDIR="$OUTROOT/run_r0_sigreg_only_lam${LAM_TAG}"
mkdir -p "$OUTDIR"
LOGFILE="$OUTDIR/train.log"

echo "[$(date -Iseconds)] R0 SIGReg-only control on cuda:${CARD} lambda=$LAM -> $OUTDIR"

python -m src.training.train_jepa \
    --gpu "$CARD" \
    --partition v1 --all-train --max-iters 20000 --seed 0 \
    --observable-head none \
    --projection-norm batchnorm --anticollapse sigreg \
    --lambda-sigreg "$LAM" \
    --diagnostic-every 500 --checkpoint-every 2000 --log-every 50 \
    --output-dir "$OUTDIR" \
    --wandb-mode "$WANDB_MODE" \
    --tag-suffix "run_r0_sigreg_only_lam${LAM_TAG}" \
    >"$LOGFILE" 2>&1

RC=$?
echo "[$(date -Iseconds)] R0 exit=$RC"
exit $RC
