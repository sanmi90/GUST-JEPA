#!/usr/bin/env bash
# After the foreground F-L training on cuda:2 (torch index 2 = first RTX 6000)
# finishes, run F-CD then F-S on the same GPU. Polls for F-L's final
# checkpoint and the absence of its python process before claiming the card.
#
# Companion to scripts/run_session6_cuda3_parallel.sh which runs F-NC and
# F-OBS on the second RTX 6000 (cuda:3 in torch's unfiltered view).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
WANDB_MODE="${WANDB_MODE:-offline}"
unset CUDA_VISIBLE_DEVICES  # leave torch's full view; require_rtx6000 picks the first RTX 6000

OUTROOT="outputs/runs/session6"
SUMMARY="$OUTROOT/_session6_summary_cuda2.log"
: > "$SUMMARY"

FL_CKPT="$OUTROOT/run_f_l/checkpoint_iter005000.pt"

# Wait until F-L's final checkpoint exists AND the F-L python process has exited.
echo "[$(date -Iseconds)] waiting for F-L final checkpoint at $FL_CKPT" | tee -a "$SUMMARY"
while true; do
    if [ -f "$FL_CKPT" ]; then
        # Final checkpoint present; also confirm no more F-L python process is allocating cuda:2
        if ! pgrep -f "outputs/runs/session6/run_f_l" >/dev/null; then
            break
        fi
    fi
    sleep 30
done
echo "[$(date -Iseconds)] F-L checkpoint present and process exited; claiming cuda:2" | tee -a "$SUMMARY"

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
        echo "BEGIN $name at $t_start (cuda:2)"
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

# F-CD: per-batch c-dropout 0.5
run_variant run_f_cd \
    --cases-from configs/cases/smoke_5cases.yaml \
    --c-dropout-prob 0.5 \
    --tag-suffix run_f_cd_seed0_p0p5

# F-S: 24-case scale-up
run_variant run_f_s \
    --cases-from configs/cases/smoke_24cases.yaml \
    --tag-suffix run_f_s_seed0_24cases

{
    echo
    echo "=========================================================="
    echo "cuda:2 chain done at $(date -Iseconds)"
    for key in run_f_cd run_f_s; do
        echo "    $key  exit=${EXITS[$key]:-NA}"
    done
} | tee -a "$SUMMARY"

overall=0
for key in run_f_cd run_f_s; do
    if [ "${EXITS[$key]:-1}" -ne 0 ]; then overall=1; fi
done
exit "$overall"
