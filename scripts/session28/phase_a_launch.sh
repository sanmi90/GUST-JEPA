#!/usr/bin/env bash
# Session 28 Phase A: one-shot pre-launch sequence + training-matrix launch.
# Master plan A0/A1 execution order:
#   1. sanity gate + GPU check (RTX 6000s idle)
#   2. v2p1 cache symlink (v2p1 -> v1, the v2 pattern)
#   3. data integrity audit on split_v2p1 (expect 0 flags of 382)
#   4. wake-observable recompute for v2p1 (backup v2 stats first; --force so all
#      382 encounters carry v2p1-normalised targets)
#   5. launch the wave queue on both cards (background, logs under outputs/runs/session28)
#
# Each step is idempotent / skip-aware so the script can be re-run.
# Usage: bash scripts/session28/phase_a_launch.sh [--skip-audit] [--no-launch]
set -uo pipefail
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
REPO=$(pwd)
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
CACHE="${VORTEX_JEPA_CACHE:-$PREVENT_ROOT/data/processed/vortex-jepa}"
SKIP_AUDIT=0; NO_LAUNCH=0
for a in "$@"; do
    [[ "$a" == "--skip-audit" ]] && SKIP_AUDIT=1
    [[ "$a" == "--no-launch" ]] && NO_LAUNCH=1
done
LOG_DIR="outputs/runs/session28"; mkdir -p "$LOG_DIR"

echo "=== [1/5] sanity gate + GPUs ==="
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
RTX_BUSY=$(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader \
    | grep -i "RTX" | awk -F', ' '{gsub(/ MiB/,"",$2); if ($2+0 > 2000) n++} END {print n+0}')
if [[ "$RTX_BUSY" != "0" ]]; then
    echo "WARNING: $RTX_BUSY RTX 6000 card(s) show >2 GB in use; queue will contend with the existing job(s)."
fi
python -m src.training.sanity_checks --all --require-gpu || { echo "FATAL: sanity checks failed"; exit 1; }

echo "=== [2/5] v2p1 cache symlink ==="
if [[ ! -e "$CACHE/v2p1" ]]; then
    ln -s "$CACHE/v1" "$CACHE/v2p1"
    echo "created $CACHE/v2p1 -> v1"
else
    echo "exists: $(ls -la "$CACHE/v2p1" | head -1 || true)"
fi

if [[ "$SKIP_AUDIT" == "0" ]]; then
    echo "=== [3/5] data integrity audit (split_v2p1, expect 0 flags of 382) ==="
    python scripts/data_integrity_audit.py --split configs/splits/split_v2p1.json \
        --cl-hard-cap 12 --pwall-hard-cap 15 2>&1 | tee "$LOG_DIR/integrity_audit.log" | tail -15
    # The audit script always exits 0; gate on its summary line (PF-A1 demands 0 of 382).
    grep -q "\[audit\] 0 flagged of 382 encounters" "$LOG_DIR/integrity_audit.log" \
        || { echo "FATAL: integrity audit did not report '0 flagged of 382'"; exit 1; }
else
    echo "=== [3/5] audit SKIPPED by flag ==="
fi

echo "=== [4/5] wake-observable recompute for v2p1 ==="
WOBS="$CACHE/v1/wake_observables"
MARKER_OK=$(grep -l '"v2p1"' "$WOBS/_manifest.json" 2>/dev/null || true)
if [[ -n "$MARKER_OK" && -f "$WOBS/G-1.50_D1.50_Y+0.00/encounter_00.h5" ]]; then
    echo "already recomputed for v2p1; skipping"
else
    if [[ -f "$WOBS/_train_stats.json" && ! -f "$WOBS/_train_stats_v2_backup.json" ]]; then
        cp "$WOBS/_train_stats.json" "$WOBS/_train_stats_v2_backup.json"
        echo "backed up v2 train stats -> _train_stats_v2_backup.json"
    fi
    python scripts/session11_precompute_wake_observables.py \
        --partition v2p1 \
        --split configs/splits/split_v2p1.json \
        --omega-pipeline-manifest outputs/data_pipeline/v2p1/manifest.json \
        --force 2>&1 | tee "$LOG_DIR/wake_precompute.log" | tail -10
fi
grep -q '"v2p1"' "$WOBS/_manifest.json" || { echo "FATAL: wake manifest lacks v2p1 marker"; exit 1; }

if [[ "$NO_LAUNCH" == "1" ]]; then
    echo "=== [5/5] launch SKIPPED by flag (--no-launch) ==="
    exit 0
fi
echo "=== [5/5] launching the wave queue on both cards ==="
nohup bash scripts/session28/launch_queue.sh > "$LOG_DIR/launch_queue.nohup.log" 2>&1 &
QPID=$!
echo "queue driver pid=$QPID; logs: $LOG_DIR/queue_gpu{0,1}.log"
sleep 90
echo "--- first runs after 90 s ---"
tail -5 "$LOG_DIR/queue_gpu0.log" "$LOG_DIR/queue_gpu1.log" 2>/dev/null || true
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
python scripts/session28/build_manifest_runs.py || true
