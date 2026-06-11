#!/usr/bin/env bash
# T8 orchestration (author decision 2026-06-11, canonical sum-KL + L-curve elbow):
# 1. wait for the original GPU-1 queue to complete (its bvae production cells
#    fail fast against the missing beta pin, by design);
# 2. re-invoke launch_queue.sh for GPU 1: completed cells SKIP, the
#    bvae_lcurve_* sweep runs (5 short cells, 2 per card);
# 3. pick the rate-distortion elbow and pin it (pick_bvae_beta.py);
# 4. re-invoke launch_queue.sh for GPU 1: only the production bvae cells run,
#    now at the pinned beta.
# Every stage is idempotent; re-run this script after any interruption.
set -uo pipefail
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
LOG=outputs/runs/session28/queue_gpu1.log

echo "[bvae-runner] waiting for the gpu1 queue to complete ($LOG)"
while ! grep -q "QUEUE COMPLETE" "$LOG" 2>/dev/null; do sleep 300; done
echo "[bvae-runner] gpu1 free at $(date -Iseconds); running the L-curve sweep"

bash scripts/session28/launch_queue.sh 1

source .venv/bin/activate
python scripts/session28/pick_bvae_beta.py || {
    echo "[bvae-runner] FATAL: elbow pick failed; production bvae cells NOT launched"
    exit 1
}

echo "[bvae-runner] beta pinned; running production bvae cells"
bash scripts/session28/launch_queue.sh 1
echo "[bvae-runner] T8 complete at $(date -Iseconds)"
