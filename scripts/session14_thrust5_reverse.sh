#!/usr/bin/env bash
# Session 14 Thrust 5 launcher: reverse-factorisation predictor (forces -> latent).
#
# Usage:
#   bash scripts/session14_thrust5_reverse.sh
#
# Self-blocks on the GPU 1 JEPA d=64 seed queue
# (scripts/session14_thrust6_jepa_seeds.sh) by polling its END_OF_QUEUE
# marker in the queue log. Once that queue drains, launches the reverse
# predictor training on --gpu 1 in the background via nohup.
#
# Reference: SESSION14_JFM_NATCOMM_PUSH.md "Thrust 5".

set -euo pipefail

cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"

ROOT="outputs/runs/session14/thrust5_reverse"
mkdir -p "$ROOT"
LAUNCH_LOG="${ROOT}/launch.log"
RUN_LOG="${ROOT}/train.log"
META="${ROOT}/launch_metadata.json"

QUEUE_LOG_PREDECESSOR="outputs/runs/session14/thrust6/jepa_queue_gpu1.log"
QUEUE_END_MARKER="\[s14-t6-jepa\] queue end gpu=1"
QUEUE_SUPERVISOR_PID=3134978  # from outputs/runs/session14/thrust6/launch_metadata.json

echo "[s14-t5] launcher start at $(date -Iseconds)" | tee -a "$LAUNCH_LOG"
echo "[s14-t5] predecessor queue=$QUEUE_LOG_PREDECESSOR supervisor_pid=$QUEUE_SUPERVISOR_PID" \
    | tee -a "$LAUNCH_LOG"

# Wait for the JEPA d=64 queue to finish. Two-stage check: (1) the supervisor
# bash process disappears, OR (2) the queue log emits the queue-end marker.
# Both should happen at the same time; the OR keeps the launcher robust to PID
# reuse vs missing log file.
echo "[s14-t5] waiting on JEPA d=64 GPU 1 queue (poll every 60s)..." \
    | tee -a "$LAUNCH_LOG"

while true; do
    if [ -f "$QUEUE_LOG_PREDECESSOR" ] && grep -qE "$QUEUE_END_MARKER" "$QUEUE_LOG_PREDECESSOR"; then
        echo "[s14-t5] predecessor queue-end marker observed at $(date -Iseconds)" \
            | tee -a "$LAUNCH_LOG"
        break
    fi
    if [ ! -d "/proc/${QUEUE_SUPERVISOR_PID}" ]; then
        echo "[s14-t5] predecessor supervisor PID ${QUEUE_SUPERVISOR_PID} gone at $(date -Iseconds); proceeding" \
            | tee -a "$LAUNCH_LOG"
        break
    fi
    sleep 60
done

# Drain interval so the GPU's last cuda context fully releases before we grab it.
sleep 30
echo "[s14-t5] launching reverse predictor on --gpu 1 at $(date -Iseconds)" \
    | tee -a "$LAUNCH_LOG"

source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

nohup python -u -m src.training.train_reverse_predictor \
    --encoder-checkpoint outputs/runs/session12/S12_E_d64/encoder/checkpoint_iter020000.pt \
    --latents-dir outputs/session14/latents/S12_E_d64 \
    --partition v1 \
    --output-dir "$ROOT" \
    --gpu 1 \
    --seed 42 \
    --B 16 --T 32 \
    --max-iters 20000 \
    --lr 5e-4 \
    --weight-decay 0.05 --warmup-frac 0.05 \
    --hidden-dim 384 --depth 6 --heads 16 --dropout 0.1 \
    --num-workers 2 \
    --log-every 50 --diagnostic-every 500 --checkpoint-every 2000 \
    --standardize-forces \
    --tag-suffix thrust5_reverse_S12Ed64 \
    --wandb-mode offline \
    >> "$RUN_LOG" 2>&1 &

RUN_PID=$!
echo "[s14-t5] reverse predictor PID=$RUN_PID at $(date -Iseconds)" \
    | tee -a "$LAUNCH_LOG"

# Persist launch metadata for downstream tooling.
cat > "$META" <<EOF
{
  "session": "Session 14 Thrust 5 reverse predictor (forces -> S12_E_d64 latent)",
  "spec": "SESSION14_JFM_NATCOMM_PUSH.md",
  "launcher_script": "scripts/session14_thrust5_reverse.sh",
  "predecessor_queue_log": "$QUEUE_LOG_PREDECESSOR",
  "predecessor_supervisor_pid": $QUEUE_SUPERVISOR_PID,
  "armed_iso8601": "$(date -Iseconds)",
  "output_dir": "$ROOT",
  "train_log": "$RUN_LOG",
  "metrics_jsonl": "${ROOT}/metrics.jsonl",
  "run_pid": $RUN_PID,
  "gpu_index": 1,
  "estimated_wall_h": 7,
  "estimated_iters": 20000,
  "hardware_policy": "RTX 6000 Blackwell (cuda:1); L40S forbidden per CLAUDE.md Hardware rule"
}
EOF
echo "[s14-t5] metadata written to $META" | tee -a "$LAUNCH_LOG"
