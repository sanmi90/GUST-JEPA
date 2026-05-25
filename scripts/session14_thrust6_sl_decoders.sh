#!/usr/bin/env bash
# Session 14 Thrust 6 follow-up: train LapFiLM SL decoders for each JEPA
# d=64 seed produced by scripts/session14_thrust6_jepa_seeds.sh.
#
# Usage:
#   bash scripts/session14_thrust6_sl_decoders.sh
#
# Self-blocks on the GPU 0 Fukami queue
# (scripts/session14_thrust6_fukami_seeds.sh) by polling its END_OF_QUEUE
# marker and supervisor PID. Once that queue drains, launches three
# region_pyr_specloss decoder retrains on --gpu 0 sequentially, one per
# (jepa_d64_seed0, jepa_d64_seed1, jepa_d64_seed2) encoder.
#
# Each retrain is ~2 h on an RTX 6000 Blackwell (12k iters; Session 13 D99
# convergence horizon). Total queue ~6 h.
#
# The decoder recipe matches Session 13's queue (D99 production weights):
#   --decoder-loss region_pyr_specloss
#   --lambda-region 1.0 --lambda-pyramid 0.4
#   --lambda-gradient 1.0 --lambda-spectral-amp 1.0
#   --lambda-enstrophy 0.02 --lambda-circulation 0.01
#   --spectral-window hann --spectral-wake-only
#   --max-iters 12000 --B 16 --T 32 --seed 42
#
# Reference: scripts/session13_queue_specloss_retrains.sh,
# scripts/session11_launch_decoder.sh, SESSION14_JFM_NATCOMM_PUSH.md
# Thrust 6.

set -euo pipefail

cd "$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"

ROOT="outputs/runs/session14/thrust6"
SL_ROOT="${ROOT}/sl_decoders"
mkdir -p "$SL_ROOT"
LAUNCH_LOG="${SL_ROOT}/launch.log"
QUEUE_LOG="${SL_ROOT}/queue.log"
META="${SL_ROOT}/launch_metadata.json"

QUEUE_LOG_PREDECESSOR="${ROOT}/fukami_queue_gpu0.log"
QUEUE_END_MARKER="\[s14-t6-fuk\] queue end gpu=0"
QUEUE_SUPERVISOR_PID=3134633  # from outputs/runs/session14/thrust6/launch_metadata.json

echo "[s14-t6-sl] launcher start at $(date -Iseconds)" | tee -a "$LAUNCH_LOG"
echo "[s14-t6-sl] predecessor queue=$QUEUE_LOG_PREDECESSOR supervisor_pid=$QUEUE_SUPERVISOR_PID" \
    | tee -a "$LAUNCH_LOG"

# Wait for the Fukami queue (GPU 0) to finish.
echo "[s14-t6-sl] waiting on Fukami GPU 0 queue (poll every 60s)..." \
    | tee -a "$LAUNCH_LOG"
while true; do
    if [ -f "$QUEUE_LOG_PREDECESSOR" ] && grep -qE "$QUEUE_END_MARKER" "$QUEUE_LOG_PREDECESSOR"; then
        echo "[s14-t6-sl] predecessor queue-end marker observed at $(date -Iseconds)" \
            | tee -a "$LAUNCH_LOG"
        break
    fi
    if [ ! -d "/proc/${QUEUE_SUPERVISOR_PID}" ]; then
        echo "[s14-t6-sl] predecessor supervisor PID ${QUEUE_SUPERVISOR_PID} gone at $(date -Iseconds); proceeding" \
            | tee -a "$LAUNCH_LOG"
        break
    fi
    sleep 60
done

# Drain interval so the GPU's last cuda context fully releases.
sleep 30

source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

# Cards used:
#   --gpu 0 (the first RTX 6000 Blackwell; the Fukami queue's slot)
GPU=0

# Persist launch metadata before kicking the queue so downstream tooling can
# locate the run directories even before they finish.
cat > "$META" <<EOF
{
  "session": "Session 14 Thrust 6 SL decoders (LapFiLM region_pyr_specloss on JEPA d=64 seeds)",
  "spec": "SESSION14_JFM_NATCOMM_PUSH.md",
  "launcher_script": "scripts/session14_thrust6_sl_decoders.sh",
  "predecessor_queue_log": "$QUEUE_LOG_PREDECESSOR",
  "predecessor_supervisor_pid": $QUEUE_SUPERVISOR_PID,
  "armed_iso8601": "$(date -Iseconds)",
  "gpu_index": $GPU,
  "queue_log": "$QUEUE_LOG",
  "estimated_per_run_wall_h": 2,
  "estimated_total_wall_h": 6,
  "max_iters": 12000,
  "decoder_loss": "region_pyr_specloss",
  "jobs": [
    {"tag": "jepa_d64_seed0", "encoder_run": "${ROOT}/jepa_d64_seed0/encoder", "decoder_dir": "${ROOT}/jepa_d64_seed0/encoder/decoder_specloss_recipe"},
    {"tag": "jepa_d64_seed1", "encoder_run": "${ROOT}/jepa_d64_seed1/encoder", "decoder_dir": "${ROOT}/jepa_d64_seed1/encoder/decoder_specloss_recipe"},
    {"tag": "jepa_d64_seed2", "encoder_run": "${ROOT}/jepa_d64_seed2/encoder", "decoder_dir": "${ROOT}/jepa_d64_seed2/encoder/decoder_specloss_recipe"}
  ],
  "hardware_policy": "RTX 6000 Blackwell (cuda:0); L40S forbidden per CLAUDE.md Hardware rule"
}
EOF

echo "[s14-t6-sl] queue size = 3 (seeds 0/1/2)" | tee -a "$QUEUE_LOG"

run_decoder() {
    local tag=$1
    local enc_run="${ROOT}/${tag}/encoder"
    local dec_dir="${enc_run}/decoder_specloss_recipe"
    local log="${enc_run}/../decoder_specloss_launch.log"
    mkdir -p "$dec_dir"

    if [ ! -d "$enc_run" ]; then
        echo "[s14-t6-sl] $(date -Iseconds) MISSING encoder dir $enc_run; skipping $tag" \
            | tee -a "$QUEUE_LOG"
        return 0
    fi
    # Pick the latest checkpoint_iter*.pt (defensive: usually iter020000.pt).
    local ckpt=$(ls -1 "$enc_run"/checkpoint_iter*.pt 2>/dev/null | sort | tail -n 1 || true)
    if [ -z "$ckpt" ]; then
        echo "[s14-t6-sl] $(date -Iseconds) MISSING checkpoint in $enc_run; skipping $tag" \
            | tee -a "$QUEUE_LOG"
        return 0
    fi

    echo "[s14-t6-sl] $(date -Iseconds) starting $tag encoder=$enc_run ckpt=$(basename $ckpt) -> $dec_dir" \
        | tee -a "$QUEUE_LOG"

    # Foreground call so the queue blocks until this retrain finishes.
    python -u scripts/session9_train_decoder.py \
        --encoder-run "$enc_run" \
        --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \
        --decoder-type lapfilm \
        --decoder-upsample pixelshuffle \
        --decoder-loss region_pyr_specloss \
        --lambda-region 1.0 --lambda-pyramid 0.4 \
        --lambda-gradient 1.0 --lambda-spectral-amp 1.0 \
        --lambda-enstrophy 0.02 --lambda-circulation 0.01 \
        --spectral-window hann --spectral-wake-only \
        --max-iters 12000 \
        --B 16 --T 32 --seed 42 \
        --gpu "$GPU" \
        --output-dir "$dec_dir" \
        --eval-every 2000 --checkpoint-every 2000 --log-every 200 \
        2>&1 | tee "$log"

    echo "[s14-t6-sl] $(date -Iseconds) finished $tag" | tee -a "$QUEUE_LOG"
}

for seed in 0 1 2; do
    tag="jepa_d64_seed${seed}"
    run_decoder "$tag" || true
done

echo "[s14-t6-sl] $(date -Iseconds) ALL DONE" | tee -a "$QUEUE_LOG"
echo "[s14-t6-sl] queue end at $(date -Iseconds)" | tee -a "$LAUNCH_LOG"
