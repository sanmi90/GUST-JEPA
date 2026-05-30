#!/usr/bin/env bash
# Session 20 Track A launcher (v2, concurrency-packed).
#
# The RTX 6000 Blackwell cards run a single d=64 JEPA/Fukami encoder at only
# ~33% SM util and <20/96 GB, so we pack 3 concurrent trainings per card
# (2 waves of 6 instead of 6 serial waves). Encoder training stays RTX-only
# (CLAUDE.md); the two RTX cards are --gpu 0 and --gpu 1.
#
# Session-scoped 4-card, memory-aware static assignment (user authorised both
# L40S this session). Under the bypass --gpu == torch cuda index:
#   cuda 0,1 = L40S (48 GB)   cuda 2,3 = RTX 6000 (102 GB)
# Fukami cells carry a full-field decoder (~20 GB each), so the heavy cells go
# on the 102 GB RTX cards; the 48 GB L40S each take 2 light jobs. 12 jobs, 1 wave.
#   cuda 2 (RTX) : A3 s0,s1,s2 + A2 s0        (~79 GB)
#   cuda 3 (RTX) : A4 s0,s1,s2 + A2 s1        (~73 GB)
#   cuda 0 (L40S): A5 s0, s1                   (~38 GB)
#   cuda 1 (L40S): A5 s2 + A2 s2               (~32 GB)
# A1 already exists (thrust6 jepa_d64_seed{0,1,2}); not retrained.
#
# Usage: bash scripts/session20/launch_track_a.sh

set -uo pipefail
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
REPO=$(pwd)
ROOT="outputs/runs/session20/track_a"
mkdir -p "$ROOT"
ONE="$REPO/scripts/session20/_train_one.sh"
chmod +x "$ONE"

run_card() {  # gpu  "cell seed" "cell seed" ... -> all concurrent on that card
    local gpu=$1; shift
    printf '%s %s\n' "$@" | awk -v g="$gpu" '{print $1, $2, g}' \
        | xargs -P "$#" -L1 bash "$ONE"
}

# Each card's job list is run fully concurrent (xargs -P = list length).
{ for j in "A3_recon_cnnvit 0" "A3_recon_cnnvit 1" "A3_recon_cnnvit 2" "A2_pred_cnn 0"; do echo "$j 2"; done | xargs -P4 -L1 bash "$ONE"; } > "$ROOT/cuda2.log" 2>&1 &
Q2=$!
{ for j in "A4_recon_cnn 0" "A4_recon_cnn 1" "A4_recon_cnn 2" "A2_pred_cnn 1"; do echo "$j 3"; done | xargs -P4 -L1 bash "$ONE"; } > "$ROOT/cuda3.log" 2>&1 &
Q3=$!
{ for j in "A5_pred_nowake 0" "A5_pred_nowake 1"; do echo "$j 0"; done | xargs -P2 -L1 bash "$ONE"; } > "$ROOT/cuda0.log" 2>&1 &
Q0=$!
{ for j in "A5_pred_nowake 2" "A2_pred_cnn 2"; do echo "$j 1"; done | xargs -P2 -L1 bash "$ONE"; } > "$ROOT/cuda1.log" 2>&1 &
Q1=$!
echo "[track-a] launched 4-card memory-aware at $(date -Iseconds): cuda2=A3x3+A2s0 cuda3=A4x3+A2s1 cuda0=A5s0,s1 cuda1=A5s2+A2s2"
wait $Q2 $Q3 $Q0 $Q1
echo "[track-a] ALL ENCODER TRAINING COMPLETE at $(date -Iseconds)"
