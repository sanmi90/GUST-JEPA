#!/usr/bin/env bash
# Re-run Stage 6 (S16) + Stage 7 (S17) interpretability analyses against the
# JEPA d=32 encoder + latents. Uses bulk in-place sed of the path constants in
# scripts/session{16,17}/*.py to swap d64 references for d32, then runs the
# encode-seed-latents + analysis drivers, then reverses the sed at the end.
#
# Outputs land at outputs/session16_d32/ and outputs/session17_d32/ so the
# d=64 outputs are NOT clobbered.

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
source "$REPO/.venv/bin/activate"
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export VORTEX_JEPA_CACHE="${VORTEX_JEPA_CACHE:-$PREVENT_ROOT/data/processed/vortex-jepa}"

LOG="$REPO/outputs/runs/d32_rerun_driver.log"
mkdir -p "$(dirname "$LOG")"
echo "[d32-rerun] start at $(date -Iseconds)" | tee "$LOG"

# ---- 0. Back up scripts before in-place sed ----
BACKUP_DIR="/tmp/d32_rerun_scripts_backup_$(date +%s)"
mkdir -p "$BACKUP_DIR"
cp -r scripts/session16 "$BACKUP_DIR/session16"
cp -r scripts/session17 "$BACKUP_DIR/session17"
echo "[d32-rerun] script backup at $BACKUP_DIR" | tee -a "$LOG"

# ---- 1. Bulk sed: rewrite d=64 paths to d=32 paths ----
do_sed () {
    local pattern_from="$1"
    local pattern_to="$2"
    for f in scripts/session16/*.py scripts/session17/*.py; do
        [[ -f "$f" ]] || continue
        sed -i "s|${pattern_from}|${pattern_to}|g" "$f"
    done
}
echo "[d32-rerun] bulk sed:" | tee -a "$LOG"
do_sed "S12_E_d64"                    "S12_E_d32"
do_sed "jepa_d64_seed"                "jepa_d32_seed"
do_sed "outputs/session16/exp"        "outputs/session16_d32/exp"
do_sed "outputs/session16/figures"    "outputs/session16_d32/figures"
do_sed "outputs/session17/exp"        "outputs/session17_d32/exp"
do_sed "outputs/session17/figures"    "outputs/session17_d32/figures"
do_sed "outputs/session17/diagnostic_d" "outputs/session17_d32/diagnostic_d"
do_sed "outputs/session17/seed_latents" "outputs/session17_d32/seed_latents"
echo "  done" | tee -a "$LOG"

# ---- 2. Prepare d=32 output dirs + encode d=32 seed latents ----
mkdir -p outputs/session16_d32/{exp1,exp2,exp3,exp4,figures}
mkdir -p outputs/session17_d32/{exp1,exp2,exp3,exp4,exp5,figures,diagnostic_d}

echo "[d32-rerun] encoding d=32 seed latents" | tee -a "$LOG"
python -u scripts/session17/encode_seed_latents.py --gpu 0 \
    --seeds production seed0 seed1 seed2 \
    >> "$LOG" 2>&1 \
    || echo "[d32-rerun] FAIL encode_seed_latents" | tee -a "$LOG"

# ---- 3. Run S16 + S17 drivers (CPU-mostly) ----
echo "[d32-rerun] running Stage 6 + Stage 7 analyses (d=32)" | tee -a "$LOG"
bash scripts/_oneoff_run_stage6.sh >> "$LOG" 2>&1 &
S6=$!
bash scripts/_oneoff_run_stage7.sh >> "$LOG" 2>&1 &
S7=$!
wait $S6 $S7

# ---- 4. Restore original scripts ----
echo "[d32-rerun] restoring original scripts from $BACKUP_DIR" | tee -a "$LOG"
rm -rf scripts/session16 scripts/session17
cp -r "$BACKUP_DIR/session16" scripts/session16
cp -r "$BACKUP_DIR/session17" scripts/session17

echo "[d32-rerun] end at $(date -Iseconds)" | tee -a "$LOG"
echo "[d32-rerun] outputs at outputs/session16_d32/ and outputs/session17_d32/" | tee -a "$LOG"
