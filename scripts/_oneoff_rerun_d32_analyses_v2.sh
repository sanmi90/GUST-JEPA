#!/usr/bin/env bash
# Safer version: move d=64 output dirs aside, sed only the encoder/latents
# paths (which use single-component substrings the sed catches reliably), let
# the analyses write to the now-empty default output dirs, then rename to
# *_d32 and restore the d=64 dirs.
#
# Outputs land at outputs/session16_d32/ and outputs/session17_d32/.

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
source "$REPO/.venv/bin/activate"
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export VORTEX_JEPA_CACHE="${VORTEX_JEPA_CACHE:-$PREVENT_ROOT/data/processed/vortex-jepa}"

LOG="$REPO/outputs/runs/d32_rerun_v2_driver.log"
mkdir -p "$(dirname "$LOG")"
echo "[d32-rerun-v2] start at $(date -Iseconds)" | tee "$LOG"

TS=$(date +%Y%m%d_%H%M)
BACKUP_DIR="/tmp/d32_rerun_v2_scripts_backup_${TS}"
mkdir -p "$BACKUP_DIR"
cp -r scripts/session16 "$BACKUP_DIR/session16"
cp -r scripts/session17 "$BACKUP_DIR/session17"
echo "[d32-rerun-v2] script backup at $BACKUP_DIR" | tee -a "$LOG"

# Move d=64 output dirs aside (we'll restore at the end)
mv outputs/session16 outputs/session16_d64_stash_${TS}
mv outputs/session17 outputs/session17_d64_stash_${TS}
mkdir -p outputs/session16/{exp1,exp2,exp3,exp4,figures}
mkdir -p outputs/session17/{exp1,exp2,exp3,exp4,exp5,figures,diagnostic_d,seed_latents}
echo "[d32-rerun-v2] moved d=64 outputs aside (stash suffix ${TS})" | tee -a "$LOG"

# Sed ONLY the encoder + latents path constants (single-component substrings)
do_sed () {
    local p_from="$1"
    local p_to="$2"
    for f in scripts/session16/*.py scripts/session17/*.py; do
        [[ -f "$f" ]] || continue
        sed -i "s|${p_from}|${p_to}|g" "$f"
    done
}
do_sed "S12_E_d64"      "S12_E_d32"
do_sed "jepa_d64_seed"  "jepa_d32_seed"
echo "[d32-rerun-v2] sed done (S12_E_d64 -> S12_E_d32; jepa_d64_seed -> jepa_d32_seed)" | tee -a "$LOG"

# Encode d=32 seed latents (writes to outputs/session17/seed_latents — now empty)
echo "[d32-rerun-v2] encoding d=32 seed latents" | tee -a "$LOG"
python -u scripts/session17/encode_seed_latents.py --gpu 0 \
    --seeds production seed0 seed1 seed2 \
    >> "$LOG" 2>&1 \
    || echo "[d32-rerun-v2] FAIL encode_seed_latents" | tee -a "$LOG"

# Run Stage 6 + Stage 7 drivers in parallel
echo "[d32-rerun-v2] running Stage 6 + Stage 7 drivers" | tee -a "$LOG"
bash scripts/_oneoff_run_stage6.sh >> "$LOG" 2>&1 &
S6=$!
bash scripts/_oneoff_run_stage7.sh >> "$LOG" 2>&1 &
S7=$!
wait $S6 $S7

# Rename d=32 outputs aside
mv outputs/session16 outputs/session16_d32
mv outputs/session17 outputs/session17_d32
# Restore d=64
mv outputs/session16_d64_stash_${TS} outputs/session16
mv outputs/session17_d64_stash_${TS} outputs/session17

# Restore scripts
rm -rf scripts/session16 scripts/session17
cp -r "$BACKUP_DIR/session16" scripts/session16
cp -r "$BACKUP_DIR/session17" scripts/session17

echo "[d32-rerun-v2] DONE at $(date -Iseconds)" | tee -a "$LOG"
echo "[d32-rerun-v2] d=32 outputs at outputs/session16_d32/ and outputs/session17_d32/" | tee -a "$LOG"
echo "[d32-rerun-v2] d=64 outputs restored to outputs/session16/ and outputs/session17/" | tee -a "$LOG"
