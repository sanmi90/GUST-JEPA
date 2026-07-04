#!/bin/bash
# SESSION 34 Track C eval chain. Runs after every queue run has its done-marker.
#
# E1  encode latent caches for all 33 cell runs (both GPUs, split)
# E2  lift eval (C2) + head closure (C4)          (CPU, parallel)
# E3  region SSIM (C3, s0 cells, gpu0)
# E4  OSP taps (CPU) -> per-cell rho tuning (both GPUs) -> merge -> envelopes
# E5  gates + numbers (if scripts present)
#
# Usage: scripts/session34/run_trackc_eval.sh
set -u
cd /home/carlos/GUST-JEPA
source .venv/bin/activate
export PREVENT_ROOT=$HOME/PREVENT WANDB_PROJECT=vortex-jepa OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
OUT=outputs/session34
LOG=$OUT/eval_chain.log
mkdir -p "$OUT"
exec >> "$LOG" 2>&1
echo "[eval] chain start @ $(date -Iseconds)"

# Symlinks for reused runs + fail if any checkpoint is missing.
python -m scripts.session34.trackc_cells | tee /dev/stderr
MISSING=$(python -c "from scripts.session34.trackc_cells import missing_checkpoints; m=missing_checkpoints(); print(len(m))")
if [ "$MISSING" != "0" ]; then
  echo "[eval] ABORT: $MISSING checkpoints missing"; exit 2
fi

# ---- E1: encode (split runs across both GPUs) -------------------------------
ALL_RUNS=$(python -c "from scripts.session34.trackc_cells import all_run_names; print(' '.join(all_run_names()))")
read -r -a RUNS <<< "$ALL_RUNS"
HALF=$(( (${#RUNS[@]} + 1) / 2 ))
G0="${RUNS[@]:0:$HALF}"
G1="${RUNS[@]:$HALF}"
taskset -c 0-7  python -m scripts.session34.trackc_encode --gpu 0 --models $G0 &
P0=$!
taskset -c 8-15 python -m scripts.session34.trackc_encode --gpu 1 --models $G1 &
P1=$!
wait $P0 $P1
echo "[eval] E1 encode done @ $(date -Iseconds)"

# ---- E2: lift eval + head closure (CPU) -------------------------------------
taskset -c 0-7  python -m scripts.session34.trackc_lift_eval &
P0=$!
taskset -c 8-15 python -m scripts.session34.trackc_head_closure &
P1=$!
wait $P0 $P1
echo "[eval] E2 lift+closure done @ $(date -Iseconds)"

# ---- E3: region SSIM (gpu0) --------------------------------------------------
taskset -c 0-15 python -m scripts.session34.trackc_region_ssim --gpu 0
echo "[eval] E3 region-ssim done @ $(date -Iseconds)"

# ---- E4: taps -> rho tuning -> envelopes -------------------------------------
taskset -c 0-15 python -m scripts.session34.trackc_taps
S0_RUNS=$(python -c "from scripts.session34.trackc_cells import CELLS; print(' '.join(CELLS[c][0] for c in CELLS))")
i=0
PIDS=()
for RN in $S0_RUNS; do
  GPU=$((i % 2)); CORES=$([ "$GPU" = 0 ] && echo 0-7 || echo 8-15)
  if [ ! -f "$OUT/tuning_${RN}.json" ]; then
    taskset -c "$CORES" python -m scripts.session32.track_b_freeze_tuning \
      --gpu "$GPU" --model "$RN" --run-dir "outputs/runs/session34/$RN" \
      --osp-taps "$OUT/osp_taps_trackc.json" \
      --train-cache "$OUT/trackc_latents/latents_%s_train.npz" \
      --out "$OUT/tuning_${RN}.json" --sweep-out "$OUT/tuning_sweep_${RN}.json" &
    PIDS+=($!)
    # two at a time (one per GPU)
    if [ $((i % 2)) = 1 ]; then wait "${PIDS[@]}"; PIDS=(); fi
    i=$((i + 1))
  fi
done
[ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"
python - <<'EOF'
import json
from pathlib import Path
from scripts.session34.trackc_cells import CELLS
out = Path("outputs/session34")
merged = {}
for cell in CELLS:
    rn = CELLS[cell][0]
    p = out / f"tuning_{rn}.json"
    if p.exists():
        blob = json.loads(p.read_text())
        merged[rn] = {"rho": float(blob["selection"]["selected_rho"]), "cell": cell}
(out / "filter_tuning_trackc.json").write_text(json.dumps(merged, indent=2))
print(f"[eval] merged rho tuning for {len(merged)} cells")
EOF
echo "[eval] E4 tuning done @ $(date -Iseconds)"

# Envelopes: split cells across GPUs (each invocation skips existing outputs).
CELLS_LIST=$(python -c "from scripts.session34.trackc_cells import CELLS; print(' '.join(CELLS))")
read -r -a CARR <<< "$CELLS_LIST"
CHALF=$(( (${#CARR[@]} + 1) / 2 ))
C0="${CARR[@]:0:$CHALF}"
C1="${CARR[@]:$CHALF}"
taskset -c 0-7  python -m scripts.session34.trackc_envelope --gpu 0 --cells $C0 &
P0=$!
taskset -c 8-15 python -m scripts.session34.trackc_envelope --gpu 1 --cells $C1 &
P1=$!
wait $P0 $P1
echo "[eval] E4 envelopes done @ $(date -Iseconds)"

# ---- E5: gates + numbers (if present) ----------------------------------------
if [ -f scripts/session34/trackc_gates.py ]; then
  taskset -c 0-15 python -m scripts.session34.trackc_gates || echo "[eval] gates FAILED"
fi
echo "[eval] chain complete @ $(date -Iseconds)"
