#!/usr/bin/env bash
# Early d16 read on the IDLE card (slot 4 st_d16_s42 trains on gpu0; gpu1 free):
# extract st_d16 s0/s1 latents (ALL splits, canonical dir so PHASE C skips them
# and decoders still work), roll their own predictors, compute the d16 ST
# forecast band. --gpu 1 / --device cuda:3 = the idle RTX (zero contention).
set -uo pipefail
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}" WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 OMP_WAIT_POLICY=PASSIVE
SPLIT="configs/splits/split_v2p1.json"; MAN="outputs/data_pipeline/v2p1/manifest.json"
RUN=outputs/runs/session29/st; LAT=outputs/session28/latents; ROLL=outputs/session29/lowd_rollouts
LOG=outputs/runs/session29/st_d16_early.log
echo "[early16] START $(date -Iseconds)" >> "$LOG"

for s in 0 1; do
  ck=$RUN/st_d16_s${s}/encoder/checkpoint_iter020000.pt
  [ -f "$ck" ] || { echo "[early16] no ckpt st_d16_s${s}; skip" >> "$LOG"; continue; }
  o=$LAT/st_d16_s${s}
  if [ ! -f "$o/test_b.npz" ]; then
    echo "[early16] extract st_d16_s${s} $(date -Iseconds)" >> "$LOG"
    taskset -c 16-31 python scripts/session18/encode_baseline_latents.py --baseline jepa --d 16 \
      --checkpoint "$ck" --partition v2p1 --split "$SPLIT" --pipeline-manifest "$MAN" \
      --splits train test_a test_b test_c --gpu 1 --output-dir "$o" >> "$LOG" 2>&1 || echo "[early16] extract FAIL s${s}" >> "$LOG"
  fi
  r=$ROLL/st_own_d16_s${s}
  if [ ! -f "$r/test_b.npz" ] && [ -f "$o/test_b.npz" ]; then
    echo "[early16] roll st_own_d16_s${s} $(date -Iseconds)" >> "$LOG"
    taskset -c 16-31 python scripts/session29/roll_own_predictor.py "$ck" "$o/test_b.npz" "$r" \
      --device cuda:3 >> "$LOG" 2>&1 || echo "[early16] roll FAIL s${s}" >> "$LOG"
  fi
done

echo "[early16] FORECAST BAND $(date -Iseconds)" >> "$LOG"
taskset -c 16-23 python - >> "$LOG" 2>&1 <<'PY'
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path("scripts/session29").resolve()))
import m_lowd_forecast as M
HS = (1, 2, 4, 8, 12, 16); OBS = "wake_enstrophy"
otr, itr, _ = M.load_obs("train"); otb, itb, _ = M.load_obs("test_b")
curves, used = [], []
for s in (0, 1):
    lat, rk = f"st_d16_s{s}", f"st_own_d16_s{s}"
    if not (M.ROLL / rk / "test_b.npz").exists() or not (M.LAT / lat / "train.npz").exists():
        continue
    gs = M.fit_probe(lat, OBS, otr, itr)
    c = M.forecast_curve(rk, lat, OBS, gs, otb, itb)
    if c is not None:
        curves.append(c); used.append(s)
print("=== ST d=16 EARLY (2-seed) forecast band, wake_enstrophy, R^2 mean[min,max] ===")
print("    h:   " + "  ".join(f"{h:>15d}" for h in HS))
if curves:
    arr = {h: np.array([c[h] for c in curves]) for h in HS}
    print("  ST-own:" + "  ".join(f"{f'{arr[h].mean():+.2f}[{arr[h].min():+.2f},{arr[h].max():+.2f}]':>15s}" for h in HS) + f"   (seeds {used})")
    for s in used:
        z = np.load(M.LAT / f"st_d16_s{s}" / "test_b.npz")["z_full"].reshape(-1, 16)
        sv = np.linalg.svd(z - z.mean(0), compute_uv=False)**2
        print(f"  st_d16_s{s}: PR={sv.sum()**2/(sv**2).sum():.1f}/16")
else:
    print("  (no rolls yet)")
PY
echo "[early16] DONE $(date -Iseconds)" >> "$LOG"
