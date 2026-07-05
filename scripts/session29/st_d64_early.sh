#!/usr/bin/env bash
# Ad-hoc early de-risk read: the four d64 ST encoders are trained; extract
# s0/s1/s2 latents (ALL splits, canonical dir so PHASE C skips them and the
# decoders still work), roll their own predictors, and compute the d64 ST
# forecast band NOW (rather than waiting for PHASE C). Low-contention: sequential
# extraction, shares a card with the running d16 training. RTX 6000 only.
set -uo pipefail
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}" WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 OMP_WAIT_POLICY=PASSIVE
SPLIT="configs/splits/split_v2p1.json"; MAN="outputs/data_pipeline/v2p1/manifest.json"
RUN=outputs/runs/session29/st; LAT=outputs/session28/latents; ROLL=outputs/session29/lowd_rollouts
LOG=outputs/runs/session29/st_d64_early.log
echo "[early] START $(date -Iseconds)" >> "$LOG"

for s in 0 1 2; do
  ck=$RUN/st_d64_s${s}/encoder/checkpoint_iter020000.pt
  [ -f "$ck" ] || { echo "[early] no ckpt st_d64_s${s}; skip" >> "$LOG"; continue; }
  o=$LAT/st_d64_s${s}
  if [ ! -f "$o/test_b.npz" ]; then
    echo "[early] extract st_d64_s${s} $(date -Iseconds)" >> "$LOG"
    taskset -c 0-31 python scripts/session18/encode_baseline_latents.py --baseline jepa --d 64 \
      --checkpoint "$ck" --partition v2p1 --split "$SPLIT" --pipeline-manifest "$MAN" \
      --splits train test_a test_b test_c --gpu 0 --output-dir "$o" >> "$LOG" 2>&1 || echo "[early] extract FAIL s${s}" >> "$LOG"
  fi
  r=$ROLL/st_own_d64_s${s}
  if [ ! -f "$r/test_b.npz" ] && [ -f "$o/test_b.npz" ]; then
    echo "[early] roll st_own_d64_s${s} $(date -Iseconds)" >> "$LOG"
    taskset -c 0-31 python scripts/session29/roll_own_predictor.py "$ck" "$o/test_b.npz" "$r" \
      --device cuda:2 >> "$LOG" 2>&1 || echo "[early] roll FAIL s${s}" >> "$LOG"
  fi
done

echo "[early] FORECAST BAND $(date -Iseconds)" >> "$LOG"
taskset -c 0-7 python - >> "$LOG" 2>&1 <<'PY'
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path("scripts/session29").resolve()))
import m_lowd_forecast as M
HS = (1, 2, 4, 8, 12, 16); OBS = "wake_enstrophy"
otr, itr, _ = M.load_obs("train"); otb, itb, _ = M.load_obs("test_b")
curves, used = [], []
for s in (0, 1, 2):
    lat, rk = f"st_d64_s{s}", f"st_own_d64_s{s}"
    if not (M.ROLL / rk / "test_b.npz").exists() or not (M.LAT / lat / "train.npz").exists():
        continue
    gs = M.fit_probe(lat, OBS, otr, itr)
    c = M.forecast_curve(rk, lat, OBS, gs, otb, itb)
    if c is not None:
        curves.append(c); used.append(s)
print("=== ST d=64 EARLY forecast band, wake_enstrophy, R^2 mean[min,max] ===")
print("    h:   " + "  ".join(f"{h:>15d}" for h in HS))
if curves:
    arr = {h: np.array([c[h] for c in curves]) for h in HS}
    cells = [f"{arr[h].mean():+.2f}[{arr[h].min():+.2f},{arr[h].max():+.2f}]" for h in HS]
    print("  ST-own:" + "  ".join(f"{c:>15s}" for c in cells) + f"   (seeds {used})")
    # PR + null-dim per seed
    for s in used:
        z = np.load(M.LAT / f"st_d64_s{s}" / "test_b.npz")["z_full"].reshape(-1, 64)
        v = z.var(0); sv = np.linalg.svd(z - z.mean(0), compute_uv=False)**2
        pr = sv.sum()**2 / (sv**2).sum()
        print(f"  st_d64_s{s}: PR={pr:.1f}/64  null(<1%)={int((v<0.01*v.max()).sum())}/64")
else:
    print("  (no rolls yet)")
PY
echo "[early] DONE $(date -Iseconds)" >> "$LOG"
