#!/usr/bin/env bash
# Closure chain for the no-conditioning JEPA d=64 (JEPA_d64_noc). Mirrors
# closure_dsweep.sh but trains the CLOSURE predictor with --cond-dim 0, so neither
# the JEPA predictor nor the closure predictor ever sees (G,D,Y). The rollout reads
# cond_dim=0 from the saved predictor_config and the predictor ignores the cond
# tensor. Metrics via the canonical exp_closure_r2 probe -> closure_r2_noc.csv.
#   bash scripts/session23/closure_noc.sh [gpu]
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
source "$REPO/.venv/bin/activate"
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}" WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
GPU="${1:-0}"

tag="jepa_d64_noc_noBN"
ENC="outputs/runs/session23/JEPA_d64_noc/checkpoint_iter020000.pt"
LAT="outputs/session18/exp_b1/latents_$tag"
PREDIR="outputs/session18/exp_b1_test3/predictor_$tag"
PRED="$PREDIR/checkpoint_iter020000.pt"
ROLL="outputs/session18/exp_b1_test3/rollouts_$tag"
[[ -f "$ENC" ]] || { echo "[FATAL] encoder missing: $ENC"; exit 2; }

echo ">>> [$tag] 1/3 encode latents (d=64) -> $LAT"
[[ -f "$LAT/test_b.npz" ]] || python scripts/session18/encode_baseline_latents.py \
  --baseline jepa --d 64 --checkpoint "$ENC" --output-dir "$LAT" --gpu "$GPU"

echo ">>> [$tag] 2/3 train noBN, NO-COND predictor -> $PREDIR"
[[ -f "$PRED" ]] || python scripts/session18/train_baseline_predictor.py \
  --latents-dir "$LAT" --tag "$tag" --output-dir "$PREDIR" \
  --no-output-bn --cond-dim 0 --gpu "$GPU" --seed 0 --num-workers 0

echo ">>> [$tag] 3/3 rollouts -> $ROLL"
[[ -f "$ROLL/test_b.npz" ]] || python scripts/session18/eval_baseline_rollouts.py \
  --latents-dir "$LAT" --predictor "$PRED" --tag "$tag" --output-dir "$ROLL" --gpu "$GPU"

echo ">>> closure R^2 + MAE (reuses exp_closure_r2) -> outputs/session23_closure/closure_r2_noc.csv"
python - <<'PY'
import csv, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[0] if False else Path.cwd()
sys.path.insert(0, str(REPO / "scripts" / "session20"))
import exp_closure_r2 as cr
dns = np.load(cr.DNS_METRICS_PATH, allow_pickle=True)
rows = cr.evaluate("jepa_d64_noc_noBN", "jepa", 64, dns, (1,4,8,16,32,64), ["test_b","test_c"], n_boot=2000)
out = REPO / "outputs/session23_closure/closure_r2_noc.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"wrote {len(rows)} rows to {out}")
PY

echo "=== CLOSURE NOC COMPLETE -> outputs/session23_closure/closure_r2_noc.csv ==="
