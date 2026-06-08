#!/usr/bin/env bash
# Clean test of the coauthor question on the PRODUCTION pipeline (reproduces 0.449):
# hold the production encoder FROZEN (latents_jepa_d64_test1_noBN) and train the
# closure predictor with --cond-dim 0 (no gust c), 3 seeds. Rollout + canonical
# closure (exp_closure_r2). If wake R2 does NOT collapse to ~0.038, the published
# no-c collapse was an ENCODER effect (JEPA_d64_noc), not a predictor-input effect.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
source "$REPO/.venv/bin/activate"
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}" WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
GPU="${1:-0}"
PRODLAT="outputs/session18/exp_b1/latents_jepa_d64_test1_noBN"

for seed in 0 1 2; do
  tag="test1_noBN_predc0_s${seed}"
  PREDIR="outputs/session27/predictor_${tag}"
  PRED="${PREDIR}/checkpoint_iter020000.pt"
  ROLL="outputs/session18/exp_b1_test3/rollouts_${tag}"
  # symlink so exp_closure_r2 (LATENTS_ROOT/latents_{tag}) finds the frozen-encoder train latents
  ln -sfn "$REPO/${PRODLAT}" "outputs/session18/exp_b1/latents_${tag}"
  echo ">>> [${tag}] train cond-0 predictor on frozen production encoder"
  [[ -f "$PRED" ]] || python scripts/session18/train_baseline_predictor.py \
    --latents-dir "$PRODLAT" --tag "$tag" --output-dir "$PREDIR" \
    --no-output-bn --cond-dim 0 --gpu "$GPU" --seed "$seed" --num-workers 2
  echo ">>> [${tag}] rollouts"
  [[ -f "$ROLL/test_b.npz" ]] || python scripts/session18/eval_baseline_rollouts.py \
    --latents-dir "$PRODLAT" --predictor "$PRED" --tag "$tag" --output-dir "$ROLL" --gpu "$GPU"
done

echo ">>> closure R^2 (canonical exp_closure_r2) for all seeds"
python - <<'PY'
import sys, numpy as np
from pathlib import Path
REPO = Path.cwd(); sys.path.insert(0, str(REPO / "scripts" / "session20"))
import exp_closure_r2 as cr
dns = np.load(cr.DNS_METRICS_PATH, allow_pickle=True)
def wake16(tag, split):
    rows = cr.evaluate(tag, "jepa", 64, dns, (16,), [split], n_boot=200)
    for r in rows:
        if r["metric"] == "wake_enstrophy" and r["mode"] == "z_markov" and r["horizon"] == 16:
            return r["r2"]
    return float("nan")
for split in ("test_b", "test_c"):
    vals = [wake16(f"test1_noBN_predc0_s{s}", split) for s in (0, 1, 2)]
    vals = [v for v in vals if v == v]
    print(f"  cond-0 (frozen prod encoder) {split}: wake R2@16 z_markov = "
          f"{np.mean(vals):+.3f} +/- {np.std(vals):.3f}  (seeds: {[round(v,3) for v in vals]})")
print("  REFERENCE conditioned (production): test_b 0.449, test_c 0.325; paper 3-seed 0.46+/-0.03")
print("  REFERENCE published no-c (JEPA_d64_noc encoder + cond0): test_b 0.038")
PY
echo "=== PROD COND0 SWEEP COMPLETE ==="
