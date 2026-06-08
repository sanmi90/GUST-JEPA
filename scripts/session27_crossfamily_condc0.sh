#!/usr/bin/env bash
# Fairness check for the predictability claim under an UNCONDITIONED predictor.
# Train the cond-0 transformer (same architecture/recipe as the paper's predictor,
# minus the gust c) on each encoder family's frozen latents, 3 seeds, and score
# wake-enstrophy closure (canonical exp_closure_r2). Compare to the conditioned
# baselines (JEPA 0.449, Fukami -0.48, POD -0.09) and to JEPA cond-0 (0.473+/-0.074).
# If JEPA cond-0 still dominates POD/Fukami cond-0, the forward-closure advantage is
# a property of the REPRESENTATION, robust to dropping the conditioning.
#   bash scripts/session27_crossfamily_condc0.sh [gpu]
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
source "$REPO/.venv/bin/activate"
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}" WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
GPU="${1:-0}"

for fam in pod_d64_noBN fukami_d64_noBN; do
  FAMLAT="outputs/session18/exp_b1/latents_${fam}"
  for seed in 0 1 2; do
    tag="${fam}_predc0_s${seed}"
    PREDIR="outputs/session27/predictor_${tag}"
    PRED="${PREDIR}/checkpoint_iter020000.pt"
    ROLL="outputs/session18/exp_b1_test3/rollouts_${tag}"
    ln -sfn "$REPO/${FAMLAT}" "outputs/session18/exp_b1/latents_${tag}"
    echo ">>> [${tag}] train cond-0 predictor on frozen ${fam} encoder"
    [[ -f "$PRED" ]] || python scripts/session18/train_baseline_predictor.py \
      --latents-dir "$FAMLAT" --tag "$tag" --output-dir "$PREDIR" \
      --no-output-bn --cond-dim 0 --gpu "$GPU" --seed "$seed" --num-workers 2
    echo ">>> [${tag}] rollouts"
    [[ -f "$ROLL/test_b.npz" ]] || python scripts/session18/eval_baseline_rollouts.py \
      --latents-dir "$FAMLAT" --predictor "$PRED" --tag "$tag" --output-dir "$ROLL" --gpu "$GPU"
  done
done

echo ">>> closure R^2 (canonical exp_closure_r2)"
python - <<'PY'
import sys, numpy as np
from pathlib import Path
REPO = Path.cwd(); sys.path.insert(0, str(REPO / "scripts" / "session20"))
import exp_closure_r2 as cr
dns = np.load(cr.DNS_METRICS_PATH, allow_pickle=True)
def wake16(tag, kind, split):
    rows = cr.evaluate(tag, kind, 64, dns, (16,), [split], n_boot=100)
    for r in rows:
        if r["metric"] == "wake_enstrophy" and r["mode"] == "z_markov" and r["horizon"] == 16:
            return r["r2"]
    return float("nan")
print("\n=== UNCONDITIONED (cond-0) wake R2@16, matched predictor across families ===")
for fam, kind in (("pod_d64_noBN", "pod"), ("fukami_d64_noBN", "fukami")):
    for split in ("test_b", "test_c"):
        vals = [wake16(f"{fam}_predc0_s{s}", kind, split) for s in (0, 1, 2)]
        vals = [v for v in vals if v == v]
        print(f"  {fam:16s} {split}: {np.mean(vals):+.3f} +/- {np.std(vals):.3f}  {[round(v,3) for v in vals]}")
print("\n  REFERENCE cond-0 JEPA: test_b 0.473+/-0.074, test_c -0.26+/-0.40")
print("  REFERENCE conditioned: JEPA 0.449, Fukami -0.48, POD -0.09 (test_b wake)")
PY
echo "=== CROSS-FAMILY COND0 COMPLETE ==="
