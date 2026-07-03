"""Frozen-filter gust-intensity envelope for the D250 flagship (jepa_pool_vec).

Runs the SAME frozen protocol as scripts/session32/envelope_by_gust.py (D220:
rho=1.0, K=8, osp_per_model, N=64, stochastic, field-free init) on the vec
retrain, whose training predictor is the vector AutoregressivePredictor
(HANDOFF D250). Nothing is tuned; the taps file comes from vec_o1_recovery.py
(frozen session32 staircases merged with the fresh jepa_pool_vec staircase),
so run that first.

Output: outputs/session33/envelope_vec.json (same schema as envelope_by_gust).

Run (RTX 6000):
    taskset -c 0-15 python -m scripts.session33.vec_envelope --gpu 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import scripts.session32.envelope_by_gust as env  # noqa: E402

VEC_MODELS = {
    "jepa_pool_vec": (
        "outputs/runs/session33/jepa_pool_vec",
        False,
        "outputs/session33/q1_vec_latents/latents_%s_train.npz",
    ),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description="frozen-filter envelope for jepa_pool_vec (D250)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=list(VEC_MODELS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--osp-taps", default="outputs/session33/osp_taps_vec.json")
    ap.add_argument("--out", default="outputs/session33/envelope_vec.json")
    args = ap.parse_args(argv)

    taps_path = REPO_ROOT / args.osp_taps
    if not taps_path.exists():
        raise FileNotFoundError(
            f"{taps_path} missing; run scripts.session33.vec_o1_recovery first "
            "(it builds the merged taps file)."
        )

    env.MODEL_RUN.update(VEC_MODELS)
    argv2 = [
        "--gpu", str(args.gpu),
        "--models", *args.models,
        "--osp-taps", str(taps_path),
        "--out", args.out,
        "--seed", str(args.seed),
    ]
    if args.limit:
        argv2 += ["--limit", str(args.limit)]
    return env.main(argv2)


if __name__ == "__main__":
    raise SystemExit(main())
