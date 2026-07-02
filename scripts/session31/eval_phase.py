"""Session 31 Track D Q2-phase: phase-resolved forecast eval entrypoint.

Thin CLI over :func:`src.evaluation.rollout.run_q2_phase`. Using the gust impact
instant as the reference, every Test B (anchor, horizon) sample is bucketed by
which phase its TARGET frame ``a+h`` falls in -- pre_impact (lead_in),
impact, or post_impact (relaxation) -- and field VRMSE + observable merit are
computed PER phase (pooled over horizons 1..H, plus a fixed h=8 split). The
matched ResUNet predictor and decode-floor decoder are fit ONCE per model,
identical to the Q2 harness; only the sample bucketing changes. Writes
``outputs/session31/q2_phase.json`` (the canonical q*.json are untouched).

RTX 6000 only (require_rtx6000 inside run_q2_phase). Cap CPU per the hardware
rule when launching:

    taskset -c 0-7 python scripts/session31/eval_phase.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.rollout import CANONICAL_MODELS, run_q2_phase  # noqa: E402

DEFAULT_MODELS = list(CANONICAL_MODELS) + ["fukami", "fukami_wake", "pod"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Session 31 Q2 phase-resolved forecast eval")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--runs-base", default="outputs/runs/session31")
    p.add_argument("--checkpoint", default="checkpoint_iter010000.pt")
    p.add_argument("--cache-dir", default="outputs/session31/q1_latents")
    p.add_argument("--out", default="outputs/session31/q2_phase.json")
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--horizon-train", type=int, default=8)
    p.add_argument("--ref-horizon", type=int, default=8)
    p.add_argument("--min-samples", type=int, default=10)
    p.add_argument("--predictor-steps", type=int, default=4000)
    p.add_argument("--predictor-batch", type=int, default=64)
    p.add_argument("--predictor-lr", type=float, default=5e-4)
    p.add_argument("--windows-tag", default="anchored_local")
    p.add_argument("--windows", default="outputs/session31/windows_v2p2.json")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--decoder-steps", type=int, default=6000)
    p.add_argument("--decoder-batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device is None and os.environ.get("VORTEX_JEPA_ROM_DEVICE"):
        args.device = os.environ["VORTEX_JEPA_ROM_DEVICE"]
    run_q2_phase(
        models=args.models,
        runs_base=args.runs_base,
        checkpoint_name=args.checkpoint,
        cache_dir=args.cache_dir,
        out_json=args.out,
        horizon=args.horizon,
        horizon_train=args.horizon_train,
        ref_horizon=args.ref_horizon,
        min_samples=args.min_samples,
        predictor_steps=args.predictor_steps,
        predictor_batch=args.predictor_batch,
        predictor_lr=args.predictor_lr,
        windows_tag=args.windows_tag,
        windows_path=args.windows,
        gpu=args.gpu,
        device_override=args.device,
        decoder_steps=args.decoder_steps,
        decoder_batch=args.decoder_batch,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
