"""Session 18 B1: Fukami AE reconstruction-quality verification gate.

Per SESSION18_B1_PROTOCOL.md, a trained Fukami AE must pass at least one of:

    Test A SSIM_mean >= 0.60          (Fukami's reconstruction criterion)
    Test A ratio_mean < 2.0           (MSE within 2x of per-case noise floor)

Either threshold passing is acceptable. A trained model that fails BOTH is
not usable as a baseline for downstream predictor training; the protocol
requires the implementation to be debugged before proceeding.

Exit status:
    0 if the gate passes (or the model is exactly at the threshold).
    1 if the gate fails.
    2 if the eval JSON cannot be read or is malformed.

Usage:
    python scripts/session18/verify_fukami_gate.py \\
        --eval-json outputs/session18/exp_b1/fukami_ae_d3/final_eval.json \\
        --d 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SSIM_MIN = 0.60
RATIO_MAX = 2.0


def main() -> int:
    p = argparse.ArgumentParser(description="Verify Fukami AE B1 gate.")
    p.add_argument("--eval-json", type=Path, required=True)
    p.add_argument("--d", type=int, required=True)
    p.add_argument(
        "--ssim-min",
        type=float,
        default=SSIM_MIN,
        help=f"Minimum Test A SSIM_mean (default {SSIM_MIN}).",
    )
    p.add_argument(
        "--ratio-max",
        type=float,
        default=RATIO_MAX,
        help=f"Maximum Test A ratio_mean = MSE/floor (default {RATIO_MAX}).",
    )
    args = p.parse_args()

    if not args.eval_json.exists():
        print(f"[gate] FAIL: eval JSON missing: {args.eval_json}", file=sys.stderr)
        return 2

    try:
        with open(args.eval_json) as f:
            ev = json.load(f)
    except Exception as exc:
        print(f"[gate] FAIL: cannot read {args.eval_json}: {exc}", file=sys.stderr)
        return 2

    ta = ev.get("test_a", {})
    ssim = float(ta.get("ssim_mean", float("nan")))
    ratio = float(ta.get("ratio_mean", float("nan")))
    floor = float(ta.get("floor_mean", float("nan")))
    mse = float(ta.get("mse_mean", float("nan")))
    n = int(ta.get("n_encounters", 0))

    print(f"[gate] Fukami AE d={args.d}")
    print(f"[gate]   Test A: n={n}  MSE_mean={mse:.4f}  floor_mean={floor:.4f}")
    print(f"[gate]   Test A: SSIM_mean={ssim:.4f}  ratio_mean={ratio:.4f}")

    ssim_pass = ssim >= args.ssim_min
    ratio_pass = ratio < args.ratio_max
    overall = ssim_pass or ratio_pass

    print(
        f"[gate]   SSIM gate (>= {args.ssim_min}): "
        f"{'PASS' if ssim_pass else 'FAIL'}"
    )
    print(
        f"[gate]   Ratio gate (< {args.ratio_max}): "
        f"{'PASS' if ratio_pass else 'FAIL'}"
    )
    print(f"[gate] OVERALL: {'PASS' if overall else 'FAIL'}")

    if not overall:
        print(
            "[gate] action: debug the Fukami AE implementation; do NOT "
            "proceed to predictor training with this checkpoint.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
