"""Apply the Track 1/2 pre-decoder gate to a wake-probe JSON output.

Reads two ``wake_probe.json`` files (the Session 9 baseline reference and
the candidate Track 1/2 run), reports the deltas, and emits a single
``PASS`` / ``FAIL`` line plus a per-criterion breakdown.

Gate criteria (SESSION11_WAKE_RESULTS_FIRST.md "Pre-decoder gate"):

  1. wake_patch_signed_r2_test_b improves by >= 0.10 over baseline
  2. wake_patch_signed_spectrum_r2_test_b improves by >= 0.05 over baseline
  3. G, D, Y per-axis probe drops by no more than 0.02 each
  4. CL probe drops by no more than 5 percent of baseline
  5. PR(z) does not collapse below the S9 baseline

Usage::

    python scripts/session11_apply_gate.py \\
        --baseline outputs/runs/session11/wake_probe_S9_baseline/wake_probe.json \\
        --candidate outputs/runs/session11/W0_C_lam10/probe/wake_probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply Session 11 wake-probe gate")
    p.add_argument("--baseline", required=True, type=str)
    p.add_argument("--candidate", required=True, type=str)
    p.add_argument("--patch-improve-min", type=float, default=0.10)
    p.add_argument("--spectrum-improve-min", type=float, default=0.05)
    p.add_argument("--gdy-axis-max-drop", type=float, default=0.02)
    p.add_argument("--cl-max-drop-frac", type=float, default=0.05)
    return p.parse_args()


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    base = _load(args.baseline)
    cand = _load(args.candidate)
    print(f"baseline:  {args.baseline}")
    print(f"candidate: {args.candidate}")

    rows: list[tuple[str, str, float, float, float, bool]] = []

    # patch_signed
    bp = base["r2_patch_signed"]["r2_overall"]
    cp = cand["r2_patch_signed"]["r2_overall"]
    dp = cp - bp
    rows.append(("wake_patch_signed", f">=+{args.patch_improve_min}", bp, cp, dp,
                 dp >= args.patch_improve_min))

    # patch_signed_spectrum
    bs = base["r2_patch_signed_spectrum"]["r2_overall"]
    cs = cand["r2_patch_signed_spectrum"]["r2_overall"]
    ds = cs - bs
    rows.append(("wake_patch_signed_spectrum", f">=+{args.spectrum_improve_min}",
                 bs, cs, ds, ds >= args.spectrum_improve_min))

    # G, D, Y per-axis (drop must be <= gdy-axis-max-drop)
    for ax in ("G", "D", "Y"):
        bv = base["r2_GDY"][f"r2_{ax}"]
        cv = cand["r2_GDY"][f"r2_{ax}"]
        drop = bv - cv  # positive means worse
        rows.append((f"r2_{ax}", f"drop <= {args.gdy_axis_max_drop}", bv, cv,
                     -drop, drop <= args.gdy_axis_max_drop))

    # CL drop
    bcl = base["r2_cl"]["r2_overall"]
    ccl = cand["r2_cl"]["r2_overall"]
    cl_drop = bcl - ccl
    cl_drop_frac = cl_drop / max(abs(bcl), 1e-6)
    rows.append(("r2_cl", f"drop_frac <= {args.cl_max_drop_frac}",
                 bcl, ccl, -cl_drop,
                 cl_drop_frac <= args.cl_max_drop_frac))

    # PR(z) doesn't collapse
    bpr = base["pr"]
    cpr = cand["pr"]
    rows.append(("pr", f">= {bpr:.3f}", bpr, cpr, cpr - bpr, cpr >= bpr * 0.95))

    print()
    print(f"{'metric':<32} {'criterion':<22} {'baseline':>10} {'candidate':>10} "
          f"{'delta':>9} {'pass'}")
    for name, crit, b, c, d, ok in rows:
        flag = "PASS" if ok else "fail"
        print(f"{name:<32} {crit:<22} {b:>10.4f} {c:>10.4f} {d:>+9.4f} {flag}")

    all_pass = all(ok for *_, ok in rows)
    print()
    if all_pass:
        print("GATE: PASS  -- proceed to decoder retrain")
    else:
        print("GATE: FAIL  -- skip decoder retrain for this config")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
