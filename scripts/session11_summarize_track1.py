"""Aggregate Track 1 sweep results into a single summary JSON.

For each completed Track 1 config (W0_A_lam03, W0_B_lam03, ...), looks for
the latest JEPA checkpoint, runs the wake-probe (if not already cached
as ``probe/wake_probe.json``), and applies the pre-decoder gate.

Output: ``outputs/runs/session11/track1_summary.json`` with one entry per
config containing baseline + candidate r2 deltas, gate verdicts, and a
flag identifying the winner (the highest patch_spectrum r2 among
gate-passing configs).

Usage::

    python scripts/session11_summarize_track1.py \\
        --baseline-probe outputs/runs/session11/wake_probe_S9_baseline/wake_probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


_CONFIGS = (
    "W0_A_lam03",
    "W0_B_lam03",
    "W0_B_lam10",
    "W0_C_lam03",
    "W0_C_lam10",
    "W0_C_lam30",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Session 11 Track 1 sweep summary")
    p.add_argument(
        "--baseline-probe",
        default="outputs/runs/session11/wake_probe_S9_baseline/wake_probe.json",
    )
    p.add_argument(
        "--track1-root",
        default="outputs/runs/session11",
    )
    p.add_argument(
        "--output",
        default="outputs/runs/session11/track1_summary.json",
    )
    p.add_argument(
        "--run-probes",
        action="store_true",
        help="Run wake-probe for any config whose probe JSON is missing. "
             "Requires PREVENT_ROOT + an available RTX 6000.",
    )
    p.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index used by --run-probes invocations.",
    )
    return p.parse_args()


def latest_checkpoint(run_dir: Path) -> Path | None:
    cands = sorted(run_dir.glob("checkpoint_iter*.pt"))
    return cands[-1] if cands else None


def ensure_probe(run_dir: Path, gpu: int, run_probes: bool) -> Path | None:
    probe_dir = run_dir / "probe"
    probe_json = probe_dir / "wake_probe.json"
    if probe_json.exists():
        return probe_json
    if not run_probes:
        return None
    ckpt = latest_checkpoint(run_dir)
    if ckpt is None:
        print(f"[summarize-t1] no checkpoint for {run_dir.name}; skipping probe")
        return None
    probe_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(REPO / "scripts" / "session11_wake_probe.py"),
        "--jepa-checkpoint", str(ckpt),
        "--output-dir", str(probe_dir),
        "--gpu", str(gpu),
        "--cl-future-deltas", "0",
    ]
    print(f"[summarize-t1] running wake-probe for {run_dir.name}")
    subprocess.run(cmd, check=True)
    return probe_json


def gate_verdict(baseline: dict, candidate: dict) -> dict:
    """Return per-criterion deltas and a single PASS/FAIL flag."""
    out: dict = {}
    bp = baseline["r2_patch_signed"]["r2_overall"]
    cp = candidate["r2_patch_signed"]["r2_overall"]
    out["dp_patch"] = cp - bp
    out["pass_patch"] = (cp - bp) >= 0.10

    bs = baseline["r2_patch_signed_spectrum"]["r2_overall"]
    cs = candidate["r2_patch_signed_spectrum"]["r2_overall"]
    out["ds_spectrum"] = cs - bs
    out["pass_spectrum"] = (cs - bs) >= 0.05

    pass_gdy = True
    for ax in ("G", "D", "Y"):
        bv = baseline["r2_GDY"][f"r2_{ax}"]
        cv = candidate["r2_GDY"][f"r2_{ax}"]
        drop = bv - cv
        out[f"drop_{ax}"] = drop
        if drop > 0.02:
            pass_gdy = False
    out["pass_gdy"] = pass_gdy

    bcl = baseline["r2_cl"]["r2_overall"]
    ccl = candidate["r2_cl"]["r2_overall"]
    out["dcl"] = ccl - bcl
    cl_drop_frac = (bcl - ccl) / max(abs(bcl), 1e-6)
    out["cl_drop_frac"] = cl_drop_frac
    out["pass_cl"] = cl_drop_frac <= 0.05

    bpr = baseline["pr"]
    cpr = candidate["pr"]
    out["dpr"] = cpr - bpr
    out["pass_pr"] = cpr >= 0.95 * bpr

    out["GATE"] = bool(
        out["pass_patch"] and out["pass_spectrum"] and out["pass_gdy"]
        and out["pass_cl"] and out["pass_pr"]
    )
    return out


def main() -> None:
    args = parse_args()
    with open(args.baseline_probe) as f:
        baseline = json.load(f)
    root = Path(args.track1_root)
    summary: dict = {
        "baseline_probe": args.baseline_probe,
        "configs": {},
    }
    for cfg in _CONFIGS:
        run_dir = root / cfg
        if not run_dir.is_dir():
            summary["configs"][cfg] = {"status": "missing"}
            continue
        probe_json = ensure_probe(run_dir, args.gpu, args.run_probes)
        if probe_json is None:
            summary["configs"][cfg] = {"status": "no_probe"}
            continue
        with open(probe_json) as f:
            cand = json.load(f)
        verdict = gate_verdict(baseline, cand)
        summary["configs"][cfg] = {
            "status": "probed",
            "probe_json": str(probe_json),
            "r2_patch_signed": cand["r2_patch_signed"]["r2_overall"],
            "r2_patch_signed_spectrum": cand["r2_patch_signed_spectrum"]["r2_overall"],
            "r2_wake_coarse_pool": cand["r2_wake_coarse_pool"]["r2_overall"],
            "r2_enstrophy_scalar": cand["r2_enstrophy_scalar"]["r2_overall"],
            "r2_cl": cand["r2_cl"]["r2_overall"],
            "r2_GDY_overall": cand["r2_GDY"]["r2_overall"],
            "pr": cand["pr"],
            "gate": verdict,
        }

    # Identify winner among gate-passers.
    passers = [
        (cfg, info) for cfg, info in summary["configs"].items()
        if isinstance(info, dict) and info.get("gate", {}).get("GATE", False)
    ]
    if passers:
        winner = max(passers, key=lambda kv: kv[1]["r2_patch_signed_spectrum"])
        summary["winner"] = winner[0]
    else:
        summary["winner"] = None
    summary["n_passing_gate"] = len(passers)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[summarize-t1] wrote {out_path}")
    print(f"[summarize-t1] n_passing_gate={summary['n_passing_gate']}, winner={summary['winner']}")
    print()
    print(f"{'config':<14} {'patch':>8} {'spec':>8} {'pool':>8} {'cl':>7} {'gdy':>7} {'pr':>6} {'gate'}")
    for cfg in _CONFIGS:
        info = summary["configs"].get(cfg, {})
        if info.get("status") != "probed":
            print(f"{cfg:<14} {info.get('status', '?'):>50}")
            continue
        g = "PASS" if info["gate"]["GATE"] else "fail"
        print(
            f"{cfg:<14} "
            f"{info['r2_patch_signed']:>8.3f} "
            f"{info['r2_patch_signed_spectrum']:>8.3f} "
            f"{info['r2_wake_coarse_pool']:>8.3f} "
            f"{info['r2_cl']:>7.3f} "
            f"{info['r2_GDY_overall']:>7.3f} "
            f"{info['pr']:>6.2f} "
            f"{g}"
        )


if __name__ == "__main__":
    main()
