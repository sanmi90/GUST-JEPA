"""Track C eight-tap EnKF filter envelope, test_b, per-cell tuned rho (C5 step 3).

Runs the frozen D220 filter protocol (K=8, osp_per_model taps, N=64 members,
stochastic, field-free init) once per Track C cell (s0), RESTRICTED to test_b
(42 encounters), with each cell's OWN tuned inflation rho (D252 per-method
precedent; read from the tuning JSON produced by track_b_freeze_tuning, falling
back to 1.0 with a warning). The CLW cell at rho=1.0 doubles as the anchor
against the session33 ``envelope_vec.json`` test_b records.

Outputs: ``outputs/session34/envelope_trackc_<cell>.json`` per cell.

Run (RTX 6000, after trackc_taps + per-cell rho tuning):
    taskset -c 0-15 python -m scripts.session34.trackc_envelope --gpu 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import scripts.session32.envelope_by_gust as env  # noqa: E402
from scripts.session34.trackc_cells import CELLS, RUNS_BASE  # noqa: E402

LATENTS_PATTERN = "outputs/session34/trackc_latents/latents_%s_train.npz"


def _testb_only_enumerate(split_path: Path) -> list[dict]:
    return [e for e in _ORIG_ENUM(split_path) if e["split"] == "test_b"]


_ORIG_ENUM = env.enumerate_encounters


def tuned_rho(tuning_path: Path, run_name: str) -> float:
    if tuning_path.exists():
        blob = json.loads(tuning_path.read_text())
        entry = blob.get(run_name)
        if isinstance(entry, dict) and "rho" in entry:
            return float(entry["rho"])
        if isinstance(entry, (int, float)):
            return float(entry)
    print(f"[trackc-env] WARNING: no tuned rho for {run_name}; using 1.0", flush=True)
    return 1.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Track C filter envelope (test_b)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--osp-taps", default="outputs/session34/osp_taps_trackc.json")
    ap.add_argument("--tuning", default="outputs/session34/filter_tuning_trackc.json")
    ap.add_argument("--out-dir", default="outputs/session34")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    taps_path = REPO_ROOT / args.osp_taps
    if not taps_path.exists():
        raise FileNotFoundError(f"{taps_path} missing; run scripts.session34.trackc_taps first.")
    tuning_path = REPO_ROOT / args.tuning

    env.enumerate_encounters = _testb_only_enumerate  # test_b restriction

    for cell in args.cells:
        run_name = CELLS[cell][0]
        out = REPO_ROOT / args.out_dir / f"envelope_trackc_{cell}.json"
        if out.exists():
            print(f"[trackc-env] skip (exists): {out.name}", flush=True)
            continue
        rho = tuned_rho(tuning_path, run_name)
        env.MODEL_RUN[run_name] = (
            str(RUNS_BASE / run_name),
            False,
            LATENTS_PATTERN,
        )
        print(f"[trackc-env] === {cell} ({run_name}) rho={rho} ===", flush=True)
        argv2 = [
            "--gpu", str(args.gpu),
            "--models", run_name,
            "--osp-taps", str(taps_path),
            "--out", str(out),
            "--seed", str(args.seed),
            "--rho", str(rho),
        ]
        if args.limit:
            argv2 += ["--limit", str(args.limit)]
        env.main(argv2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
