"""Session 9 Step 3 ablation evaluation helper.

Evaluates one JEPA-based ablation checkpoint (A2 VICReg, A7 no-SS, or
similar Step 3 runs) with the same per-split metric table that
``session9_bisection_analysis.py`` produces for the bisection points.

Output: a one-row CSV ``<output_dir>/ablation_eval.csv`` with the
Test A / Test B / Test C metrics, plus a one-line stdout summary
matching the bisection's seed=0 Test B summary format.

Usage:
    python scripts/session9_evaluate_ablation.py \\
        --checkpoint outputs/runs/session9/run_a2_vicreg_only/checkpoint_iter020000.pt \\
        --code A2 --label "A2 VICReg-only" \\
        --output-dir outputs/runs/session9/run_a2_vicreg_only \\
        --gpu 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.session9_bisection_analysis import (  # noqa: E402
    build_cl_future,
    c_of,
    case_of,
    evaluate_one,
    gather_encounters,
)
from src.utils.device import require_rtx6000  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Session 9 ablation evaluation helper")
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--code", required=True, type=str,
                   help="Short code (e.g., A2, A7) for the ablation.")
    p.add_argument("--label", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--iters", type=int, default=20000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint {ckpt} not found")
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    saved_args = blob["args"]
    seed = int(saved_args.get("seed", 0))
    lam = float(saved_args.get("lambda_sigreg") or 0.0)

    run_dir = ckpt.parent
    print(f"[ablation-eval] {args.code} {args.label}", flush=True)
    print(f"[ablation-eval] checkpoint={ckpt}", flush=True)
    print(f"[ablation-eval] seed={seed} lambda_sigreg={lam} d={saved_args['d']}",
          flush=True)

    SPLITS = {s: gather_encounters(s) for s in ("test_a", "test_b", "test_c")}
    CL_RAW = {s: build_cl_future(SPLITS[s]) for s in SPLITS}
    MASK = {s: np.isfinite(CL_RAW[s]).reshape(CL_RAW[s].shape[0], -1).all(axis=1)
            for s in SPLITS}
    CL_FUTURE = {s: CL_RAW[s][MASK[s]] for s in SPLITS}
    case_of_split = {s: case_of(SPLITS[s]) for s in SPLITS}
    c_of_split = {s: c_of(SPLITS[s]) for s in SPLITS}
    T = SPLITS["test_a"][0]["omega_z"].shape[0]

    rows = evaluate_one(
        run_dir=run_dir, code=args.code, lam=lam, seed=seed, label=args.label,
        splits=SPLITS, cl_future=CL_FUTURE, mask=MASK,
        case_of_split=case_of_split, c_of_split=c_of_split,
        T=T, iters=args.iters, device=device,
    )
    df = pd.DataFrame(rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "ablation_eval.csv", index=False)
    print(f"[ablation-eval] wrote {out_dir / 'ablation_eval.csv'}", flush=True)

    test_b = df[df["split"] == "test_b"].iloc[0]
    print(f"[ablation-eval] Test B: PR_all={test_b['PR_all']:.2f} "
          f"r2_z_c={test_b['r2_z_c']:.3f} r2(CL_future)={test_b['r2_CL_future']:.3f} "
          f"r2(c, t)={test_b['r2_ct_baseline']:.3f} "
          f"delta={test_b['delta']:+.3f}", flush=True)


if __name__ == "__main__":
    main()
