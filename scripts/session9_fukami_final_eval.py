"""Run the final Fukami evaluation (MSE / SSIM / eps) on a saved checkpoint.

Mirrors what session9_train_fukami's evaluate_split does at the end of
a complete training, but loads from a checkpoint so an early-stopped
run can still produce the final_eval.json + log lines.

Usage:
    python scripts/session9_fukami_final_eval.py
        --checkpoint outputs/runs/session9/run_a11_fukami_pipeline_v1/checkpoint_iter006000.pt
        --output-dir outputs/runs/session9/run_a11_fukami_pipeline_v1
        --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from scripts.session9_fukami_evaluation import load_fukami  # noqa: E402
from scripts.session9_train_fukami import (  # noqa: E402
    evaluate_split,
    gather_eval_encounters,
)
from src.utils.device import require_rtx6000  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fukami AE final evaluation on a checkpoint")
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wrapper = load_fukami(Path(args.checkpoint), device)
    print(f"[fukami-eval] loaded {args.checkpoint}", flush=True)
    pipe = getattr(wrapper, "omega_pipeline", None)
    if pipe is not None:
        print(f"  pipeline mask cells: {int(pipe.mask.sum().item())}", flush=True)
        print(f"  pipeline thresholds: {sum(len(v) for v in pipe.thresholds.values())} encs",
              flush=True)
        print(f"  pipeline train_stats: mean={pipe.train_stats.mean:.4f}, "
              f"std={pipe.train_stats.std:.4f}", flush=True)

    encs_a = gather_eval_encounters("test_a")
    encs_b = gather_eval_encounters("test_b")
    encs_c = gather_eval_encounters("test_c")
    print(f"[fukami-eval] test_a={len(encs_a)}, test_b={len(encs_b)}, test_c={len(encs_c)}",
          flush=True)

    ev_a = evaluate_split(wrapper, encs_a, device)
    ev_b = evaluate_split(wrapper, encs_b, device)
    ev_c = evaluate_split(wrapper, encs_c, device)

    summary = {
        "checkpoint": str(args.checkpoint),
        "test_a": ev_a,
        "test_b": ev_b,
        "test_c": ev_c,
    }
    out_path = out_dir / "final_eval.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[fukami-eval] wrote {out_path}", flush=True)
    for split, ev in (("test_a", ev_a), ("test_b", ev_b), ("test_c", ev_c)):
        print(f"  {split}: MSE_mean={ev['mse_mean']:.4f} "
              f"floor={ev['floor_mean']:.4f} ratio={ev['ratio_mean']:.3f} "
              f"SSIM_mean={ev['ssim_mean']:.4f} "
              f"eps_per_frame_mean={ev['eps_per_frame_mean']:.4f} "
              f"eps_volume_mean={ev['eps_volume_mean']:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
