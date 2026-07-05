"""Track C latent-cache builder (C1): encode train + test_b for every cell run.

Encode-only subset of ``src.evaluation.represent.run_q1``: rebuilds each frozen
encoder from its checkpoint, encodes the train and test_b splits into aligned
per-frame latent caches (``latents_<run>_{train,test_b}.npz``), and writes the
model-independent field caches once. No decode floor and no probes here -- the
Track C metrics (trackc_lift_eval, trackc_region_ssim, trackc_head_closure)
fit their own frozen-protocol readouts from these caches.

Run (RTX 6000):
    taskset -c 0-15 python -m scripts.session34.trackc_encode --gpu 0 --models <runs...>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.session34.trackc_cells import CHECKPOINT, RUNS_BASE, all_run_names  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Track C encode-only latent sweep")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=None,
                    help="Run names under outputs/runs/session34 (default: all cells).")
    ap.add_argument("--cache-dir", default="outputs/session34/trackc_latents")
    ap.add_argument("--partition", default="v2p2")
    ap.add_argument("--split", default="configs/splits/split_v2p2.json")
    ap.add_argument("--pipeline-manifest", default="outputs/data_pipeline/v2p2/manifest.json")
    ap.add_argument("--windows", default="outputs/session31/windows_v2p2.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    import torch

    from src.evaluation import rom_eval as re
    from src.utils.device import require_rtx6000

    device = require_rtx6000(gpu_index=args.gpu)
    gpu_name = torch.cuda.get_device_name(device.index)
    models = args.models or all_run_names()
    windows = re.load_windows(args.windows)
    cache_dir = re._resolve(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[trackc-encode] device={device} ({gpu_name}) models={models}", flush=True)

    t0 = time.time()
    for name in models:
        done = all(
            (cache_dir / f"latents_{name}_{s}.npz").exists() for s in ("train", "test_b")
        )
        if done:
            print(f"[trackc-encode] skip (cached): {name}", flush=True)
            continue
        run_dir = RUNS_BASE / name
        if any(name == m or name.startswith(m + "_") for m in re.REFERENCE_MODELS):
            frozen = re.load_reference_model(run_dir, CHECKPOINT, device=device)
        else:
            frozen = re.load_frozen_model(run_dir, CHECKPOINT, device=device)
        for split in ("train", "test_b"):
            enc = re.encode_split(
                frozen,
                split,
                partition=args.partition,
                split_manifest_path=args.split,
                pipeline_manifest=args.pipeline_manifest,
                windows=windows,
                device=device,
                limit=args.limit,
            )
            re.save_latents(cache_dir / f"latents_{name}_{split}.npz", enc)
            if not (cache_dir / f"fields_{split}.npz").exists():
                re.save_field_cache(cache_dir / f"fields_{split}.npz", enc)
        del frozen
        torch.cuda.empty_cache()
        print(f"[trackc-encode] done {name} ({time.time() - t0:.0f}s elapsed)", flush=True)
    print(f"[trackc-encode] all done in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
