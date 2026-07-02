"""Encode the jepa_pool pooled latents into the Session 31 Q1 latent cache.

The Session 31 Q1 cache (``outputs/session31/q1_latents``) has the six canonical
spatial models plus the references (POD, fukami) but NOT the ``jepa_pool`` gate-E
ablation. Track O needs its pooled coefficient state, so we encode it here with
the SAME frozen-encoder path Session 31 used (``rom_eval.load_frozen_model`` ->
``encode_split`` -> ``save_latents``), writing
``latents_jepa_pool_{train,test_b}.npz`` in the identical schema.

For the pooled encoder ``encode_split`` broadcasts the (T, d) pooled latent to
the spatial grid, so the saved ``z_gap`` IS the native pooled d=32 coefficient
state (GAP of a constant map) and ``z_spatial`` is its broadcast.

Run (GPU, RTX 6000):
    taskset -c 16-23 python -m scripts.session32.encode_jepa_pool
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Encode jepa_pool latents into the Q1 cache")
    ap.add_argument("--model", default="jepa_pool")
    ap.add_argument("--runs-base", default="outputs/runs/session31")
    ap.add_argument("--checkpoint-name", default="checkpoint_iter010000.pt")
    ap.add_argument("--partition", default="v2p2")
    ap.add_argument("--split-manifest", default="configs/splits/split_v2p2.json")
    ap.add_argument("--pipeline-manifest", default="outputs/data_pipeline/v2p2/manifest.json")
    ap.add_argument("--windows", default="outputs/session31/windows_v2p2.json")
    ap.add_argument("--cache-dir", default="outputs/session31/q1_latents")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--frame-chunk", type=int, default=40)
    args = ap.parse_args(argv)

    import torch  # noqa: F401

    from src.evaluation import rom_eval as re
    from src.utils.device import require_rtx6000

    device = require_rtx6000(gpu_index=args.gpu)
    gpu_name = torch.cuda.get_device_name(device.index)
    if "RTX" not in gpu_name or "6000" not in gpu_name:
        raise RuntimeError(f"hardware policy: gpu_name={gpu_name!r} is not an RTX 6000")
    print(f"[encode] device={device} ({gpu_name}) model={args.model}", flush=True)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    windows = re.load_windows(args.windows)

    run_dir = REPO_ROOT / args.runs_base / args.model
    frozen = re.load_frozen_model(run_dir, args.checkpoint_name, device=device)
    print(
        f"[encode] latent_mode={getattr(frozen.encoder, 'latent_mode', 'spatial')} "
        f"grid={frozen.latent_grid} d={frozen.latent_dim}",
        flush=True,
    )

    for split in ("train", "test_b"):
        enc = re.encode_split(
            frozen,
            split,
            partition=args.partition,
            split_manifest_path=args.split_manifest,
            pipeline_manifest=args.pipeline_manifest,
            windows=windows,
            device=device,
            frame_chunk=args.frame_chunk,
        )
        out = cache_dir / f"latents_{args.model}_{split}.npz"
        re.save_latents(out, enc)
        print(
            f"[encode] wrote {out}  z_gap={enc.z_gap.shape} z_spatial={enc.z_spatial.shape}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
