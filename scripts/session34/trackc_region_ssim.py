"""Track C region-masked SSIM on the near-body support (C3).

Per cell (s0 only -- decoders are the expensive stage), refits the shared
decode-floor decoder (``represent.fit_decode_floor_decoder``, 6000 steps, the
identical-capacity field readout used for the Q1 decode floor), decodes test_b,
and computes per-frame SSIM maps (skimage ``structural_similarity`` with
``full=True``, Wang constants, data_range = pinned v2p2 L) averaged over three
supports:

- ``nearbody``: the sign-symmetric lift-element band (``w > 0``),
- ``wake``: the wake ROI mask (reference),
- ``full``: the whole frame (reference; matches the Q1 decode-floor SSIM).

Decoder state dicts are persisted under ``outputs/session34/trackc_decoders/``
so the decoded fields are reproducible without refitting.

Run (RTX 6000, after the C1 latent sweep):
    taskset -c 0-15 python -m scripts.session34.trackc_region_ssim --gpu 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.session34.trackc_cells import CELLS  # noqa: E402
from src.data.nearbody_observables import get_nearbody_band  # noqa: E402
from src.data.omega_pipeline import ssim_data_range  # noqa: E402
from src.data.wake_observables import get_wake_mask_tensor  # noqa: E402
from src.evaluation.represent import fit_decode_floor_decoder  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402


def masked_ssim_means(
    pred: np.ndarray, true: np.ndarray, data_range: float, masks: dict[str, np.ndarray]
) -> dict[str, float]:
    """Per-frame SSIM maps averaged over each mask's support, then over frames."""
    from skimage.metrics import structural_similarity

    sums = {k: 0.0 for k in masks}
    for i in range(pred.shape[0]):
        _, smap = structural_similarity(
            true[i].astype(np.float64),
            pred[i].astype(np.float64),
            data_range=float(data_range),
            full=True,
        )
        for k, m in masks.items():
            sums[k] += float(smap[m].mean())
    n = max(pred.shape[0], 1)
    return {k: v / n for k, v in sums.items()}


def decode_in_batches(decoder, z_spatial: np.ndarray, device, batch: int = 64) -> np.ndarray:
    outs = []
    decoder.eval()
    with torch.no_grad():
        for i in range(0, z_spatial.shape[0], batch):
            zb = torch.from_numpy(z_spatial[i : i + batch]).float().to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                pred = decoder(zb)
            outs.append(pred.float().squeeze(1).cpu().numpy())
    return np.concatenate(outs, axis=0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Track C region SSIM")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--cache-dir", default="outputs/session34/trackc_latents")
    ap.add_argument("--decoder-dir", default="outputs/session34/trackc_decoders")
    ap.add_argument("--out", default="outputs/session34/trackc_region_ssim.json")
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--seed", type=int, default=0, help="Cell seed to evaluate (default s0).")
    ap.add_argument("--decoder-steps", type=int, default=6000)
    ap.add_argument("--pipeline-manifest", default="outputs/data_pipeline/v2p2/manifest.json")
    args = ap.parse_args(argv)

    device = require_rtx6000(gpu_index=args.gpu)
    cache_dir = REPO_ROOT / args.cache_dir
    dec_dir = REPO_ROOT / args.decoder_dir
    dec_dir.mkdir(parents=True, exist_ok=True)
    ssim_L = float(ssim_data_range(str(REPO_ROOT / args.pipeline_manifest)))

    band = get_nearbody_band()
    masks = {
        "nearbody": band > 0,
        "wake": get_wake_mask_tensor().numpy() > 0,
        "full": np.ones_like(band, dtype=bool),
    }

    fields_tr = np.load(cache_dir / "fields_train.npz", allow_pickle=True)
    fields_tb = np.load(cache_dir / "fields_test_b.npz", allow_pickle=True)
    omega_tr = fields_tr["omega_norm"].astype(np.float32)
    omega_tb = fields_tb["omega_norm"].astype(np.float32)

    results: dict[str, dict] = {}
    t0 = time.time()
    for cell in args.cells:
        run_name = CELLS[cell][args.seed]
        z_tr = np.load(cache_dir / f"latents_{run_name}_train.npz")["z_spatial"]
        z_tb_npz = np.load(cache_dir / f"latents_{run_name}_test_b.npz", allow_pickle=True)
        z_tb = z_tb_npz["z_spatial"]
        grid = tuple(int(x) for x in z_tb_npz["latent_grid"])
        print(f"[region-ssim] {cell} ({run_name}): fitting decoder "
              f"({args.decoder_steps} steps)", flush=True)
        decoder = fit_decode_floor_decoder(
            z_tr, omega_tr, grid, device=device,
            steps=args.decoder_steps, seed=args.seed, verbose=False,
        )
        torch.save(decoder.state_dict(), dec_dir / f"decoder_{run_name}.pt")
        pred_tb = decode_in_batches(decoder, z_tb, device)
        vals = masked_ssim_means(pred_tb, omega_tb, ssim_L, masks)
        results[cell] = {"run_name": run_name, "ssim": vals, "n_frames": int(pred_tb.shape[0])}
        print(f"[region-ssim] {cell}: " + "  ".join(
            f"{k}={v:.4f}" for k, v in vals.items()), flush=True)
        del decoder
        torch.cuda.empty_cache()

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "protocol": {
            "decoder": f"fit_decode_floor_decoder steps={args.decoder_steps} seed={args.seed}",
            "ssim_L": ssim_L,
            "masks": {k: int(m.sum()) for k, m in masks.items()},
            "split": "test_b",
        },
        "results": results,
    }, indent=1))
    print(f"[region-ssim] wrote {out_path} in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
