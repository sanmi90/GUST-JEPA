"""ROM field-forecast metric: decode a model's rolled latent at horizon h and
measure SSIM/MSE of the predicted field vs the DNS field (Solera-Rico style).

For a (decoder, rollout, latent) triple on the v2p1 split, this rolls a model's
latent forward, DECODES each predicted latent to a field, and reports how well
the PREDICTED field matches the true DNS field as a function of forecast horizon.
The SSIM and the raw-scale evaluation convention are mirrored exactly from
``scripts/session9_train_decoder.py::evaluate_split`` (the same path whose
``ssim_mean`` the decoder summaries report), so the h=0 reconstruction SSIM is a
direct sanity check against ``decoder_summary.json``'s test_b ``ssim_mean``.

Decoder loading and DNS-omega loading mirror
``scripts/session17/exp2_rollout_decode_metrics.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.models.lap_film_decoder import LapFiLMDecoder  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402

# Same SSIM as the decoder summaries (Fukami arXiv:2305.18394 Eq.1; hardcoded
# c1=0.16, c2=1.44 on RAW-scale omega, NOT the data-range-dependent variant).
from scripts.session9_train_decoder import _ssim  # noqa: E402


OMEGA_MANIFEST = REPO / "outputs" / "data_pipeline" / "v2p1" / "manifest.json"
SPLIT_MANIFEST = REPO / "configs" / "splits" / "split_v2p1.json"
PARTITION = "v2p1"
HORIZONS = (1, 2, 4, 8, 12, 16, 24, 32)

CACHE_ROOT = Path(
    os.environ.get(
        "VORTEX_JEPA_CACHE",
        str(
            Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
            / "data"
            / "processed"
            / "vortex-jepa"
        ),
    )
)

OUT = REPO / "outputs" / "session29" / "rom_field_forecast"
OUT.mkdir(parents=True, exist_ok=True)


def load_decoder(device: torch.device, decoder_ckpt: Path) -> tuple[LapFiLMDecoder, dict]:
    """Reconstruct a LapFiLMDecoder from a decoder checkpoint blob.

    Mirrors exp2_rollout_decode_metrics.load_decoder: rebuild the channel taper
    from decoder_base_ch and load decoder_state_dict.
    """
    blob = torch.load(decoder_ckpt, map_location="cpu", weights_only=False)
    dargs = blob["args"]
    if not isinstance(dargs, dict):
        dargs = vars(dargs)
    sd = blob["decoder_state_dict"]
    # latent_dim is not stored in args; infer from the init projection weight
    # (LapFiLMDecoder init_proj maps latent_dim -> first feature map).
    if "init_proj.weight" in sd:
        latent_dim = int(sd["init_proj.weight"].shape[1])
    else:
        latent_dim = int(dargs.get("latent_dim") or 64)
    bc = int(dargs.get("decoder_base_ch", 64))
    channels = (bc, bc, int(bc * 0.75), int(bc * 0.5), int(bc * 0.375))
    fb = dargs.get("decoder_fourier_bands")
    dec = LapFiLMDecoder(
        latent_dim=latent_dim,
        channels=channels,
        resblocks_per_level=int(dargs.get("decoder_resblocks_per_level", 2)),
        upsample=dargs.get("decoder_upsample", "pixelshuffle"),
        fourier_bands=int(fb) if fb is not None else 4,
        use_film=bool(dargs.get("decoder_use_film", True)),
        airfoil_mask_path=dargs.get("airfoil_mask_path"),
    )
    dec.load_state_dict(sd)
    dec.eval().to(device)
    for p in dec.parameters():
        p.requires_grad_(False)
    return dec, dargs


@torch.no_grad()
def decode_latents(
    z: np.ndarray, dec: LapFiLMDecoder, pipeline: OmegaPipeline, device: torch.device
) -> np.ndarray:
    """Decode z of shape (n, d) to raw-scale omega (n, 192, 96).

    Mirrors evaluate_split: dec(z) with z as (1, n, d); take ['pred'] if dict;
    unnormalize back to raw scale; squeeze to (n, H, W).
    """
    zt = torch.from_numpy(z.astype(np.float32)).unsqueeze(0).to(device)  # (1, n, d)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = dec(zt)
        if isinstance(out, dict):
            out = out["pred"]
        out = pipeline.unnormalize(out)
    out = out.float().squeeze(0)  # (n, 1, H, W) or (n, H, W)
    while out.dim() > 3:
        out = out.squeeze(1)
    arr = out.cpu().numpy()
    return arr.reshape(-1, 192, 96)


def gather_test_b() -> dict[tuple[str, int], Path]:
    """Map (case_id, encounter_index) -> cached v2p1 encounter HDF5 path for test_b."""
    with open(SPLIT_MANIFEST) as f:
        manifest = json.load(f)
    out: dict[tuple[str, int], Path] = {}
    for cid, case in manifest["cases"].items():
        if case["split"] != "test_b":
            continue
        for k in range(int(case["n_encounters_full"])):
            path = CACHE_ROOT / PARTITION / cid / f"encounter_{int(k):02d}.h5"
            if path.exists():
                out[(cid, int(k))] = path
    return out


def load_raw_omega(path: Path, pipeline: OmegaPipeline, case_id: str, k: int) -> np.ndarray:
    """Load cached omega and apply pipeline preprocess_raw (mask + clip), RAW scale.

    Same target side as evaluate_split: preprocess_raw (no normalize); metrics
    are computed on this raw cleaned target.
    """
    with h5py.File(path, "r") as f:
        omega_raw = np.asarray(f["omega_z"], dtype=np.float32)
    return pipeline.preprocess_raw(omega_raw, case_id, int(k))


def run_one(
    decoder_ckpt: Path,
    rollout_npz: Path,
    latent_npz: Path,
    device: torch.device,
) -> dict:
    pipeline = OmegaPipeline.from_manifest(OMEGA_MANIFEST)
    dec, _ = load_decoder(device, decoder_ckpt)

    roll = np.load(rollout_npz, allow_pickle=True)
    lat = np.load(latent_npz, allow_pickle=True)
    z_roll = np.asarray(roll["z_full"], dtype=np.float32)          # (n,120,d) ROLLED
    z_enc = np.asarray(lat["z_full"], dtype=np.float32)            # (n,120,d) ENCODED
    impact = np.asarray(roll["impact_frame"]).astype(int)          # (n,)
    case_ids = [str(x) for x in roll["case_ids"]]
    enc_idx = np.asarray(roll["encounter_indices"]).astype(int)

    # The latent dir is the decoder's training/eval space; align its rows to the
    # rollout's (case_id, encounter) ordering so the encoded h=0 anchor matches.
    lat_cids = [str(x) for x in lat["case_ids"]]
    lat_eidx = np.asarray(lat["encounter_indices"]).astype(int)
    lat_lookup = {(c, int(e)): i for i, (c, e) in enumerate(zip(lat_cids, lat_eidx))}

    cache = gather_test_b()
    n = z_roll.shape[0]

    # Accumulators: SSIM/MSE summed over encounters per horizon (+ h=0 anchor).
    ssim_sum = {h: 0.0 for h in (0,) + HORIZONS}
    mse_sum = {h: 0.0 for h in (0,) + HORIZONS}
    count = {h: 0 for h in (0,) + HORIZONS}

    for i in range(n):
        cid, k = case_ids[i], int(enc_idx[i])
        if (cid, k) not in cache:
            continue
        imp = int(impact[i])
        omega_raw = load_raw_omega(cache[(cid, k)], pipeline, cid, k)  # (T,H,W) raw
        T = omega_raw.shape[0]

        # h=0 anchor: decode the ENCODED latent at the impact frame (reconstruction).
        li = lat_lookup.get((cid, k))
        if li is not None and 0 <= imp < T:
            z0 = z_enc[li, imp][None, :]  # (1,d)
            om0 = decode_latents(z0, dec, pipeline, device)[0]  # (H,W) raw
            ssim_sum[0] += _ssim(omega_raw[imp], om0)
            mse_sum[0] += float(((omega_raw[imp] - om0) ** 2).mean())
            count[0] += 1

        # Forecast horizons: decode the ROLLED latent at impact+h.
        valid_h = [h for h in HORIZONS if imp + h < min(T, z_roll.shape[1])]
        if not valid_h:
            continue
        frames = [imp + h for h in valid_h]
        z_sel = z_roll[i, frames]  # (len(valid_h), d)
        om_pred = decode_latents(z_sel, dec, pipeline, device)  # (len, H, W) raw
        for j, h in enumerate(valid_h):
            true = omega_raw[imp + h]
            pred = om_pred[j]
            ssim_sum[h] += _ssim(true, pred)
            mse_sum[h] += float(((true - pred) ** 2).mean())
            count[h] += 1

    result = {"per_horizon": {}}
    for h in (0,) + HORIZONS:
        if count[h] > 0:
            result["per_horizon"][str(h)] = {
                "ssim": ssim_sum[h] / count[h],
                "mse": mse_sum[h] / count[h],
                "n": count[h],
            }
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="ROM field-forecast metric (Solera-Rico style).")
    p.add_argument("--decoder", required=True, type=str, help="path to decoder_iter*.pt")
    p.add_argument("--rollout", required=True, type=str, help="rollout tag under lowd_rollouts/")
    p.add_argument("--latents", required=True, type=str, help="latent dir tag under session28/latents/")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = require_rtx6000(gpu_index=args.gpu)
    print(f"[rom] device={device}  gpu_name={torch.cuda.get_device_name(device.index)}")

    decoder_ckpt = Path(args.decoder)
    if not decoder_ckpt.is_absolute():
        decoder_ckpt = REPO / decoder_ckpt
    rollout_npz = REPO / "outputs" / "session29" / "lowd_rollouts" / args.rollout / "test_b.npz"
    latent_npz = REPO / "outputs" / "session28" / "latents" / args.latents / "test_b.npz"

    for label, pth in (("decoder", decoder_ckpt), ("rollout", rollout_npz), ("latents", latent_npz)):
        if not pth.exists():
            raise FileNotFoundError(f"{label} not found: {pth}")
    print(f"[rom] decoder={decoder_ckpt}")
    print(f"[rom] rollout={rollout_npz}")
    print(f"[rom] latents={latent_npz}")

    t0 = time.time()
    result = run_one(decoder_ckpt, rollout_npz, latent_npz, device)
    print(f"[rom] done in {time.time() - t0:.1f}s\n")

    print(f"{'h':>4}  {'SSIM':>8}  {'MSE':>12}  {'n':>4}")
    for h_str, rec in result["per_horizon"].items():
        tag = " (recon)" if h_str == "0" else ""
        print(f"{h_str:>4}  {rec['ssim']:>8.4f}  {rec['mse']:>12.4f}  {rec['n']:>4}{tag}")

    result["meta"] = {
        "decoder": str(decoder_ckpt),
        "rollout": str(rollout_npz),
        "latents": str(latent_npz),
        "horizons": list(HORIZONS),
        "split": "test_b",
        "partition": PARTITION,
        "omega_manifest": str(OMEGA_MANIFEST),
        "ssim_convention": "Fukami c1=0.16 c2=1.44 on raw-scale omega (matches decoder_summary)",
    }
    out_path = OUT / f"{args.rollout}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[rom] wrote {out_path}")


if __name__ == "__main__":
    main()
