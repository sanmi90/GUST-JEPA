"""Session 28 / T9 visualization-decoder operating-point selection + decode.

STEP 1 (--select): honour the FROZEN decoder operating-point rule
(PROTOCOL.md Section 6; eval_protocol_v2p1.yaml decoders.operating_point):
per family, the saved checkpoint (every 2000 iters) with maximum validation
(test_a) SSIM_mean; the SAME rule for all families. The SSIM is the trainer's
own evaluate_split (Fukami SSIM definition; raw-scale via the v2p1 omega
pipeline), run on the FULL test_a (87 encounters), not the cheap subset8 the
trainer logs to decoder_metrics.jsonl. Writes operating_points.json.

STEP 2 (--decode): at the per-family operating point, decode rolled latents to
omega_z fields on test_b and test_c:
  recon    = decode(z_dns)   (representational decode-ceiling reference)
  forecast = decode(z_full)  (full-context autoregressive rollout; markov RETIRED)
For tf-no-c / fukami / pod the rollout NPZ already exist under
outputs/session28/rollouts/<key>/. For lstm-no-c no rollout NPZ exists, so we
roll its OWN cond_dim=0 predictor (full-context) from the encoded latents, per
scripts/session27_gen_rollouts_uncond.py.

Saves outputs/session28/decode/<family>/{recon,forecast}_{test_b,test_c}.npz.
To keep size bounded we save the impact window [25, 55] (31 frames) AND the
forecast horizons H in {0, 8, 16, 24} relative to impact (HORIZONS). float32.

HARD RULE: every GPU call targets the RTX 6000 at require_rtx6000(gpu_index=1)
(maps to torch cuda:3 / nvidia-smi index 1 on this box). Never the L40S.

Usage:
    python scripts/session28/decode_operating_points.py --select
    python scripts/session28/decode_operating_points.py --decode
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "session17"))

import session9_train_decoder as DEC  # noqa: E402
import exp2_rollout_decode_metrics as E2  # noqa: E402
from src.data.omega_pipeline import OmegaPipeline, ssim_data_range  # noqa: E402
from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.models.predictor import AutoregressivePredictor, LSTMLatentPredictor  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402

RUNS = REPO / "outputs" / "runs" / "session28"
LAT = REPO / "outputs" / "session28" / "latents"
ROLL = REPO / "outputs" / "session28" / "rollouts"
OUTDIR = REPO / "outputs" / "session28" / "decode"
SPLIT = REPO / "configs" / "splits" / "split_v2p1.json"
PARTITION = "v2p1"
OMEGA_MANIFEST = REPO / "outputs" / "data_pipeline" / "v2p1" / "manifest.json"
ITERS = list(range(2000, 30001, 2000))

# Impact-window + horizon save policy (documented in README).
WIN_LO, WIN_HI = 25, 56  # frames [25, 55] inclusive (31 frames)
HORIZONS = [0, 8, 16, 24]

# Family registry. mode: "encoder" (JEPA, frozen encoder -> z) or "npz" (post-hoc,
# latent_store). rollout_key names the rollouts/<key> dir holding z_dns / z_full
# (None => must be rolled from the family's own predictor; lstm case).
FAMILIES = {
    "dec_jepa_tf_noc_d64_s42": {
        "mode": "encoder",
        "encoder_ckpt": RUNS / "jepa_tf_noc_d64_s42" / "encoder" / "checkpoint_iter020000.pt",
        "rollout_key": "jepa_d64",
        "latents_dir": None,
    },
    "dec_jepa_lstm_noc_d64_s42": {
        "mode": "encoder",
        "encoder_ckpt": RUNS / "jepa_lstm_noc_d64_s42" / "encoder" / "checkpoint_iter020000.pt",
        "rollout_key": None,  # no rollout NPZ; roll own predictor (full-context)
        "latents_dir": LAT / "jepa_lstm_noc_d64_s42",
    },
    "dec_posthoc_fukami_d64": {
        "mode": "npz",
        "encoder_ckpt": None,
        "rollout_key": "fukami_d64",
        "latents_dir": LAT / "fukami_d64_s42",
    },
    "dec_posthoc_pod_d64": {
        "mode": "npz",
        "encoder_ckpt": None,
        "rollout_key": "pod_d64",
        "latents_dir": LAT / "pod_d64",
    },
}


def load_pipeline():
    pipe = OmegaPipeline.from_manifest(OMEGA_MANIFEST)
    try:
        L = float(ssim_data_range(str(OMEGA_MANIFEST)))
    except Exception:
        L = 8.45
    return pipe, L


def build_decoder_from_ckpt(ckpt_path: Path, latent_dim: int, device):
    """Reconstruct the LapFiLM decoder from a checkpoint's stored args + state."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    da = blob["args"]
    ns = argparse.Namespace(**da)
    dec = DEC.build_decoder(ns, latent_dim=latent_dim, device=device, spatial_init=False)
    dec.load_state_dict(blob["decoder_state_dict"])
    dec.eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    return dec


def load_frozen_encoder(ckpt_path: Path, device):
    enc, d = DEC.load_encoder(ckpt_path, device)
    return enc, d


# ---------------------------------------------------------------------------
# STEP 1: operating-point selection
# ---------------------------------------------------------------------------
def select(device):
    pipe, L = load_pipeline()
    print(f"[select] device={device} gpu={torch.cuda.get_device_name(device.index)} "
          f"ssim_L={L:.2f}", flush=True)
    # Point the trainer's gather helpers at v2p1.
    DEC.SPLIT_PATH = SPLIT
    DEC.CACHE_PARTITION = PARTITION
    test_a_encs = DEC.gather_eval_encounters("test_a")
    print(f"[select] full test_a: {len(test_a_encs)} encounters", flush=True)

    out = {
        "rule": ("Per family, the saved checkpoint (every 2000 iters) with maximum "
                 "validation (test_a) SSIM_mean; the SAME rule for all families. "
                 "SSIM = trainer evaluate_split (Fukami def, raw-scale via v2p1 omega "
                 "pipeline) on the FULL test_a; ssim_data_range registry split_v2p1."),
        "frozen_protocol": "scripts/session28/PROTOCOL.md Section 6 / eval_protocol_v2p1.yaml decoders.operating_point",
        "ssim_data_range": L,
        "n_test_a_encounters": len(test_a_encs),
        "iters_evaluated": ITERS,
        "families": {},
    }

    for fam, cfg in FAMILIES.items():
        fam_dir = RUNS / fam
        # Frozen encoder OR latent store(s) for this family.
        enc = None
        latent_store_a = None
        if cfg["mode"] == "encoder":
            enc, d = load_frozen_encoder(cfg["encoder_ckpt"], device)
        else:
            latent_store_a = DEC.LatentStore(cfg["latents_dir"] / "test_a.npz")
            d = latent_store_a.d
        ssims = []
        best = (-1.0, None)
        for it in ITERS:
            ckpt = fam_dir / f"decoder_iter{it:06d}.pt"
            dec = build_decoder_from_ckpt(ckpt, latent_dim=d, device=device)
            ev = DEC.evaluate_split(
                enc, dec, test_a_encs, device,
                omega_scale=1.0, omega_pipeline=pipe,
                latent_store=latent_store_a,
            )
            s = float(ev["ssim_mean"])
            ssims.append(s)
            if s > best[0]:
                best = (s, it)
            print(f"[select] {fam} iter{it}: test_a ssim_mean={s:.5f} "
                  f"mse_mean={ev['mse_mean']:.4f}", flush=True)
            del dec
            torch.cuda.empty_cache()
        best_ssim, best_iter = best
        # Sanity: is SSIM still rising at 30k?
        rising = bool(ssims[-1] >= max(ssims) - 1e-9)
        out["families"][fam] = {
            "best_iter": int(best_iter),
            "best_val_ssim": float(best_ssim),
            "all_iters_ssim": [float(x) for x in ssims],
            "iters": ITERS,
            "still_rising_at_30k": rising,
            "ssim_30k": float(ssims[-1]),
            "ssim_data_range": L,
        }
        print(f"[select] *** {fam}: best_iter={best_iter} best_val_ssim={best_ssim:.5f} "
              f"(30k={ssims[-1]:.5f}, still_rising={rising}) ***", flush=True)
        if enc is not None:
            del enc
            torch.cuda.empty_cache()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    op_path = OUTDIR / "operating_points.json"
    op_path.write_text(json.dumps(out, indent=2))
    print(f"[select] wrote {op_path}", flush=True)
    return out


# ---------------------------------------------------------------------------
# STEP 2: decode at the operating point
# ---------------------------------------------------------------------------
def load_lstm_predictor(encoder_ckpt: Path, device):
    blob = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
    a = blob["args"]
    sd = blob["jepa_state_dict"]
    psd = {k.removeprefix("predictor."): v for k, v in sd.items() if k.startswith("predictor.")}
    pred = LSTMLatentPredictor(
        int(a["d"]), cond_dim=0, hidden_dim=int(a["predictor_hidden"]),
        num_layers=int(a["predictor_layers"]), max_seq_len=int(a.get("T", 32)),
    )
    pred.load_state_dict(psd, strict=True)
    pred.eval().to(device)
    for p in pred.parameters():
        p.requires_grad_(False)
    return pred


@torch.no_grad()
def roll_lstm_fullctx(pred, z_dns_np, impact_np, device):
    """Full-context rollout of the LSTM predictor (cond_dim=0). Matches
    scripts/session27_gen_rollouts_uncond.py: z_full[:impact+1] = z_dns,
    z_full[impact+1:] = autoregressive free-run from the pre-impact window."""
    n, T, d = z_dns_np.shape
    z_dns = torch.from_numpy(z_dns_np).to(device)
    z_full = z_dns.clone()
    cond0 = torch.zeros(1, 0, device=device)
    msl = int(pred.max_seq_len)
    for i in range(n):
        ti = int(impact_np[i])
        steps = T - ti - 1
        if steps <= 0:
            continue
        lo = max(0, ti + 1 - msl)
        seed = z_dns[i, lo:ti + 1]
        zfu = E2.rollout_autoregressive(pred, seed, cond0, steps, device)
        z_full[i, ti + 1:T] = zfu[0, -steps:].float()
    return z_full.cpu().numpy()


@torch.no_grad()
def decode_latents(dec, z_seq_np, pipe, device, batch=16):
    """z_seq_np: (k, d) -> (k, 192, 96) raw-scale omega (pipeline-unnormalized)."""
    z = torch.from_numpy(z_seq_np.astype(np.float32))
    T = z.shape[0]
    out = np.zeros((T, 192, 96), dtype=np.float32)
    for b in range(0, T, batch):
        zb = z[b:b + batch].unsqueeze(0).to(device)  # (1, B, d)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            o = dec(zb)
        if isinstance(o, dict):
            o = o["pred"]
        o = o.float().reshape(-1, 192, 96).cpu().numpy()
        out[b:b + o.shape[0]] = o
    # Unnormalize each frame back to raw omega scale.
    return np.stack([pipe.unnormalize(out[i]) for i in range(out.shape[0])])


def get_rollout(fam, cfg, split, device):
    """Return dict with z_dns, z_full, case_ids, encounter_indices, impact_frame."""
    if cfg["rollout_key"] is not None:
        p = ROLL / cfg["rollout_key"] / f"{split}.npz"
        b = np.load(p, allow_pickle=True)
        return {
            "z_dns": b["z_dns"].astype(np.float32),
            "z_full": b["z_full"].astype(np.float32),
            "case_ids": b["case_ids"],
            "encounter_indices": b["encounter_indices"].astype(np.int64),
            "impact_frame": b["impact_frame"].astype(np.int64),
            "source": str(p),
        }
    # lstm: roll own predictor full-context from encoded latents.
    latp = cfg["latents_dir"] / f"{split}.npz"
    b = np.load(latp, allow_pickle=True)
    z_dns = b["z_full"].astype(np.float32)  # encoded per-frame latents = z_dns
    impact = b["impact_frame"].astype(np.int64)
    pred = load_lstm_predictor(cfg["encoder_ckpt"], device)
    z_full = roll_lstm_fullctx(pred, z_dns, impact, device)
    del pred
    torch.cuda.empty_cache()
    return {
        "z_dns": z_dns,
        "z_full": z_full,
        "case_ids": b["case_ids"],
        "encounter_indices": b["encounter_indices"].astype(np.int64),
        "impact_frame": impact,
        "source": f"rolled-own-lstm-fullctx from {latp}",
    }


def dns_truth_ref():
    """Path template for the DNS truth omega (per-encounter cache files)."""
    return ("${VORTEX_JEPA_CACHE}/v2p1/<case_id>/encounter_<k:02d>.h5 :: omega_z, "
            "pipeline-preprocessed (mask+clip) before comparison")


def decode_family(fam, cfg, op_iter, device, pipe):
    fam_short = (fam.replace("dec_jepa_", "").replace("dec_posthoc_", "")
                 .replace("_d64_s42", "").replace("_d64", ""))
    fam_out = OUTDIR / fam_short
    fam_out.mkdir(parents=True, exist_ok=True)
    ckpt = RUNS / fam / f"decoder_iter{op_iter:06d}.pt"
    # Latent dim from the rollout/latent arrays.
    written = []
    for split in ("test_b", "test_c"):
        roll = get_rollout(fam, cfg, split, device)
        d = roll["z_dns"].shape[2]
        dec = build_decoder_from_ckpt(ckpt, latent_dim=d, device=device)
        n, T, _ = roll["z_dns"].shape
        impact = roll["impact_frame"]
        win_idx = np.arange(WIN_LO, WIN_HI)  # frames [25, 55]
        for which, zkey in (("recon", "z_dns"), ("forecast", "z_full")):
            z = roll[zkey]  # (n, 120, d) absolute-frame indexed
            # Impact window block: (n, len(win), 192, 96)
            win_fields = np.zeros((n, win_idx.size, 192, 96), dtype=np.float32)
            # Horizon block: (n, len(HORIZONS), 192, 96) at impact + H
            hor_fields = np.zeros((n, len(HORIZONS), 192, 96), dtype=np.float32)
            hor_abs_frames = np.zeros((n, len(HORIZONS)), dtype=np.int64)
            for i in range(n):
                win_fields[i] = decode_latents(dec, z[i, win_idx], pipe, device)
                hframes = np.array([min(impact[i] + h, T - 1) for h in HORIZONS], dtype=np.int64)
                hor_abs_frames[i] = hframes
                hor_fields[i] = decode_latents(dec, z[i, hframes], pipe, device)
            out_npz = fam_out / f"{which}_{split}.npz"
            np.savez_compressed(
                out_npz,
                omega_decoded_window=win_fields,        # (n, 31, 192, 96)
                window_frames=win_idx.astype(np.int64),  # absolute frames [25..55]
                omega_decoded_horizon=hor_fields,        # (n, 4, 192, 96)
                horizon_offsets=np.array(HORIZONS, dtype=np.int64),
                horizon_abs_frames=hor_abs_frames,       # (n, 4) absolute frame index
                impact_frame=impact,
                case_ids=roll["case_ids"],
                encounter_indices=roll["encounter_indices"],
                decoder_iter=np.int64(op_iter),
                family=fam,
                endpoint=which,
                split=split,
                latent_source=roll["source"],
                rollout_zkey=zkey,
                dns_truth_ref=dns_truth_ref(),
                omega_scale="raw (pipeline-unnormalized); pipeline=outputs/data_pipeline/v2p1/manifest.json",
            )
            written.append((str(out_npz), win_fields.shape, hor_fields.shape))
            print(f"[decode] {fam} {which} {split}: "
                  f"window{win_fields.shape} horizon{hor_fields.shape} -> {out_npz.name}",
                  flush=True)
        del dec
        torch.cuda.empty_cache()
    return written


def decode_all(device):
    pipe, L = load_pipeline()
    op_path = OUTDIR / "operating_points.json"
    op = json.loads(op_path.read_text())
    print(f"[decode] device={device} gpu={torch.cuda.get_device_name(device.index)}", flush=True)
    all_written = {}
    for fam, cfg in FAMILIES.items():
        op_iter = int(op["families"][fam]["best_iter"])
        print(f"[decode] === {fam} @ operating iter {op_iter} ===", flush=True)
        all_written[fam] = decode_family(fam, cfg, op_iter, device, pipe)
    return all_written


def write_numbers_part(op):
    """eval_all.py PART format: each family's operating iter + val SSIM, macro-bound."""
    fam_macro = {
        "dec_jepa_tf_noc_d64_s42": "JepaTf",
        "dec_jepa_lstm_noc_d64_s42": "JepaLstm",
        "dec_posthoc_fukami_d64": "Fukami",
        "dec_posthoc_pod_d64": "Pod",
    }
    numbers = {}
    for fam, info in op["families"].items():
        mf = fam_macro[fam]
        numbers[f"decode_op_iter_{fam}"] = {
            "macro": f"DecOpIter{mf}",
            "value": int(info["best_iter"]),
            "fmt": "%d",
            "split": "test_a",
            "source": "decode_operating_points.py",
            "note": (f"T9 visualization decoder operating point: argmax full-test_a "
                     f"SSIM_mean over 15 checkpoints; frozen rule (PROTOCOL.md S6). "
                     f"30k SSIM={info['ssim_30k']:.4f}, still_rising_at_30k="
                     f"{info['still_rising_at_30k']}"),
        }
        numbers[f"decode_op_val_ssim_{fam}"] = {
            "macro": f"DecOpValSsim{mf}",
            "value": float(info["best_val_ssim"]),
            "fmt": "%.3f",
            "split": "test_a",
            "endpoint": "representational",
            "source": "decode_operating_points.py",
            "note": (f"val (test_a) SSIM_mean at the operating point iter "
                     f"{info['best_iter']}; Fukami SSIM def, raw-scale via v2p1 omega "
                     f"pipeline, n_enc={op['n_test_a_encounters']}, "
                     f"ssim_data_range={op['ssim_data_range']:.2f}"),
        }
    part = {"part": "decode_operating_points", "numbers": numbers}
    pp = REPO / "outputs" / "session28" / "numbers_parts" / "decode_operating_points.json"
    pp.write_text(json.dumps(part, indent=2))
    print(f"[numbers] wrote {pp} ({len(numbers)} numbers)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--select", action="store_true", help="Step 1: operating-point selection.")
    ap.add_argument("--decode", action="store_true", help="Step 2: decode at the operating point.")
    ap.add_argument("--numbers", action="store_true", help="Write the eval_all numbers part.")
    args = ap.parse_args()
    if not (args.select or args.decode or args.numbers):
        ap.error("pass at least one of --select / --decode / --numbers")
    device = require_rtx6000(gpu_index=1)
    name = torch.cuda.get_device_name(device.index)
    assert "RTX" in name and "6000" in name, f"refusing non-RTX6000 device: {name}"
    print(f"[main] enforced device={device} ({name})", flush=True)
    if args.select:
        select(device)
    if args.decode:
        decode_all(device)
    if args.numbers:
        op = json.loads((OUTDIR / "operating_points.json").read_text())
        write_numbers_part(op)


if __name__ == "__main__":
    main()
