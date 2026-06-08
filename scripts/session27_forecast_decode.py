"""Decoded forecast for the UNCONDITIONED (tf-no-c) JEPA, rolling its OWN predictor.

Encode DNS omega up to impact, roll the JEPA's own (cond_dim=0) transformer
predictor autoregressively in latent space (markov seed = impact frame only;
fullctx seed = pre-impact history window), decode the rolled latents with the
decoder trained on the frozen unconditioned encoder, and compare the predicted
omega fields to DNS. No (G,D,Y) is given to anything.

Reports SSIM (normalised, registry data-range) + wake-region normalised MSE-ratio
at horizons, for reconstruction (z_dns ceiling), markov forecast, and fullctx
forecast, aggregated over test_b/test_c. Writes a Fig-3-style decoded-forecast
panel for a representative moderate-G held-out (test_b) encounter (median error,
not best/worst).

Usage: python scripts/session27_forecast_decode.py --device cuda:2 --decoder-iter 30000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "session17"))
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import exp2_rollout_decode_metrics as E2  # noqa: E402
from src.data.omega_pipeline import OmegaPipeline, ssim_data_range  # noqa: E402
from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.models.predictor import AutoregressivePredictor, LSTMLatentPredictor  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402
import figstyle  # noqa: E402

HMAX = 24
HORIZONS = [0, 8, 16, 24]


def load_jepa(device, enc_ckpt):
    """Predictor-agnostic loader: builds the encoder + the JEPA's OWN predictor
    (transformer or lstm, cond_dim=0) from the checkpoint args."""
    blob = torch.load(enc_ckpt, map_location="cpu", weights_only=False)
    a = blob["args"]; d = int(a["d"]); T = int(a.get("T", 32))
    ptype = a.get("predictor_type", "transformer")
    sd = blob["jepa_state_dict"]
    enc = HybridCNNViTEncoder(latent_dim=d, projection_norm=a.get("projection_norm", "batchnorm"))
    enc.load_state_dict({k.removeprefix("encoder."): v for k, v in sd.items() if k.startswith("encoder.")}, strict=False)
    if ptype == "lstm":
        pred = LSTMLatentPredictor(d, cond_dim=0, hidden_dim=int(a["predictor_hidden"]),
                                   num_layers=int(a["predictor_layers"]), max_seq_len=T)
    else:
        pred = AutoregressivePredictor(latent_dim=d, cond_dim=int(a.get("predictor_cond_dim", 0)), max_seq_len=T)
    pred.load_state_dict({k.removeprefix("predictor."): v for k, v in sd.items() if k.startswith("predictor.")}, strict=True)
    enc.eval().to(device); pred.eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    for p in pred.parameters():
        p.requires_grad_(False)
    print(f"[forecast] loaded {enc_ckpt.parent.name}: ptype={ptype} d={d} T={T}", flush=True)
    return enc, pred, ptype


def wake_mask():
    xx = E2.X_GRID[:, None]; yy = E2.Y_GRID[None, :]
    return ((xx >= E2.WAKE_X_MIN) & (xx <= E2.WAKE_X_MAX) & (np.abs(yy) <= E2.WAKE_Y_MAX)).astype(np.float32)


def mse_ratio(pred_n, truth_n, m):
    num = float((((pred_n - truth_n) ** 2) * m).sum())
    den = float(((truth_n ** 2) * m).sum())
    return num / max(den, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default="noc_tf")
    ap.add_argument("--decoder-iter", type=int, default=30000)
    ap.add_argument("--splits", nargs="+", default=["test_b", "test_c"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    device = torch.device(args.device) if args.device else require_rtx6000(gpu_index=0)

    E2.ENCODER_CKPT = REPO / f"outputs/session27/JEPA_d64_{args.tag}/checkpoint_iter020000.pt"
    E2.DECODER_CKPT = REPO / f"outputs/session27/decoder_{args.tag}/decoder_iter{args.decoder_iter:06d}.pt"
    enc, pred, ptype = load_jepa(device, E2.ENCODER_CKPT)
    pipe = OmegaPipeline.from_manifest(E2.OMEGA_MANIFEST)
    try:
        L = float(ssim_data_range(str(E2.OMEGA_MANIFEST)))
    except Exception:
        L = 8.31
    have_dec = E2.DECODER_CKPT.exists()
    print(f"[forecast] device={device} decoder={'FOUND' if have_dec else 'MISSING'} ssim_L={L:.2f}", flush=True)
    cond0 = torch.zeros(1, 0, device=device)
    m = wake_mask()

    dec = E2.load_decoder(device, torch.load(E2.ENCODER_CKPT, map_location="cpu", weights_only=False)["args"]) if have_dec else None

    @torch.no_grad()
    def decode_sel(z_sel):  # (k, d) latents -> (k, 192, 96) raw omega (robust reshape, no squeeze loop)
        with torch.autocast("cuda", torch.bfloat16):
            out = dec(z_sel.unsqueeze(0).to(device))
        if isinstance(out, dict):
            out = out["pred"]
        out = out.float().reshape(-1, 192, 96).cpu().numpy()
        return np.stack([pipe.unnormalize(out[i]) for i in range(out.shape[0])])

    agg = {(mode, h): [] for mode in ("recon", "markov", "fullctx") for h in HORIZONS}
    agg_r = {(mode, h): [] for mode in ("recon", "markov", "fullctx") for h in HORIZONS}
    cand = []
    nproc = 0

    for split in args.splits:
        encs = E2.gather_split_encounters(split)
        if args.limit:
            encs = encs[:args.limit]
        for e in encs:
            with h5py.File(e["path"], "r") as h:
                omega_raw = np.asarray(h["omega_z"], np.float32)
                impact = int(h.attrs.get("impact_frame_estimate", E2.DEFAULT_IMPACT_FRAME))
            if impact + HMAX >= omega_raw.shape[0]:
                continue
            proc = pipe.preprocess_raw(omega_raw, e["case_id"], e["k"])
            omega_norm = pipe.normalize(proc)
            z_dns = E2.encode_full(enc, omega_norm, device)                      # (120, d)
            if ptype == "transformer":
                z_m = E2.rollout_markov_only(pred, z_dns[impact:impact + 1], cond0, HMAX, device)[0]
            else:  # lstm: windowed free-run from the single impact frame
                z_m = E2.rollout_autoregressive(pred, z_dns[impact:impact + 1], cond0, HMAX, device)[0]
            lo = max(0, impact + 1 - int(pred.max_seq_len)); seed = z_dns[lo:impact + 1]
            z_f = E2.rollout_autoregressive(pred, seed, cond0, HMAX, device)[0]
            if not have_dec:
                if e is encs[0]:
                    print(f"  [rollout-smoke {split}] |z_dns|={z_dns.norm(dim=1).mean():.2f} "
                          f"|z_m[-1]|={z_m[-1].norm():.2f} |z_f[-1]|={z_f[-1].norm():.2f}", flush=True)
                continue
            sl = seed.shape[0]
            # decode ONLY the horizon frames we score (not all 120) -- big speedup
            rec_sel = decode_sel(z_dns[[impact + h for h in HORIZONS]])
            dm_sel = decode_sel(z_m[[h for h in HORIZONS]])
            df_sel = decode_sel(z_f[[sl - 1 + h for h in HORIZONS]])
            for hi, h in enumerate(HORIZONS):
                t_n = pipe.normalize(proc[impact + h])
                for mode, fld in (("recon", rec_sel[hi]), ("markov", dm_sel[hi]), ("fullctx", df_sel[hi])):
                    p_n = pipe.normalize(fld)
                    agg[(mode, h)].append(float(ssim(t_n, p_n, data_range=L)))
                    agg_r[(mode, h)].append(mse_ratio(p_n, t_n, m))
            nproc += 1
            if nproc % 10 == 0:
                print(f"  [{split}] processed {nproc} encounters", flush=True)
            if split == "test_b" and 1.0 <= abs(e["G"]) <= 2.0:
                cand.append((e, proc, impact, rec_sel, df_sel))

    print(f"\n=== unconditioned JEPA decoded forecast (own predictor), {args.tag} ===")
    print(f"{'horizon':>8} | {'recon SSIM':>11} {'markov SSIM':>12} {'fullctx SSIM':>13} | "
          f"{'mk MSEratio':>11} {'fc MSEratio':>11}")
    for h in HORIZONS:
        def mn(d, k): return float(np.mean(d[k])) if d[k] else float("nan")
        print(f"{h:>8} | {mn(agg,('recon',h)):>11.3f} {mn(agg,('markov',h)):>12.3f} "
              f"{mn(agg,('fullctx',h)):>13.3f} | {mn(agg_r,('markov',h)):>11.3f} {mn(agg_r,('fullctx',h)):>11.3f}")

    if cand:
        # representative = candidate whose fullctx SSIM@16 is closest to the median over candidates
        H16I = HORIZONS.index(16)
        s16 = []
        for (e, proc, impact, rec_sel, df_sel) in cand:
            t = pipe.normalize(proc[impact + 16]); p = pipe.normalize(df_sel[H16I])
            s16.append(ssim(t, p, data_range=L))
        target = np.median(s16); pick = int(np.argmin(np.abs(np.array(s16) - target)))
        e, proc, impact, rec_sel, df_sel = cand[pick]
        rows = [("DNS", [proc[impact + h] for h in HORIZONS]),
                ("JEPA forecast (own pred, fullctx)", [df_sel[hi] for hi in range(len(HORIZONS))]),
                ("decode(z_dns) ceiling", [rec_sel[hi] for hi in range(len(HORIZONS))])]
        fig, axes = plt.subplots(len(rows), len(HORIZONS), figsize=(2.0 * len(HORIZONS), 2.0 * len(rows)))
        for ri, (label, flds) in enumerate(rows):
            for ci, (h, fld) in enumerate(zip(HORIZONS, flds)):
                ax = axes[ri, ci]
                figstyle.vort_panel(ax, pipe.normalize(fld))
                if ri == 0:
                    ax.set_title(f"impact+{h}", fontsize=8)
                if ci == 0:
                    ax.set_ylabel(label, fontsize=7)
        fig.suptitle(f"Unconditioned JEPA decoded forecast | {e['case_id']} enc{e['k']} (G={e['G']:+.2f})", fontsize=9)
        fig.tight_layout()
        out = REPO / f"outputs/session27/forecast_decode_{args.tag}.pdf"
        fig.savefig(out, bbox_inches="tight"); print(f"\n[forecast] figure -> {out}  (pick SSIM@16={s16[pick]:.3f}, median={target:.3f})")


if __name__ == "__main__":
    main()
