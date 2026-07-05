"""Latent fixed-lag smoothers (Session 34, user-directed): filter vs smoother.

Two smoother arms on the flagship latents, evaluated with the identical
phase-resolved protocol as da_phase_eval (C_L in physical units + decoded-field
SSIM per pre/impact/relax phase):

1. LINEAR KF + RTS: with the spectrally-projected linear operator A, fitted
   process noise Q, and the latent-encoded observation (E_obs, H = I,
   Gamma_tilde), the model is linear-Gaussian, so the Kalman filter and the
   Rauch-Tung-Striebel smoother are closed form. Variants: filter only,
   full-interval RTS (offline reanalysis upper bound), and fixed-lag RTS with
   lag L in {5, 10} frames (deployable with a 0.25-0.5 t/c delay).

2. REX-EnKS: fixed-lag ensemble Kalman smoother wrapped around the
   deployment-clean REX-EnKF: at each analysis, the previous L member states
   are also updated through their cross-covariance with the current predicted
   observation (standard EnKS lag update).

The smoother targets the information-poor relaxation phase (future taps
refine past analyses).

Run (RTX 6000): taskset -c 8-15 python -m scripts.session34.da_smoother --gpu 1
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

from scripts.session34.da_phase_eval import cl_metrics, phase_masks  # noqa: E402
from scripts.session34.lae_enkf_pilot import (  # noqa: E402
    delay_embed,
    encounters,
    fit_linear_A,
    load_aligned,
)
from scripts.session34.latent_rex import LatentRex  # noqa: E402

BAND_TO_SIGMA = 2.5631


def kf_rts(z_obs: np.ndarray, A: np.ndarray, Q: np.ndarray, Gam: np.ndarray,
           t_init: int, lag: int | None) -> np.ndarray:
    """Closed-form Kalman filter + (fixed-lag) RTS smoother, H = I.

    lag = None -> filter only; lag = -1 -> full-interval RTS;
    lag = L > 0 -> fixed-lag smoothing (state at t uses obs up to t + L).
    """
    T, n = z_obs.shape
    xa = np.empty((T, n)); Pa = np.empty((T, n, n))
    xf = np.empty((T, n)); Pf = np.empty((T, n, n))
    xa[: t_init + 1] = z_obs[: t_init + 1]
    Pa[: t_init + 1] = Gam
    I = np.eye(n)
    for t in range(t_init + 1, T):
        xf[t] = A @ xa[t - 1]
        Pf[t] = A @ Pa[t - 1] @ A.T + Q
        K = Pf[t] @ np.linalg.inv(Pf[t] + Gam)
        xa[t] = xf[t] + K @ (z_obs[t] - xf[t])
        Pa[t] = (I - K) @ Pf[t]
    if lag is None:
        return xa
    xs = xa.copy()
    if lag == -1:                                   # full RTS backward pass
        for t in range(T - 2, t_init, -1):
            C = Pa[t] @ A.T @ np.linalg.inv(Pf[t + 1])
            xs[t] = xa[t] + C @ (xs[t + 1] - xf[t + 1])
        return xs
    # fixed-lag: for each t, run the backward pass from min(t+lag, T-1) to t
    for t in range(t_init + 1, T - 1):
        e = min(t + lag, T - 1)
        x_next = xa[e]
        for s in range(e - 1, t - 1, -1):
            C = Pa[s] @ A.T @ np.linalg.inv(Pf[s + 1])
            x_next = xa[s] + C @ (x_next - xf[s + 1])
        xs[t] = x_next
    return xs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="latent fixed-lag smoothers")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--model", default="jepa_pool_vec")
    ap.add_argument("--cache-dir", default="outputs/session33/q1_vec_latents")
    ap.add_argument("--pressure-dir", default="outputs/session31/q1_latents")
    ap.add_argument("--taps", default="outputs/session33/osp_taps_vec.json")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--delay", type=int, default=10)
    ap.add_argument("--members", type=int, default=64)
    ap.add_argument("--band-scale", type=float, default=4.0)
    ap.add_argument("--enks-lag", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/session34/da_smoother.json")
    args = ap.parse_args(argv)

    from src.data.nearbody_observables import get_nearbody_band
    from src.data.omega_pipeline import ssim_data_range
    from src.data.wake_observables import get_wake_mask_tensor
    from src.evaluation.represent import fit_linear_probe
    from src.models.decoder import SpatialLatentFieldDecoder
    from src.utils.device import require_rtx6000
    from scripts.session34.trackc_region_ssim import masked_ssim_per_frame

    device = require_rtx6000(gpu_index=args.gpu)
    rng = np.random.default_rng(args.seed)
    t0c = time.time()

    tr = load_aligned(REPO_ROOT / args.cache_dir, REPO_ROOT / args.pressure_dir,
                      args.model, "train")
    tb = load_aligned(REPO_ROOT / args.cache_dir, REPO_ROOT / args.pressure_dir,
                      args.model, "test_b")
    encs_tr, encs_tb = encounters(tr), encounters(tb)
    n = tr["z"].shape[1]
    taps = np.asarray(json.loads((REPO_ROOT / args.taps).read_text())
                      [args.model][f"K{args.k}"], dtype=int)
    p_mu = tr["p"][:, taps].mean(axis=0)
    p_sd = tr["p"][:, taps].std(axis=0) + 1e-9

    A, Q_lin, _, _ = fit_linear_A(tr, encs_tr, args.alpha)
    X_tr = np.concatenate([delay_embed((tr["p"][e["rows"]][:, taps] - p_mu) / p_sd,
                                       args.delay) for e in encs_tr])
    Zt_tr = np.concatenate([tr["z"][e["rows"]] for e in encs_tr])
    W = np.linalg.solve(X_tr.T @ X_tr + args.alpha * np.eye(X_tr.shape[1]),
                        X_tr.T @ Zt_tr)
    Gamma = np.cov((Zt_tr - X_tr @ W).T) + 1e-8 * np.eye(n)
    chol_G = np.linalg.cholesky(Gamma)
    probe = fit_linear_probe(tr["z"], tr["cl"])
    rex = LatentRex(d=n, horizon=40)
    rex.load_state_dict(torch.load(
        REPO_ROOT / "outputs/session34/latent_rex_model.pt", map_location="cpu"))
    rex.to(device).eval()

    fields_tb = np.load(REPO_ROOT / "outputs/session34/trackc_latents/fields_test_b.npz",
                        allow_pickle=True)
    fkey = {k: i for i, k in enumerate(zip(fields_tb["case_id"].tolist(),
                                           fields_tb["encounter_index"].tolist(),
                                           fields_tb["frame"].tolist()))}
    omega_tb = fields_tb["omega_norm"].astype(np.float32)
    decoder = SpatialLatentFieldDecoder(latent_dim=n, feature_h=24, feature_w=12)
    decoder.load_state_dict(torch.load(
        REPO_ROOT / f"outputs/session34/trackc_decoders/decoder_{args.model}.pt",
        map_location="cpu"))
    decoder.to(device).eval()
    ssim_L = float(ssim_data_range(str(REPO_ROOT / "outputs/data_pipeline/v2p2/manifest.json")))
    band = get_nearbody_band()
    masks = {"nearbody": band > 0, "wake": get_wake_mask_tensor().numpy() > 0,
             "full": np.ones_like(band, dtype=bool)}
    tile = lambda z: np.repeat(np.repeat(z[:, :, None, None], 24, 2), 12, 3) \
        .astype(np.float32)

    @torch.no_grad()
    def decode(z_seq):
        outs = []
        for i in range(0, z_seq.shape[0], 128):
            zb = torch.from_numpy(tile(z_seq[i:i + 128])).to(device)
            outs.append(decoder(zb).float().squeeze(1).cpu().numpy())
        return np.concatenate(outs)

    @torch.no_grad()
    def rex_step(ctx_np):
        out = rex(torch.from_numpy(ctx_np[:, -30:]).float().to(device))
        s0 = out[:, 0].cpu().numpy()
        med = s0[..., s0.shape[-1] // 2]
        sig = np.clip((s0[..., -1] - s0[..., 0]) / BAND_TO_SIGMA * args.band_scale,
                      1e-4, None)
        return med, sig

    N, L_enks = args.members, args.enks_lag
    t_init = args.delay - 1
    arms = ("kf", "rts_full", "rts_lag5", "rts_lag10", "rex_enkf", "rex_enks")
    per_est: dict[str, list] = {a: [] for a in arms}

    for e in encs_tb:
        rows = e["rows"]
        z_true, cl_true = tb["z"][rows], tb["cl"][rows]
        wmask = tb["wmask"][rows]
        T = rows.size
        pmask = phase_masks(wmask, t_init)
        pt = (tb["p"][rows][:, taps] - p_mu) / p_sd
        z_obs = delay_embed(pt, args.delay) @ W

        est_z = {
            "kf": kf_rts(z_obs, A, Q_lin, Gamma, t_init, None),
            "rts_full": kf_rts(z_obs, A, Q_lin, Gamma, t_init, -1),
            "rts_lag5": kf_rts(z_obs, A, Q_lin, Gamma, t_init, 5),
            "rts_lag10": kf_rts(z_obs, A, Q_lin, Gamma, t_init, 10),
        }

        # REX-EnKF + fixed-lag EnKS in one pass
        ctx = np.repeat(z_obs[None, :t_init + 1], N, axis=0)
        ctx[:, -1] += rng.standard_normal((N, n)) @ chol_G.T
        zA = np.empty((T, n)); zA[:t_init + 1] = z_obs[:t_init + 1]
        zS = zA.copy()
        hist: list[np.ndarray] = []                    # member states, last L_enks
        hist_t: list[int] = []
        for t in range(t_init + 1, T):
            med, sig = rex_step(ctx)
            zf = med + rng.standard_normal((N, n)) * sig
            dZ = zf - zf.mean(0, keepdims=True)
            P = dZ.T @ dZ / (N - 1)
            K_g = P @ np.linalg.inv(P + Gamma)
            pert = rng.standard_normal((N, n)) @ chol_G.T
            innov = z_obs[t][None] + pert - zf
            za = zf + innov @ K_g.T
            # EnKS lag update: correct stored past members with the same innovation
            for j, zpast in enumerate(hist):
                dPast = zpast - zpast.mean(0, keepdims=True)
                C = dPast.T @ dZ / (N - 1)             # cov(z_{t-l}, zf_t)
                Kp = C @ np.linalg.inv(P + Gamma)
                hist[j] = zpast + innov @ Kp.T
                zS[hist_t[j]] = hist[j].mean(0)
            zA[t] = za.mean(0)
            zS[t] = za.mean(0)
            hist.append(za.copy()); hist_t.append(t)
            if len(hist) > L_enks:
                hist.pop(0); hist_t.pop(0)
            ctx = np.concatenate([ctx, za[:, None]], axis=1)[:, -30:]
        est_z["rex_enkf"] = zA
        est_z["rex_enks"] = zS

        frames_idx = np.array([fkey[(e["case_id"], e["k"], int(f))]
                               for f in tb["frame"][rows]])
        truth_fields = omega_tb[frames_idx]
        for name, z_est in est_z.items():
            cl_hat = probe.predict(z_est)
            rec: dict = {"case_id": e["case_id"], "encounter_index": e["k"]}
            dec_fields = decode(z_est)
            for ph, m in pmask.items():
                if m.sum() < 3:
                    continue
                rec[ph] = cl_metrics(cl_true[m], cl_hat[m])
                pf = masked_ssim_per_frame(dec_fields[m], truth_fields[m], ssim_L, masks)
                rec[ph]["ssim"] = {k: float(v.mean()) for k, v in pf.items()}
            ip = np.nonzero(pmask["impact"])[0]
            if ip.size:
                tp = ip[np.argmax(np.abs(cl_true[ip]))]
                pp = ip[np.argmax(np.abs(cl_hat[ip]))]
                rec["peak"] = {
                    "peak_rel_error_pct": float(100 * abs(cl_hat[pp] - cl_true[tp])
                                                / max(abs(cl_true[tp]), 1e-6)),
                    "peak_timing_error_tc": float((pp - tp) * 0.05),
                }
            per_est[name].append(rec)

    def agg(recs, ph, key, sub=None):
        vals = [r[ph][key] if sub is None else r[ph][key][sub] for r in recs if ph in r]
        return float(np.median(vals))

    summary = {}
    for name, recs in per_est.items():
        summary[name] = {ph: {
            "cl_r2": agg(recs, ph, "r2"), "cl_rmse": agg(recs, ph, "rmse"),
            "ssim_nearbody": agg(recs, ph, "ssim", "nearbody"),
            "ssim_full": agg(recs, ph, "ssim", "full"),
        } for ph in ("pre", "impact", "relax")}
        summary[name]["peak_rel_error_pct_median"] = float(np.median(
            [r["peak"]["peak_rel_error_pct"] for r in recs if "peak" in r]))
        s = summary[name]
        print(f"[smoother] {name:10s} impact RMSE={s['impact']['cl_rmse']:.3f} "
              f"relax RMSE={s['relax']['cl_rmse']:.3f} "
              f"pre RMSE={s['pre']['cl_rmse']:.3f} "
              f"peak%={s['peak_rel_error_pct_median']:.1f} "
              f"| ssim nb imp={s['impact']['ssim_nearbody']:.3f}", flush=True)

    out = REPO_ROOT / args.out
    out.write_text(json.dumps({"protocol": vars(args), "summary": summary,
                               "records": per_est}, indent=1, default=float))
    print(f"[smoother] wrote {out} in {time.time() - t0c:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
