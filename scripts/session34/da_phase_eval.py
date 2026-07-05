"""Phase-resolved DA evaluation: C_L (physical units) + decoded-field SSIM (Session 34).

User-directed: judge the assimilation BEFORE impact, THROUGH impact and AFTER
impact, on both the load signal and the flow field, in physical units, not
only R^2. For each estimator this script reports, per phase
(pre-impact / impact / relaxation) on test_b:

  C_L:   R^2, RMSE, MAE (C_L units), and in the impact phase the PEAK error
         (predicted-vs-true peak value) and peak TIMING error (t/c);
  field: the analysis/forecast latent decoded through the frozen decode-floor
         decoder (identical-capacity, trained on frozen encoder latents) and
         scored as SSIM over the near-body band, the wake mask and the full
         frame against the DNS field.

Estimators (all pressure-only at run time, deployment-clean, no oracle):
  rex_enkf   REX-EnKF, global Gamma, val-calibrated defaults (the ladder top)
  linear_lae linear-A LAE filter (the robustness arm)
  eobs       static E_obs point estimate (no dynamics)
  openloop   open-loop REX forecast from the pre-impact context (no assimilation)

Representative traces (C_L values + decoded frames) are saved for figures.

Run (RTX 6000): taskset -c 8-15 python -m scripts.session34.da_phase_eval --gpu 1
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

from scripts.session34.lae_enkf_pilot import (  # noqa: E402
    delay_embed,
    encounters,
    fit_linear_A,
    load_aligned,
)
from scripts.session34.latent_rex import LatentRex  # noqa: E402

BAND_TO_SIGMA = 2.5631


def phase_masks(wmask: np.ndarray, t_init: int) -> dict[str, np.ndarray]:
    T = wmask.size
    idx = np.arange(T)
    if wmask.any():
        w0, w1 = np.nonzero(wmask)[0][[0, -1]]
    else:
        w0, w1 = 55, 55
    return {
        "pre": (idx > t_init) & (idx < w0),
        "impact": wmask.copy(),
        "relax": idx > w1,
    }


def cl_metrics(t: np.ndarray, p: np.ndarray) -> dict:
    sse = float(((t - p) ** 2).sum())
    sst = float(((t - t.mean()) ** 2).sum())
    return {
        "r2": 1.0 - sse / max(sst, 1e-12),
        "rmse": float(np.sqrt(np.mean((t - p) ** 2))),
        "mae": float(np.mean(np.abs(t - p))),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="phase-resolved DA evaluation")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--model", default="jepa_pool_vec")
    ap.add_argument("--cache-dir", default="outputs/session33/q1_vec_latents")
    ap.add_argument("--pressure-dir", default="outputs/session31/q1_latents")
    ap.add_argument("--taps", default="outputs/session33/osp_taps_vec.json")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--delay", type=int, default=10)
    ap.add_argument("--members", type=int, default=64)
    ap.add_argument("--band-scale", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rep-case", default="G-0.50_D1.00_Y-0.40")
    ap.add_argument("--out", default="outputs/session34/da_phase_eval.json")
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
    t0 = time.time()

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

    # --- shared fitted pieces (train only) -----------------------------------
    A, Q_lin, _, _ = fit_linear_A(tr, encs_tr, args.alpha)
    X_tr = np.concatenate([delay_embed((tr["p"][e["rows"]][:, taps] - p_mu) / p_sd,
                                       args.delay) for e in encs_tr])
    Zt_tr = np.concatenate([tr["z"][e["rows"]] for e in encs_tr])
    W = np.linalg.solve(X_tr.T @ X_tr + args.alpha * np.eye(X_tr.shape[1]),
                        X_tr.T @ Zt_tr)
    Gamma = np.cov((Zt_tr - X_tr @ W).T) + 1e-8 * np.eye(n)
    chol_G = np.linalg.cholesky(Gamma)
    chol_Ql = np.linalg.cholesky(Q_lin)
    probe = fit_linear_probe(tr["z"], tr["cl"])
    rex = LatentRex(d=n, horizon=40)
    rex.load_state_dict(torch.load(
        REPO_ROOT / "outputs/session34/latent_rex_model.pt", map_location="cpu"))
    rex.to(device).eval()

    # decode-floor decoder saved by the C3 chain (frozen-encoder protocol)
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
    def decode(z_seq: np.ndarray) -> np.ndarray:
        outs = []
        for i in range(0, z_seq.shape[0], 128):
            zb = torch.from_numpy(tile(z_seq[i:i + 128])).to(device)
            outs.append(decoder(zb).float().squeeze(1).cpu().numpy())
        return np.concatenate(outs)

    @torch.no_grad()
    def rex_step(ctx_np: np.ndarray):
        out = rex(torch.from_numpy(ctx_np[:, -30:]).float().to(device))
        s0 = out[:, 0].cpu().numpy()
        med = s0[..., s0.shape[-1] // 2]
        sig = np.clip((s0[..., -1] - s0[..., 0]) / BAND_TO_SIGMA * args.band_scale,
                      1e-4, None)
        return med, sig

    N = args.members
    per_est: dict[str, list] = {k: [] for k in ("rex_enkf", "linear_lae", "eobs", "openloop")}
    rep_traces = {}
    t_init = args.delay - 1

    for e in encs_tb:
        rows = e["rows"]
        z_true, cl_true = tb["z"][rows], tb["cl"][rows]
        wmask = tb["wmask"][rows]
        T = rows.size
        pmask = phase_masks(wmask, t_init)
        pt = (tb["p"][rows][:, taps] - p_mu) / p_sd
        z_obs = delay_embed(pt, args.delay) @ W

        est_z = {"eobs": z_obs.copy()}

        # --- REX-EnKF (deployment-clean: global Gamma) ------------------------
        ctx = np.repeat(z_obs[None, :t_init + 1], N, axis=0)
        ctx[:, -1] += rng.standard_normal((N, n)) @ chol_G.T
        zA = np.empty((T, n)); zA[:t_init + 1] = z_obs[:t_init + 1]
        for t in range(t_init + 1, T):
            med, sig = rex_step(ctx)
            zf = med + rng.standard_normal((N, n)) * sig
            dZ = zf - zf.mean(0, keepdims=True)
            P = dZ.T @ dZ / (N - 1)
            K_g = P @ np.linalg.inv(P + Gamma)
            innov = z_obs[t][None] + rng.standard_normal((N, n)) @ chol_G.T - zf
            za = zf + innov @ K_g.T
            zA[t] = za.mean(0)
            ctx = np.concatenate([ctx, za[:, None]], axis=1)[:, -30:]
        est_z["rex_enkf"] = zA

        # --- linear-A LAE filter ----------------------------------------------
        Zm = z_obs[t_init][None] + rng.standard_normal((N, n)) @ chol_G.T
        zL = np.empty((T, n)); zL[:t_init + 1] = z_obs[:t_init + 1]
        for t in range(t_init + 1, T):
            Zm = Zm @ A.T + rng.standard_normal((N, n)) @ chol_Ql.T
            dZ = Zm - Zm.mean(0, keepdims=True)
            P = dZ.T @ dZ / (N - 1)
            K_g = P @ np.linalg.inv(P + Gamma)
            innov = z_obs[t][None] + rng.standard_normal((N, n)) @ chol_G.T - Zm
            Zm = Zm + innov @ K_g.T
            zL[t] = Zm.mean(0)
        est_z["linear_lae"] = zL

        # --- open-loop REX from pre-impact context ----------------------------
        zO = z_obs.copy()
        with torch.no_grad():
            ctx1 = torch.from_numpy(z_true[None, :25]).float().to(device)
            roll = []
            for _ in range(T - 25):
                out = rex(ctx1[:, -30:])
                z_next = out[:, 0, :, out.shape[-1] // 2]
                roll.append(z_next[0].cpu().numpy())
                ctx1 = torch.cat([ctx1, z_next[None]], dim=1)
        zO[25:] = np.asarray(roll)
        est_z["openloop"] = zO

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
                pf = masked_ssim_per_frame(dec_fields[m], truth_fields[m],
                                           ssim_L, masks)
                rec[ph]["ssim"] = {k: float(v.mean()) for k, v in pf.items()}
            # peak diagnostics (impact phase, physical units)
            ip = np.nonzero(pmask["impact"])[0]
            if ip.size:
                tp, pp = ip[np.argmax(np.abs(cl_true[ip]))], ip[np.argmax(np.abs(cl_hat[ip]))]
                rec["peak"] = {
                    "true_peak_cl": float(cl_true[tp]),
                    "est_peak_cl": float(cl_hat[pp]),
                    "peak_value_error": float(cl_hat[pp] - cl_true[tp]),
                    "peak_rel_error_pct": float(100 * abs(cl_hat[pp] - cl_true[tp])
                                                / max(abs(cl_true[tp]), 1e-6)),
                    "peak_timing_error_tc": float((pp - tp) * 0.05),
                }
            per_est[name].append(rec)
            if e["case_id"] == args.rep_case and e["k"] == 0:
                rep_traces[name] = {"cl_true": cl_true.tolist(),
                                    "cl_est": probe.predict(z_est).tolist()}

    # --- aggregate -------------------------------------------------------------
    def agg(recs: list, ph: str, key: str, sub=None):
        vals = [r[ph][key] if sub is None else r[ph][key][sub]
                for r in recs if ph in r]
        return float(np.median(vals))

    summary = {}
    for name, recs in per_est.items():
        summary[name] = {}
        for ph in ("pre", "impact", "relax"):
            summary[name][ph] = {
                "cl_r2": agg(recs, ph, "r2"),
                "cl_rmse": agg(recs, ph, "rmse"),
                "cl_mae": agg(recs, ph, "mae"),
                "ssim_nearbody": agg(recs, ph, "ssim", "nearbody"),
                "ssim_wake": agg(recs, ph, "ssim", "wake"),
                "ssim_full": agg(recs, ph, "ssim", "full"),
            }
        pv = [abs(r["peak"]["peak_value_error"]) for r in recs if "peak" in r]
        pr = [r["peak"]["peak_rel_error_pct"] for r in recs if "peak" in r]
        summary[name]["peak_rel_error_pct_median"] = float(np.median(pr))
        ptim = [abs(r["peak"]["peak_timing_error_tc"]) for r in recs if "peak" in r]
        summary[name]["peak_abs_value_error_median"] = float(np.median(pv))
        summary[name]["peak_abs_timing_error_tc_median"] = float(np.median(ptim))
        print(f"[da-phase] {name}:")
        for ph in ("pre", "impact", "relax"):
            s = summary[name][ph]
            print(f"    {ph:7s} CL R2={s['cl_r2']:+.3f} RMSE={s['cl_rmse']:.3f} "
                  f"MAE={s['cl_mae']:.3f} | SSIM nb/wake/full "
                  f"{s['ssim_nearbody']:.3f}/{s['ssim_wake']:.3f}/{s['ssim_full']:.3f}",
                  flush=True)
        print(f"    peak |dCL|={summary[name]['peak_abs_value_error_median']:.3f} "
              f"|dt|={summary[name]['peak_abs_timing_error_tc_median']:.3f} t/c",
              flush=True)

    out = REPO_ROOT / args.out
    out.write_text(json.dumps({
        "protocol": vars(args) | {
            "phases": "pre=(t_init, impact_start); impact=window_mask; relax=(impact_end, T)",
            "decoder": "frozen decode-floor (identical-capacity, C3 checkpoint)",
        },
        "summary": summary,
        "records": per_est,
        "representative_traces": rep_traces,
    }, indent=1, default=float))
    print(f"[da-phase] wrote {out} in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
