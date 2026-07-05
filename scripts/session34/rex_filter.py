"""REX-EnKF: latent-REX forecast operator inside the latent filter (Session 34).

The coherent assembly of the night's three filter findings:
- FORECAST: latent-REX (TiRex-lessons direct quantile forecaster) propagates
  each member from its OWN analysis context; its median is the conditional
  mean an EnKF forecast wants, and its per-step, per-dim quantile band
  (q90 - q10) supplies a STATE-DEPENDENT diagonal process noise
  sigma_t = band / 2.563 * band_scale -- replacing the constant fitted Q that
  destroyed the relax phase of the transformer hybrid.
- OBSERVATION: the LAE-EnKF retrofit obs side (delay-embedded K=8 taps ->
  latent, H = I) that won the impact phase, with optional phase-dependent
  Gamma and the phase-switch to the classic C z tap observation in relax.

Modes tested (--forecast rex): obs {eobs, phase_switch} x band_scale
{1.0, 1.6, 2.5} (REX 10-90 coverage was 0.62, undercalibrated ~1.6x).
Scored identically to lae_hybrid; paired vs the frozen D220 envelope.

Run (RTX 6000): taskset -c 8-15 python -m scripts.session34.rex_filter --gpu 1
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
    load_aligned,
    r2,
)
from scripts.session34.latent_rex import LatentRex  # noqa: E402

BAND_TO_SIGMA = 2.5631  # q90 - q10 of a Gaussian = 2 * 1.28155 sigma


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="REX-EnKF")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--model", default="jepa_pool_vec")
    ap.add_argument("--cache-dir", default="outputs/session33/q1_vec_latents")
    ap.add_argument("--pressure-dir", default="outputs/session31/q1_latents")
    ap.add_argument("--taps", default="outputs/session33/osp_taps_vec.json")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--delay", type=int, default=10)
    ap.add_argument("--members", type=int, default=64)
    ap.add_argument("--band-scale", type=float, default=1.6)
    ap.add_argument("--obs-mode", choices=["eobs", "phase_switch"], default="eobs")
    ap.add_argument("--gamma-mode", choices=["global", "phase"], default="phase")
    ap.add_argument("--alpha-obs", type=float, default=1.0)
    ap.add_argument("--gamma-rel-scale", type=float, default=1.0,
                    help="Inflate relax-phase obs covariance (trust the "
                         "forecast more where the taps see least).")
    ap.add_argument("--rex-ckpt", default="outputs/session34/latent_rex_model.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--envelope", default="outputs/session33/envelope_vec.json")
    ap.add_argument("--out", default="outputs/session34/rex_filter.json")
    args = ap.parse_args(argv)

    from src.utils.device import require_rtx6000

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

    rex = LatentRex(d=n, horizon=40)
    rex.load_state_dict(torch.load(REPO_ROOT / args.rex_ckpt, map_location="cpu"))
    rex.to(device).eval()

    # ---- obs side (identical to lae_hybrid) ---------------------------------
    X_tr, Zt_tr = [], []
    for e in encs_tr:
        pt = (tr["p"][e["rows"]][:, taps] - p_mu) / p_sd
        X_tr.append(delay_embed(pt, args.delay))
        Zt_tr.append(tr["z"][e["rows"]])
    X_tr = np.concatenate(X_tr)
    Zt_tr = np.concatenate(Zt_tr)
    W = np.linalg.solve(X_tr.T @ X_tr + args.alpha_obs * np.eye(X_tr.shape[1]),
                        X_tr.T @ Zt_tr)
    resid = Zt_tr - X_tr @ W
    wmask_tr = np.concatenate([tr["wmask"][e["rows"]] for e in encs_tr])
    Gam = {"global": np.cov(resid.T) + 1e-8 * np.eye(n),
           "imp": np.cov(resid[wmask_tr].T) + 1e-8 * np.eye(n),
           "rel": args.gamma_rel_scale * (np.cov(resid[~wmask_tr].T)
                                          + 1e-8 * np.eye(n))}
    chol = {k: np.linalg.cholesky(v) for k, v in Gam.items()}
    Y_tr = np.concatenate([(tr["p"][e["rows"]][:, taps] - p_mu) / p_sd
                           for e in encs_tr])
    C_map = np.linalg.solve(Zt_tr.T @ Zt_tr + args.alpha_obs * np.eye(n),
                            Zt_tr.T @ Y_tr)
    R_taps = np.cov((Y_tr - Zt_tr @ C_map).T) + 1e-8 * np.eye(len(taps))
    chol_R = np.linalg.cholesky(R_taps)

    from src.evaluation.represent import fit_linear_probe

    probe = fit_linear_probe(tr["z"], tr["cl"])
    N = args.members
    MAX_CTX = 30

    @torch.no_grad()
    def rex_step(ctx_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N, L, d) member contexts -> next-step (median (N, d), sigma (N, d))."""
        out = rex(torch.from_numpy(ctx_np[:, -MAX_CTX:]).float().to(device))
        step0 = out[:, 0].cpu().numpy()                      # (N, d, 3)
        med = step0[..., 1]
        sig = np.clip((step0[..., 2] - step0[..., 0]) / BAND_TO_SIGMA
                      * args.band_scale, 1e-4, None)
        return med, sig

    records = []
    for e in encs_tb:
        rows = e["rows"]
        z_true, cl_true = tb["z"][rows], tb["cl"][rows]
        wmask = tb["wmask"][rows]
        pt = (tb["p"][rows][:, taps] - p_mu) / p_sd
        z_obs = delay_embed(pt, args.delay) @ W
        T = rows.size
        t_init = args.delay - 1
        ctx = np.repeat(z_obs[None, : t_init + 1], N, axis=0)
        ctx[:, -1] += rng.standard_normal((N, n)) @ chol["global"].T
        zA = np.empty((T, n))
        zA[: t_init + 1] = z_obs[: t_init + 1]
        for t in range(t_init + 1, T):
            med, sig = rex_step(ctx)
            zf = med + rng.standard_normal((N, n)) * sig
            if args.obs_mode == "phase_switch" and not wmask[t]:
                yf = zf @ C_map
                dZ = zf - zf.mean(0, keepdims=True)
                dY = yf - yf.mean(0, keepdims=True)
                K_g = (dZ.T @ dY / (N - 1)) @ np.linalg.inv(dY.T @ dY / (N - 1) + R_taps)
                innov = pt[t][None] + rng.standard_normal((N, len(taps))) @ chol_R.T - yf
                za = zf + innov @ K_g.T
            else:
                G_t = (Gam["imp"] if wmask[t] else Gam["rel"]) \
                    if args.gamma_mode == "phase" else Gam["global"]
                cG = (chol["imp"] if wmask[t] else chol["rel"]) \
                    if args.gamma_mode == "phase" else chol["global"]
                dZ = zf - zf.mean(0, keepdims=True)
                P = dZ.T @ dZ / (N - 1)
                K_g = P @ np.linalg.inv(P + G_t)
                innov = z_obs[t][None] + rng.standard_normal((N, n)) @ cG.T - zf
                za = zf + innov @ K_g.T
            zA[t] = za.mean(0)
            ctx = np.concatenate([ctx, za[:, None]], axis=1)[:, -MAX_CTX:]
        cl_hat = probe.predict(zA)
        imp = wmask
        rel = (~wmask) & (np.arange(T) > (np.nonzero(wmask)[0].max()
                                          if wmask.any() else 55))
        records.append({
            "case_id": e["case_id"], "encounter_index": e["k"],
            "CL_analysis_r2_impact": r2(cl_true[imp], cl_hat[imp]) if imp.any() else None,
            "CL_analysis_r2_relax": r2(cl_true[rel], cl_hat[rel]) if rel.any() else None,
            "latent_track_r2": r2(z_true[t_init + 1:], zA[t_init + 1:]),
        })

    imp = np.array([r["CL_analysis_r2_impact"] for r in records], float)
    rel = np.array([r["CL_analysis_r2_relax"] for r in records], float)
    agg = {"median_CL_r2_impact": float(np.nanmedian(imp)),
           "median_CL_r2_relax": float(np.nanmedian(rel)),
           "catastrophic_impact_lt_-1": int((imp < -1).sum()),
           "catastrophic_relax_lt_-1": int((rel < -1).sum()),
           "median_latent_track_r2": float(np.nanmedian(np.array(
               [r["latent_track_r2"] for r in records], float)))}

    env = json.loads((REPO_ROOT / args.envelope).read_text())
    env_recs = {(r["case_id"], r["encounter_index"]): r["filter"]
                for r in env["models"][args.model]["records"] if r["split"] == "test_b"}
    d_i = [r["CL_analysis_r2_impact"] - env_recs[(r["case_id"], r["encounter_index"])]
           ["CL_analysis_r2_impact"] for r in records]
    d_r = [r["CL_analysis_r2_relax"] - env_recs[(r["case_id"], r["encounter_index"])]
           ["CL_analysis_r2_relax"] for r in records]
    comparison = {"n": len(d_i),
                  "impact_median": float(np.median(d_i)),
                  "impact_wins": int((np.array(d_i) > 0).sum()),
                  "relax_median": float(np.median(d_r)),
                  "relax_wins": int((np.array(d_r) > 0).sum())}
    print(f"[rex-filter] obs={args.obs_mode} gamma={args.gamma_mode} "
          f"band={args.band_scale}: impact={agg['median_CL_r2_impact']:+.3f} "
          f"relax={agg['median_CL_r2_relax']:+.3f} "
          f"cat={agg['catastrophic_impact_lt_-1']}/{agg['catastrophic_relax_lt_-1']} "
          f"| vs envelope: imp med {comparison['impact_median']:+.3f} "
          f"({comparison['impact_wins']}/{comparison['n']}), "
          f"rel med {comparison['relax_median']:+.3f} "
          f"({comparison['relax_wins']}/{comparison['n']})", flush=True)

    Path(REPO_ROOT / args.out).write_text(json.dumps({
        "protocol": vars(args) | {"forecast": "latent-REX median + quantile-band Q"},
        "aggregates": agg, "records": records, "comparison_vs_envelope": comparison,
    }, indent=1, default=float))
    print(f"[rex-filter] wrote {args.out} in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
