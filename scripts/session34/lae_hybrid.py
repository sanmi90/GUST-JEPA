"""LAE-EnKF phase 2: HYBRID filter (transformer forecast + latent-encoded taps).

Combines the two winners of the Session-34 filter comparison:
- FORECAST: the flagship's frozen AutoregressivePredictor (nonlinear, tracks
  the shedding cycle in relaxation, where the linear-A filter loses), rolled
  per member on its own analysis context (max_seq_len window), with latent
  process noise Q_trans fitted from the predictor's teacher-forced one-step
  residuals on train.
- OBSERVATION: the LAE-EnKF obs side (arXiv 2603.06752 Eq. 6-7 retrofit) that
  produced ZERO divergences in the pilot: ridge from delay-embedded K=8 OSP
  taps to the full latent (E_obs, H = I), Gamma_tilde = train residual cov.

Scored with the identical protocol as the pilot and paired against BOTH the
frozen D220 transformer-EnKF envelope (latent->taps observation) and the
pilot's linear-A arm. Optional multiplicative inflation rho.

Run (RTX 6000):
    taskset -c 8-15 python -m scripts.session34.lae_hybrid --gpu 1
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

CKPT = "outputs/runs/session33/jepa_pool_vec/checkpoint_iter010000.pt"


def load_predictor(device):
    from src.config.kit_config import load_model_config
    from src.training.canonical_model import CanonicalModel

    blob = torch.load(REPO_ROOT / CKPT, map_location="cpu", weights_only=False)
    args = blob["args"]
    cfg = load_model_config(REPO_ROOT / args["config"])
    model = CanonicalModel(cfg, latent_dim=int(args.get("d", 32)),
                          predictor_class=args.get("predictor_class", "resunet"))
    model.load_state_dict(blob["model_state_dict"], strict=True)
    pred = model.predictor.to(device).eval()
    for p in pred.parameters():
        p.requires_grad_(False)
    return pred


@torch.no_grad()
def fit_q_transformer(pred, tr: dict, encs: list[dict], device,
                      win: int = 32, hop: int = 16) -> np.ndarray:
    """Teacher-forced one-step residual covariance of the frozen predictor."""
    resid = []
    for e in encs:
        z = torch.from_numpy(tr["z"][e["rows"]]).float()
        T = z.shape[0]
        for s in range(0, max(T - win, 1), hop):
            seg = z[s : s + win][None].to(device)          # (1, w, d)
            out = pred(seg)                                 # pos i predicts i+1
            lo = win // 2
            resid.append((out[0, lo:-1] - seg[0, lo + 1:]).cpu().numpy())
    R = np.concatenate(resid, axis=0).astype(np.float64)
    return np.cov(R.T) + 1e-8 * np.eye(R.shape[1])


@torch.no_grad()
def forecast_members(pred, ctx: torch.Tensor) -> torch.Tensor:
    """ctx (N, T<=32, d) -> next-step latent (N, d)."""
    out = pred(ctx)
    return out[:, -1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LAE hybrid filter")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--model", default="jepa_pool_vec")
    ap.add_argument("--cache-dir", default="outputs/session33/q1_vec_latents")
    ap.add_argument("--pressure-dir", default="outputs/session31/q1_latents")
    ap.add_argument("--taps", default="outputs/session33/osp_taps_vec.json")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--delay", type=int, default=10)
    ap.add_argument("--members", type=int, default=64)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--q-scale", type=float, default=1.0,
                    help="Scale on the fitted Q_trans covariance.")
    ap.add_argument("--alpha-obs", type=float, default=1.0)
    ap.add_argument("--obs-mode", choices=["eobs", "phase_switch", "impact_only"], default="eobs",
                    help="phase_switch: E_obs H=I update at impact, classic "
                         "latent->taps (C z) update in relax.")
    ap.add_argument("--gamma-mode", choices=["global", "phase"], default="global",
                    help="phase: separate Gamma_tilde for impact vs relax rows "
                         "(heteroscedastic obs noise from train residuals).")
    ap.add_argument("--obs-every", type=int, default=1,
                    help="Assimilate only every m-th frame (pressure recorded at full rate; update subsampled).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--envelope", default="outputs/session33/envelope_vec.json")
    ap.add_argument("--pilot", default="outputs/session34/lae_enkf_pilot.json")
    ap.add_argument("--out", default="outputs/session34/lae_hybrid.json")
    args = ap.parse_args(argv)

    from src.utils.device import require_rtx6000

    device = require_rtx6000(gpu_index=args.gpu)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
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

    pred = load_predictor(device)
    max_ctx = int(getattr(pred, "max_seq_len", 32))
    Q = args.q_scale * fit_q_transformer(pred, tr, encs_tr, device)
    print(f"[hybrid] Q_trans fitted: tr(Q)={np.trace(Q):.4f} "
          f"(vs latent var {tr['z'].var(axis=0).sum():.2f})", flush=True)

    # E_obs (identical to pilot)
    X_tr, Zt_tr = [], []
    for e in encs_tr:
        pt = (tr["p"][e["rows"]][:, taps] - p_mu) / p_sd
        X_tr.append(delay_embed(pt, args.delay))
        Zt_tr.append(tr["z"][e["rows"]])
    X_tr = np.concatenate(X_tr)
    Zt_tr = np.concatenate(Zt_tr)
    W = np.linalg.solve(X_tr.T @ X_tr + args.alpha_obs * np.eye(X_tr.shape[1]),
                        X_tr.T @ Zt_tr)
    resid_obs = Zt_tr - X_tr @ W
    Gamma = np.cov(resid_obs.T) + 1e-8 * np.eye(n)
    # Classic obs model for phase_switch: taps = C z + r  (ridge z -> taps)
    Y_tr = np.concatenate([(tr["p"][e["rows"]][:, taps] - p_mu) / p_sd
                           for e in encs_tr])
    C_map = np.linalg.solve(Zt_tr.T @ Zt_tr + args.alpha_obs * np.eye(n),
                            Zt_tr.T @ Y_tr)                      # (n, K)
    R_taps = np.cov((Y_tr - Zt_tr @ C_map).T) + 1e-8 * np.eye(len(taps))
    chol_R = np.linalg.cholesky(R_taps)
    wmask_tr = np.concatenate([tr["wmask"][e["rows"]] for e in encs_tr])
    Gamma_imp = np.cov(resid_obs[wmask_tr].T) + 1e-8 * np.eye(n)
    Gamma_rel = np.cov(resid_obs[~wmask_tr].T) + 1e-8 * np.eye(n)
    print(f"[hybrid] E_obs train R2={r2(Zt_tr, X_tr @ W):.3f}  "
          f"tr(Gamma) imp={np.trace(Gamma_imp):.1f} rel={np.trace(Gamma_rel):.1f}",
          flush=True)

    from src.evaluation.represent import fit_linear_probe

    probe = fit_linear_probe(tr["z"], tr["cl"])
    chol_G = np.linalg.cholesky(Gamma)
    chol_G_imp = np.linalg.cholesky(Gamma_imp)
    chol_G_rel = np.linalg.cholesky(Gamma_rel)
    chol_Q = np.linalg.cholesky(Q)
    N = args.members

    records = []
    for e in encs_tb:
        rows = e["rows"]
        z_true = tb["z"][rows]
        cl_true = tb["cl"][rows]
        wmask = tb["wmask"][rows]
        pt = (tb["p"][rows][:, taps] - p_mu) / p_sd
        z_obs = delay_embed(pt, args.delay) @ W
        T = rows.size
        t_init = args.delay - 1
        # shared context history from E_obs; members differ from t_init on
        ctx = np.repeat(z_obs[None, : t_init + 1], N, axis=0)   # (N, t0+1, d)
        ctx[:, -1] += rng.standard_normal((N, n)) @ chol_G.T
        ctx_t = torch.from_numpy(ctx).float().to(device)
        zA = np.empty((T, n))
        zA[: t_init + 1] = z_obs[: t_init + 1]
        for t in range(t_init + 1, T):
            zf = forecast_members(pred, ctx_t[:, -max_ctx:]).cpu().numpy()
            zf = zf + rng.standard_normal((N, n)) @ chol_Q.T
            if args.rho != 1.0:
                zf = zf.mean(0, keepdims=True) + args.rho * (zf - zf.mean(0, keepdims=True))
            if (t - t_init) % args.obs_every != 0:
                zA[t] = zf.mean(0)
                ctx_t = torch.cat(
                    [ctx_t, torch.from_numpy(zf).float().to(device)[:, None]], dim=1
                )[:, -max_ctx:]
                continue
            if args.gamma_mode == "phase":
                G_t = Gamma_imp if wmask[t] else Gamma_rel
                cG_t = chol_G_imp if wmask[t] else chol_G_rel
            else:
                G_t, cG_t = Gamma, chol_G
            if args.obs_mode == "impact_only" and not wmask[t]:
                za = zf  # predict-only through relaxation
            elif args.obs_mode == "phase_switch" and not wmask[t]:
                # Classic partial observation in relax: y = C z + r.
                yf = zf @ C_map                                  # (N, K)
                dZ = zf - zf.mean(0, keepdims=True)
                dY = yf - yf.mean(0, keepdims=True)
                P_zy = dZ.T @ dY / (N - 1)
                P_yy = dY.T @ dY / (N - 1)
                K_gain = P_zy @ np.linalg.inv(P_yy + R_taps)
                innov = pt[t][None] + rng.standard_normal((N, len(taps))) @ chol_R.T - yf
                za = zf + innov @ K_gain.T
            else:
                dZ = zf - zf.mean(0, keepdims=True)
                P = dZ.T @ dZ / (N - 1)
                K_gain = P @ np.linalg.inv(P + G_t)
                innov = z_obs[t][None] + rng.standard_normal((N, n)) @ cG_t.T - zf
                za = zf + innov @ K_gain.T
            zA[t] = za.mean(0)
            ctx_t = torch.cat(
                [ctx_t, torch.from_numpy(za).float().to(device)[:, None]], dim=1
            )[:, -max_ctx:]
        cl_hat = probe.predict(zA)
        imp = wmask
        rel = (~wmask) & (np.arange(T) > (np.nonzero(wmask)[0].max()
                                          if wmask.any() else 55))
        records.append({
            "case_id": e["case_id"], "encounter_index": e["k"],
            "CL_analysis_r2_impact": r2(cl_true[imp], cl_hat[imp]) if imp.any() else None,
            "CL_analysis_r2_relax": r2(cl_true[rel], cl_hat[rel]) if rel.any() else None,
            "CL_analysis_rmse_impact": float(np.sqrt(np.mean(
                (cl_true[imp] - cl_hat[imp]) ** 2))) if imp.any() else None,
            "latent_track_r2": r2(z_true[t_init + 1:], zA[t_init + 1:]),
        })
        print(f"[hybrid] {e['case_id']} e{e['k']}: imp={records[-1]['CL_analysis_r2_impact']:+.3f} "
              f"rel={records[-1]['CL_analysis_r2_relax']:+.3f}", flush=True)

    imp = np.array([r["CL_analysis_r2_impact"] for r in records], float)
    rel = np.array([r["CL_analysis_r2_relax"] for r in records], float)
    agg = {
        "median_CL_r2_impact": float(np.nanmedian(imp)),
        "median_CL_r2_relax": float(np.nanmedian(rel)),
        "catastrophic_impact_lt_-1": int((imp < -1).sum()),
        "median_latent_track_r2": float(np.nanmedian(np.array(
            [r["latent_track_r2"] for r in records], float))),
    }

    def paired(other_records: dict, key: str) -> dict:
        d_i, d_r = [], []
        for r in records:
            k2 = (r["case_id"], r["encounter_index"])
            if k2 in other_records:
                o = other_records[k2]
                d_i.append(r["CL_analysis_r2_impact"] - o["CL_analysis_r2_impact"])
                d_r.append(r["CL_analysis_r2_relax"] - o["CL_analysis_r2_relax"])
        return {
            "n": len(d_i),
            "impact_mean": float(np.mean(d_i)), "impact_median": float(np.median(d_i)),
            "impact_wins": int((np.array(d_i) > 0).sum()),
            "relax_mean": float(np.mean(d_r)), "relax_median": float(np.median(d_r)),
            "relax_wins": int((np.array(d_r) > 0).sum()),
        }

    env = json.loads((REPO_ROOT / args.envelope).read_text())
    env_recs = {(r["case_id"], r["encounter_index"]): r["filter"]
                for r in env["models"][args.model]["records"] if r["split"] == "test_b"}
    pil = json.loads((REPO_ROOT / args.pilot).read_text())
    pil_recs = {(r["case_id"], r["encounter_index"]): r for r in pil["records"]}
    comparison = {
        "vs_transformer_envelope": paired(env_recs, "env"),
        "vs_linear_pilot": paired(pil_recs, "pilot"),
    }
    print(f"[hybrid] median impact={agg['median_CL_r2_impact']:+.3f} "
          f"relax={agg['median_CL_r2_relax']:+.3f} "
          f"catastrophic={agg['catastrophic_impact_lt_-1']}", flush=True)
    for k2, v in comparison.items():
        print(f"[hybrid] {k2}: impact med {v['impact_median']:+.3f} "
              f"(wins {v['impact_wins']}/{v['n']}), relax med {v['relax_median']:+.3f} "
              f"(wins {v['relax_wins']}/{v['n']})", flush=True)

    out = REPO_ROOT / args.out
    out.write_text(json.dumps({
        "protocol": {
            "forecast": f"frozen {args.model} AutoregressivePredictor, Q from "
                        f"teacher-forced train residuals, ctx<=32",
            "observation": f"E_obs ridge, K={args.k} taps x delay {args.delay}, H=I",
            "members": N, "rho": args.rho, "seed": args.seed,
            "q_scale": args.q_scale, "gamma_mode": args.gamma_mode, "obs_mode": args.obs_mode,
        },
        "aggregates": agg,
        "records": records,
        "comparison": comparison,
    }, indent=1, default=float))
    print(f"[hybrid] wrote {out} in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
