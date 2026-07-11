"""d = 4 family aggregates under the REX-EnKF at the PRODUCTION band 1.77.

The frozen dims-grid arm ran band 4.0 (test-peeked, excluded from headlines
by the T5 freeze rule); this recomputes the same per-family end-to-end
rex_enkf arm at the validation-calibrated 1.77 over all test_b encounters,
for the d = 4 comparison set (POD, wake-headed AE, JEPA wake, JEPA
lift-focused). Median impact C_L R2/RMSE per family + per-encounter records.

Run (RTX 6000): taskset -c 0-15 python -m scripts.session38.d4_band177_aggregates --gpu 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from scripts.session34.lae_enkf_pilot import delay_embed, encounters, load_aligned  # noqa: E402
from scripts.session34.latent_rex import LatentRex  # noqa: E402

CACHE = REPO_ROOT / "outputs/session34/trackc_latents"
BAND_TO_SIGMA = 2.5631
DELAY, K, N, ALPHA, BAND = 10, 8, 64, 1.0, 1.77
MODELS = ["pod_d4", "fukami_wake_d4", "jepa_pool_vec_d4", "cln_rexpred_d4_s0"]


def taps_for(model):
    for f in ("outputs/session34/osp_taps_dims.json",
              "outputs/session34/osp_taps_dims2.json"):
        d = json.loads((REPO_ROOT / f).read_text())
        if model in d:
            return np.asarray(d[model][f"K{K}"], dtype=int)
    raise KeyError(model)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/session38/d4_band177_aggregates.json")
    args = ap.parse_args()
    from src.evaluation.represent import fit_linear_probe
    from src.utils.device import require_rtx6000
    device = require_rtx6000(gpu_index=args.gpu)
    torch.manual_seed(args.seed)

    results = {"_params": {"band": BAND, "K": K, "N": N, "delay": DELAY,
                           "recipe": "rex_enkf, global Gamma, own stacks",
                           "split": "test_b", "seed": args.seed}}
    for model in MODELS:
        rng = np.random.default_rng(args.seed)
        tr = load_aligned(CACHE, CACHE, model, "train")
        tb = load_aligned(CACHE, CACHE, model, "test_b")
        encs_tr, encs_tb = encounters(tr), encounters(tb)
        n = tr["z"].shape[1]
        taps = taps_for(model)
        p_mu = tr["p"][:, taps].mean(axis=0)
        p_sd = tr["p"][:, taps].std(axis=0) + 1e-9
        X_tr = np.concatenate([delay_embed((tr["p"][e["rows"]][:, taps] - p_mu) / p_sd,
                                           DELAY) for e in encs_tr])
        Zt_tr = np.concatenate([tr["z"][e["rows"]] for e in encs_tr])
        W = np.linalg.solve(X_tr.T @ X_tr + ALPHA * np.eye(X_tr.shape[1]),
                            X_tr.T @ Zt_tr)
        Gamma = np.cov((Zt_tr - X_tr @ W).T) + 1e-8 * np.eye(n)
        chol_G = np.linalg.cholesky(Gamma)
        probe = fit_linear_probe(tr["z"], tr["cl"])
        rex = LatentRex(d=n, horizon=40)
        rex.load_state_dict(torch.load(
            REPO_ROOT / f"outputs/session34/latent_rex_model_{model}.pt",
            map_location="cpu"))
        rex.to(device).eval()

        @torch.no_grad()
        def rex_step(ctx_np):
            out = rex(torch.from_numpy(ctx_np[:, -30:]).float().to(device))
            s0 = out[:, 0].cpu().numpy()
            med = s0[..., s0.shape[-1] // 2]
            sig = np.clip((s0[..., -1] - s0[..., 0]) / BAND_TO_SIGMA * BAND,
                          1e-4, None)
            return med, sig

        recs = []
        t_init = DELAY - 1
        for e in encs_tb:
            rows = e["rows"]
            cl_true = tb["cl"][rows]
            wmask = tb["wmask"][rows]
            T = rows.size
            pt = (tb["p"][rows][:, taps] - p_mu) / p_sd
            z_obs = delay_embed(pt, DELAY) @ W
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
            cl_hat = probe.predict(zA)
            imp = wmask
            ss_res = float(((cl_true[imp] - cl_hat[imp]) ** 2).sum())
            ss_tot = float(((cl_true[imp] - cl_true[imp].mean()) ** 2).sum())
            recs.append({
                "case_id": e["case_id"], "k": e["k"],
                "impact_r2": 1 - ss_res / ss_tot if ss_tot > 0 else None,
                "impact_rmse": float(np.sqrt(np.mean(
                    (cl_true[imp] - cl_hat[imp]) ** 2))),
            })
        r2s = np.array([r["impact_r2"] for r in recs if r["impact_r2"] is not None])
        rmses = np.array([r["impact_rmse"] for r in recs])
        results[model] = {
            "median_impact_r2": float(np.median(r2s)),
            "median_impact_rmse": float(np.median(rmses)),
            "divergent_lt_-1": int((r2s < -1).sum()),
            "n": len(recs),
            "records": recs,
        }
        print(f"[d4@1.77] {model}: median R2 {np.median(r2s):+.3f}, "
              f"median RMSE {np.median(rmses):.3f}, divergent {(r2s < -1).sum()}",
              flush=True)
    (REPO_ROOT / args.out).write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
