"""Real-life deployment test: streaming pressure-only state estimation.

Simulates deployment of the trained ROM stack (Session 34): the filter sees
ONLY the K=8 wall-pressure taps, streamed continuously over each test_b CASE
(all encounters concatenated in raw time order, gust re-releases included),
with a single pressure-only initialization at stream start, NO per-encounter
resets, NO oracle impact windows anywhere in the loop (deployment-clean
config: REX forecast + quantile-band Q + E_obs H=I update + global Gamma),
and optional white sensor noise on the taps (fraction of per-tap train std)
with the induced observation-covariance inflation
Gamma_eff = Gamma + sigma^2 W^T W applied, since noise propagates through the
delay-embedded regression.

Oracle windows and true latents are used for SCORING only.

Run (RTX 6000): taskset -c 8-15 python -m scripts.session34.rex_stream --gpu 1
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

from scripts.session34.lae_enkf_pilot import delay_embed, encounters, load_aligned, r2  # noqa: E402
from scripts.session34.latent_rex import LatentRex  # noqa: E402

BAND_TO_SIGMA = 2.5631


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="streaming deployment test")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--model", default="jepa_pool_vec")
    ap.add_argument("--cache-dir", default="outputs/session33/q1_vec_latents")
    ap.add_argument("--pressure-dir", default="outputs/session31/q1_latents")
    ap.add_argument("--taps", default="outputs/session33/osp_taps_vec.json")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--delay", type=int, default=10)
    ap.add_argument("--members", type=int, default=64)
    ap.add_argument("--band-scale", type=float, default=4.0)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="Sensor noise std as fraction of per-tap train std.")
    ap.add_argument("--alpha-obs", type=float, default=1.0)
    ap.add_argument("--rex-ckpt", default="outputs/session34/latent_rex_model.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from src.utils.device import require_rtx6000

    device = require_rtx6000(gpu_index=args.gpu)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    tr = load_aligned(REPO_ROOT / args.cache_dir, REPO_ROOT / args.pressure_dir,
                      args.model, "train")
    tb = load_aligned(REPO_ROOT / args.cache_dir, REPO_ROOT / args.pressure_dir,
                      args.model, "test_b")
    encs_tr = encounters(tr)
    n = tr["z"].shape[1]
    taps = np.asarray(json.loads((REPO_ROOT / args.taps).read_text())
                      [args.model][f"K{args.k}"], dtype=int)
    p_mu = tr["p"][:, taps].mean(axis=0)
    p_sd = tr["p"][:, taps].std(axis=0) + 1e-9

    rex = LatentRex(d=n, horizon=40)
    rex.load_state_dict(torch.load(REPO_ROOT / args.rex_ckpt, map_location="cpu"))
    rex.to(device).eval()

    X_tr = np.concatenate([delay_embed((tr["p"][e["rows"]][:, taps] - p_mu) / p_sd,
                                       args.delay) for e in encs_tr])
    Zt_tr = np.concatenate([tr["z"][e["rows"]] for e in encs_tr])
    W = np.linalg.solve(X_tr.T @ X_tr + args.alpha_obs * np.eye(X_tr.shape[1]),
                        X_tr.T @ Zt_tr)
    Gamma = np.cov((Zt_tr - X_tr @ W).T) + 1e-8 * np.eye(n)
    if args.noise > 0:
        Gamma = Gamma + (args.noise ** 2) * (W.T @ W)   # induced obs-noise inflation
    chol_G = np.linalg.cholesky(Gamma)

    from src.evaluation.represent import fit_linear_probe

    probe = fit_linear_probe(tr["z"], tr["cl"])
    N, MAX_CTX = args.members, 30

    @torch.no_grad()
    def rex_step(ctx_np):
        out = rex(torch.from_numpy(ctx_np[:, -MAX_CTX:]).float().to(device))
        s0 = out[:, 0].cpu().numpy()
        return s0[..., 1], np.clip((s0[..., 2] - s0[..., 0]) / BAND_TO_SIGMA
                                   * args.band_scale, 1e-4, None)

    # ---- build per-case continuous streams (raw time order) ------------------
    cases = sorted(set(tb["case_id"].tolist()))
    records = []
    for cid in cases:
        rows = np.where(tb["case_id"] == cid)[0]
        order = np.lexsort((tb["frame"][rows], tb["enc"][rows]))
        rows = rows[order]
        T = rows.size
        z_true, cl_true = tb["z"][rows], tb["cl"][rows]
        wmask, enc_idx = tb["wmask"][rows], tb["enc"][rows]
        pt = (tb["p"][rows][:, taps] - p_mu) / p_sd
        if args.noise > 0:
            pt = pt + args.noise * rng.standard_normal(pt.shape)
        z_obs = delay_embed(pt, args.delay) @ W
        t_init = args.delay - 1
        ctx = np.repeat(z_obs[None, : t_init + 1], N, axis=0)
        ctx[:, -1] += rng.standard_normal((N, n)) @ chol_G.T
        zA = np.empty((T, n))
        zA[: t_init + 1] = z_obs[: t_init + 1]
        for t in range(t_init + 1, T):          # ONE continuous pass, no resets
            med, sig = rex_step(ctx)
            zf = med + rng.standard_normal((N, n)) * sig
            dZ = zf - zf.mean(0, keepdims=True)
            P = dZ.T @ dZ / (N - 1)
            K_g = P @ np.linalg.inv(P + Gamma)
            innov = z_obs[t][None] + rng.standard_normal((N, n)) @ chol_G.T - zf
            za = zf + innov @ K_g.T
            zA[t] = za.mean(0)
            ctx = np.concatenate([ctx, za[:, None]], axis=1)[:, -MAX_CTX:]
        cl_hat = probe.predict(zA)
        # score per encounter segment (oracle windows for scoring only)
        for k in sorted(set(enc_idx.tolist())):
            seg = enc_idx == k
            imp = seg & wmask
            if imp.sum() < 5 or np.nonzero(seg)[0][0] < t_init:
                pass
            records.append({
                "case_id": cid, "encounter_index": int(k),
                "CL_analysis_r2_impact": r2(cl_true[imp], cl_hat[imp])
                if imp.any() else None,
                "latent_track_r2": r2(z_true[seg], zA[seg]),
            })
        print(f"[stream] {cid}: {T} frames, {len(set(enc_idx.tolist()))} encounters",
              flush=True)

    imp = np.array([r["CL_analysis_r2_impact"] for r in records
                    if r["CL_analysis_r2_impact"] is not None], float)
    agg = {"median_CL_r2_impact": float(np.nanmedian(imp)),
           "catastrophic_lt_-1": int((imp < -1).sum()),
           "n_encounters": int(imp.size), "noise": args.noise}
    print(f"[stream] noise={args.noise}: impact median={agg['median_CL_r2_impact']:+.3f} "
          f"catastrophic={agg['catastrophic_lt_-1']}/{agg['n_encounters']}", flush=True)
    out = args.out or f"outputs/session34/rex_stream_noise{args.noise}.json"
    Path(REPO_ROOT / out).write_text(json.dumps(
        {"protocol": vars(args), "aggregates": agg, "records": records},
        indent=1, default=float))
    print(f"[stream] wrote {out} in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
