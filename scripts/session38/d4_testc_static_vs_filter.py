"""d = 4 at the |G| = 4 boundary: static inverse vs model + pressure (Session 38).

Encodes test_c through the frozen jepa_pool_vec_d4 encoder (first time at
d = 4), aligns wall pressure, and runs BOTH estimators with train-only fits:
the windowed static delay-embedded inverse (E_obs, W = 30-equivalent delay
10 window as in the phase-eval arms) and the REX-EnKF at the production
band 1.77. Reports median impact R2/RMSE and divergent counts over the 40
boundary encounters. Completes the in-distribution d4 comparison where the
static inverse edges the filter on medians.

Run (RTX 6000): taskset -c 0-15 python -m scripts.session38.d4_testc_static_vs_filter --gpu 0
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

MODEL = "jepa_pool_vec_d4"
CACHE = REPO_ROOT / "outputs/session34/trackc_latents"
TESTC = REPO_ROOT / "outputs/session38/d4_testc"
BAND_TO_SIGMA = 2.5631
DELAY, K, N, ALPHA, BAND = 10, 8, 64, 1.0, 1.77


def ensure_testc(device):
    from src.evaluation import rom_eval as re
    from src.evaluation.pressure_infer import load_pressure_for_alignment
    TESTC.mkdir(parents=True, exist_ok=True)
    lat = TESTC / f"latents_{MODEL}_test_c.npz"
    if not lat.exists():
        windows = re.load_windows(REPO_ROOT / "outputs/session31/windows_v2p2.json")
        frozen = re.load_frozen_model(
            REPO_ROOT / "outputs/runs/session34/jepa_pool_vec_d4",
            "checkpoint_iter010000.pt", device)
        enc = re.encode_split(frozen, "test_c", partition="v2p2",
                              windows=windows, device=device)
        re.save_latents(lat, enc)
        del frozen
        torch.cuda.empty_cache()
        print(f"[d4-testc] wrote {lat}", flush=True)
    pp = TESTC / "pressure_test_c.npz"
    if not pp.exists():
        z = np.load(lat, allow_pickle=True)
        cid = z["case_id"].astype(str)
        p = load_pressure_for_alignment(cid, z["encounter_index"], z["frame"],
                                        partition="v2p2")
        np.savez(pp, p_wall=p.astype(np.float32), case_id=cid,
                 encounter_index=z["encounter_index"], frame=z["frame"])
        print(f"[d4-testc] wrote {pp}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/session38/d4_testc_static_vs_filter.json")
    args = ap.parse_args()
    from src.evaluation.represent import fit_linear_probe
    from src.utils.device import require_rtx6000
    device = require_rtx6000(gpu_index=args.gpu)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    ensure_testc(device)

    tr = load_aligned(CACHE, CACHE, MODEL, "train")
    tc = load_aligned(TESTC, TESTC, MODEL, "test_c")
    encs_tr, encs_tc = encounters(tr), encounters(tc)
    n = tr["z"].shape[1]
    taps = np.asarray(json.loads(
        (REPO_ROOT / "outputs/session34/osp_taps_dims.json").read_text())
        [MODEL][f"K{K}"], dtype=int)
    p_mu = tr["p"][:, taps].mean(axis=0)
    p_sd = tr["p"][:, taps].std(axis=0) + 1e-9
    X_tr = np.concatenate([delay_embed((tr["p"][e["rows"]][:, taps] - p_mu) / p_sd,
                                       DELAY) for e in encs_tr])
    Zt_tr = np.concatenate([tr["z"][e["rows"]] for e in encs_tr])
    W = np.linalg.solve(X_tr.T @ X_tr + ALPHA * np.eye(X_tr.shape[1]), X_tr.T @ Zt_tr)
    Gamma = np.cov((Zt_tr - X_tr @ W).T) + 1e-8 * np.eye(n)
    chol_G = np.linalg.cholesky(Gamma)
    probe = fit_linear_probe(tr["z"], tr["cl"])
    rex = LatentRex(d=n, horizon=40)
    rex.load_state_dict(torch.load(
        REPO_ROOT / f"outputs/session34/latent_rex_model_{MODEL}.pt", map_location="cpu"))
    rex.to(device).eval()

    @torch.no_grad()
    def rex_step(ctx_np):
        out = rex(torch.from_numpy(ctx_np[:, -30:]).float().to(device))
        s0 = out[:, 0].cpu().numpy()
        med = s0[..., s0.shape[-1] // 2]
        sig = np.clip((s0[..., -1] - s0[..., 0]) / BAND_TO_SIGMA * BAND, 1e-4, None)
        return med, sig

    recs = []
    t_init = DELAY - 1
    for e in encs_tc:
        rows = e["rows"]
        cl_true = tc["cl"][rows]
        wmask = tc["wmask"][rows]
        T = rows.size
        pt = (tc["p"][rows][:, taps] - p_mu) / p_sd
        z_obs = delay_embed(pt, DELAY) @ W
        cl_static = probe.predict(z_obs)
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
        cl_filter = probe.predict(zA)
        imp = wmask
        ss_tot = float(((cl_true[imp] - cl_true[imp].mean()) ** 2).sum())
        def m(cl_hat):
            ss_res = float(((cl_true[imp] - cl_hat[imp]) ** 2).sum())
            return (1 - ss_res / ss_tot if ss_tot > 0 else None,
                    float(np.sqrt(np.mean((cl_true[imp] - cl_hat[imp]) ** 2))))
        (r2_s, rmse_s), (r2_f, rmse_f) = m(cl_static), m(cl_filter)
        recs.append({"case_id": e["case_id"], "k": e["k"],
                     "static_r2": r2_s, "static_rmse": rmse_s,
                     "filter_r2": r2_f, "filter_rmse": rmse_f})
    for name, kr, kn in (("static ", "static_r2", "static_rmse"),
                         ("filter ", "filter_r2", "filter_rmse")):
        r2s = np.array([r[kr] for r in recs if r[kr] is not None])
        rm = np.array([r[kn] for r in recs])
        print(f"[d4 test_c] {name}: median R2 {np.median(r2s):+.3f}  "
              f"median RMSE {np.median(rm):.3f}  divergent {(r2s < -1).sum()}/{len(r2s)}",
              flush=True)
    deltas = np.array([r["filter_r2"] - r["static_r2"] for r in recs
                       if r["filter_r2"] is not None and r["static_r2"] is not None])
    print(f"[d4 test_c] per-encounter filter-minus-static R2: median {np.median(deltas):+.3f}, "
          f"filter better on {(deltas > 0).sum()}/{len(deltas)}", flush=True)
    (REPO_ROOT / args.out).write_text(json.dumps({
        "_params": {"model": MODEL, "band": BAND, "K": K, "N": N, "delay": DELAY,
                    "split": "test_c", "seed": args.seed,
                    "note": "first d=4 boundary encode; train-only fits"},
        "records": recs}, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
