"""LAE-EnKF retrofit pilot on frozen jepa_pool_vec latents (Session 34).

Retrofit of Tong, Wang & Yan (arXiv:2603.06752v2) onto the flagship's frozen
d=32 pooled latents, per their own two-stage logic (Stage II fits the obs
encoder on a frozen backbone; we extend the same retrofit to A):

1. LINEAR DYNAMICS GATE: fit A by ridge over train latent transitions
   (z_{t+1} ~ A z_t), then project singular values to <= 1 (the hard analogue
   of their hinge penalty R(A) = max(0, ||A||_2 - 1)^2). Report held-out
   one-step R^2 and a free-rollout R^2 through the impact window -- the paper
   never validates post-hoc linear A on latents shaped by a nonlinear
   predictor (their Fig. 2 warns they may not evolve linearly), so this gate
   is the go/no-go.
2. OBS ENCODER (their Eq. 6-7, H = I): ridge from a delay-embedded window of
   the K=8 OSP wall-pressure taps (Takens lifting, their sparse-sensor
   treatment) to the full latent; Gamma_tilde = train residual covariance.
   This converts 8 sparse taps into a full-rank latent observation.
3. LATENT EnKF (their Algorithm 1, stochastic perturbed-obs) with TWO
   additions they omit but transients need: latent process noise Q from the
   A-fit residuals (their forecast is noise-free), and optional multiplicative
   inflation rho (their no-inflation result rides on contractive dynamics; a
   gust impact is not contractive).
4. Scoring: analysis-mean C_L via the frozen RidgeCV probe (Track C protocol),
   R^2/RMSE over the impact window (the cache's window_mask, same
   windows_v2p2 provenance as the envelope) and the relaxation tail; paired
   per-encounter comparison against the session33 transformer-rollout EnKF
   envelope records (rho=1.0, same taps, same test_b encounters).

CPU-only. Run:
    taskset -c 0-15 python -m scripts.session34.lae_enkf_pilot
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def load_aligned(cache_dir: Path, pressure_dir: Path, model: str, split: str) -> dict:
    z = np.load(cache_dir / f"latents_{model}_{split}.npz", allow_pickle=True)
    p = np.load(pressure_dir / f"pressure_{split}.npz", allow_pickle=True)
    assert (p["case_id"] == z["case_id"]).all() and (p["frame"] == z["frame"]).all(), \
        "pressure cache not row-aligned with latent cache"
    return {
        "z": z["z_gap"].astype(np.float64),
        "p": p["p_wall"].astype(np.float64),
        "cl": z["target_C_L"].astype(np.float64),
        "case_id": z["case_id"],
        "enc": z["encounter_index"],
        "frame": z["frame"],
        "wmask": z["window_mask"].astype(bool),
    }


def encounters(d: dict) -> list[dict]:
    keys = sorted(set(zip(d["case_id"].tolist(), d["enc"].tolist())))
    out = []
    for cid, k in keys:
        rows = np.where((d["case_id"] == cid) & (d["enc"] == k))[0]
        rows = rows[np.argsort(d["frame"][rows])]
        out.append({"case_id": cid, "k": int(k), "rows": rows})
    return out


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sse = float(((y_true - y_pred) ** 2).sum())
    sst = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - sse / max(sst, 1e-12)


def fit_linear_A(tr: dict, encs: list[dict], alpha: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    Z0, Z1 = [], []
    for e in encs:
        z = tr["z"][e["rows"]]
        Z0.append(z[:-1])
        Z1.append(z[1:])
    Z0 = np.concatenate(Z0)
    Z1 = np.concatenate(Z1)
    n = Z0.shape[1]
    A = Z1.T @ Z0 @ np.linalg.inv(Z0.T @ Z0 + alpha * np.eye(n))
    smax_raw = float(np.linalg.norm(A, 2))
    U, S, Vt = np.linalg.svd(A)
    A_stab = U @ np.diag(np.minimum(S, 1.0)) @ Vt
    resid = Z1 - Z0 @ A_stab.T
    Q = np.cov(resid.T) + 1e-8 * np.eye(n)
    return A_stab, Q, smax_raw, float(np.minimum(S, 1.0).max())


def delay_embed(p_taps: np.ndarray, L: int) -> np.ndarray:
    """(T, K) -> (T, K*L) with rows t < L-1 padded by edge replication."""
    T, K = p_taps.shape
    out = np.empty((T, K * L))
    for t in range(T):
        idx = np.clip(np.arange(t - L + 1, t + 1), 0, T - 1)
        out[t] = p_taps[idx].reshape(-1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LAE-EnKF retrofit pilot")
    ap.add_argument("--model", default="jepa_pool_vec")
    ap.add_argument("--cache-dir", default="outputs/session33/q1_vec_latents")
    ap.add_argument("--pressure-dir", default="outputs/session31/q1_latents")
    ap.add_argument("--taps", default="outputs/session33/osp_taps_vec.json")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--delay", type=int, default=10)
    ap.add_argument("--members", type=int, default=64)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--alpha-a", type=float, default=1.0)
    ap.add_argument("--alpha-obs", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--envelope", default="outputs/session33/envelope_vec.json")
    ap.add_argument("--out", default="outputs/session34/lae_enkf_pilot.json")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    cache_dir = REPO_ROOT / args.cache_dir
    pressure_dir = REPO_ROOT / args.pressure_dir

    tr = load_aligned(cache_dir, pressure_dir, args.model, "train")
    tb = load_aligned(cache_dir, pressure_dir, args.model, "test_b")
    encs_tr = encounters(tr)
    encs_tb = encounters(tb)
    n = tr["z"].shape[1]

    taps_blob = json.loads((REPO_ROOT / args.taps).read_text())
    taps = np.asarray(taps_blob[args.model][f"K{args.k}"], dtype=int)
    print(f"[lae] model={args.model} K={args.k} taps={taps.tolist()} "
          f"delay L={args.delay}", flush=True)

    # Tap standardization from train.
    p_mu = tr["p"][:, taps].mean(axis=0)
    p_sd = tr["p"][:, taps].std(axis=0) + 1e-9

    # ---- 1. Linear dynamics gate --------------------------------------------
    A, Q, smax_raw, smax = fit_linear_A(tr, encs_tr, args.alpha_a)
    z0_tb, z1_tb = [], []
    for e in encs_tb:
        z = tb["z"][e["rows"]]
        z0_tb.append(z[:-1])
        z1_tb.append(z[1:])
    z0_tb = np.concatenate(z0_tb)
    z1_tb = np.concatenate(z1_tb)
    one_step_r2 = r2(z1_tb, z0_tb @ A.T)
    persist_r2 = r2(z1_tb, z0_tb)
    # Free rollout through the impact window: start at frame 25, 30 steps.
    roll_true, roll_pred = [], []
    for e in encs_tb:
        z = tb["z"][e["rows"]]
        if z.shape[0] < 56:
            continue
        zc = z[25]
        for s in range(30):
            zc = A @ zc
            roll_pred.append(zc)
            roll_true.append(z[26 + s])
    roll_r2 = r2(np.asarray(roll_true), np.asarray(roll_pred))
    gate = {
        "smax_raw": smax_raw, "smax_projected": smax,
        "one_step_r2_test_b": one_step_r2,
        "one_step_persistence_r2": persist_r2,
        "rollout30_from_frame25_r2": roll_r2,
    }
    print(f"[lae] GATE: smax(A)={smax_raw:.3f}->{smax:.3f}  "
          f"one-step R2={one_step_r2:.3f} (persistence {persist_r2:.3f})  "
          f"rollout30 R2={roll_r2:.3f}", flush=True)

    # ---- 2. Obs encoder: delay-embedded taps -> z, H = I ---------------------
    X_tr, Zt_tr = [], []
    for e in encs_tr:
        pt = (tr["p"][e["rows"]][:, taps] - p_mu) / p_sd
        X_tr.append(delay_embed(pt, args.delay))
        Zt_tr.append(tr["z"][e["rows"]])
    X_tr = np.concatenate(X_tr)
    Zt_tr = np.concatenate(Zt_tr)
    W = np.linalg.solve(X_tr.T @ X_tr + args.alpha_obs * np.eye(X_tr.shape[1]),
                        X_tr.T @ Zt_tr)                       # (K*L, n)
    resid_obs = Zt_tr - X_tr @ W
    Gamma = np.cov(resid_obs.T) + 1e-8 * np.eye(n)
    eobs_r2 = r2(Zt_tr, X_tr @ W)
    print(f"[lae] E_obs: train R2={eobs_r2:.3f} "
          f"(latent observable from {args.k} taps x {args.delay} frames)", flush=True)

    # ---- 3. Frozen C_L probe (Track C protocol) ------------------------------
    from src.evaluation.represent import fit_linear_probe

    probe = fit_linear_probe(tr["z"], tr["cl"])

    # ---- 4. Latent EnKF on test_b --------------------------------------------
    N = args.members
    records = []
    chol_G = np.linalg.cholesky(Gamma)
    chol_Q = np.linalg.cholesky(Q)
    for e in encs_tb:
        rows = e["rows"]
        z_true = tb["z"][rows]
        cl_true = tb["cl"][rows]
        wmask = tb["wmask"][rows]
        pt = (tb["p"][rows][:, taps] - p_mu) / p_sd
        X = delay_embed(pt, args.delay)
        z_obs = X @ W                                       # (T, n)
        T = rows.size
        t_init = args.delay - 1
        Z = z_obs[t_init][None] + rng.standard_normal((N, n)) @ chol_G.T
        zA = np.empty((T, n))
        zA[: t_init + 1] = z_obs[: t_init + 1]
        for t in range(t_init + 1, T):
            Z = Z @ A.T + rng.standard_normal((N, n)) @ chol_Q.T
            if args.rho != 1.0:
                Z = Z.mean(0, keepdims=True) + args.rho * (Z - Z.mean(0, keepdims=True))
            dZ = Z - Z.mean(0, keepdims=True)
            P = dZ.T @ dZ / (N - 1)
            K_gain = P @ np.linalg.inv(P + Gamma)
            innov = z_obs[t][None] + rng.standard_normal((N, n)) @ chol_G.T - Z
            Z = Z + innov @ K_gain.T
            zA[t] = Z.mean(0)
        cl_hat = probe.predict(zA)
        imp = wmask
        rel = (~wmask) & (np.arange(T) > (np.nonzero(wmask)[0].max()
                                          if wmask.any() else 55))
        rec = {
            "case_id": e["case_id"], "encounter_index": e["k"],
            "CL_analysis_r2_impact": r2(cl_true[imp], cl_hat[imp]) if imp.any() else None,
            "CL_analysis_r2_relax": r2(cl_true[rel], cl_hat[rel]) if rel.any() else None,
            "CL_analysis_rmse_impact": float(np.sqrt(np.mean(
                (cl_true[imp] - cl_hat[imp]) ** 2))) if imp.any() else None,
            "latent_track_r2": r2(z_true[t_init + 1:], zA[t_init + 1:]),
        }
        records.append(rec)

    imp_r2 = np.array([r["CL_analysis_r2_impact"] for r in records], dtype=float)
    agg = {
        "median_CL_r2_impact": float(np.nanmedian(imp_r2)),
        "median_CL_r2_relax": float(np.nanmedian(np.array(
            [r["CL_analysis_r2_relax"] for r in records], dtype=float))),
        "median_latent_track_r2": float(np.nanmedian(np.array(
            [r["latent_track_r2"] for r in records], dtype=float))),
        "n_encounters": len(records),
    }
    print(f"[lae] FILTER: median CL R2 impact={agg['median_CL_r2_impact']:.3f} "
          f"relax={agg['median_CL_r2_relax']:.3f} "
          f"latent track={agg['median_latent_track_r2']:.3f}", flush=True)

    # ---- 5. Paired comparison vs the transformer-EnKF envelope ---------------
    comparison = None
    env_path = REPO_ROOT / args.envelope
    if env_path.exists():
        env = json.loads(env_path.read_text())
        env_recs = {
            (r["case_id"], r["encounter_index"]): r["filter"]
            for r in env["models"][args.model]["records"]
            if r.get("split") == "test_b"
        }
        deltas_imp, deltas_rel, pairs = [], [], 0
        for r in records:
            key = (r["case_id"], r["encounter_index"])
            if key in env_recs and r["CL_analysis_r2_impact"] is not None:
                deltas_imp.append(r["CL_analysis_r2_impact"]
                                  - env_recs[key]["CL_analysis_r2_impact"])
                deltas_rel.append((r["CL_analysis_r2_relax"] or 0)
                                  - env_recs[key]["CL_analysis_r2_relax"])
                pairs += 1
        if pairs:
            comparison = {
                "n_paired": pairs,
                "mean_delta_CL_r2_impact_LAE_minus_transformer":
                    float(np.mean(deltas_imp)),
                "median_delta_impact": float(np.median(deltas_imp)),
                "wins_impact": int(np.sum(np.array(deltas_imp) > 0)),
                "mean_delta_CL_r2_relax": float(np.mean(deltas_rel)),
                "median_delta_relax": float(np.median(deltas_rel)),
                "wins_relax": int(np.sum(np.array(deltas_rel) > 0)),
                "caveat": ("init/assimilation-start protocols differ slightly "
                           "(LAE starts at frame L-1 from E_obs; envelope uses "
                           "field-free O1 init); windows share windows_v2p2 "
                           "provenance via the cache window_mask."),
            }
            print(f"[lae] vs transformer-EnKF (paired {pairs}): "
                  f"impact delta mean={comparison['mean_delta_CL_r2_impact_LAE_minus_transformer']:+.3f} "
                  f"(wins {comparison['wins_impact']}/{pairs}); "
                  f"relax delta mean={comparison['mean_delta_CL_r2_relax']:+.3f} "
                  f"(wins {comparison['wins_relax']}/{pairs})", flush=True)

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "protocol": {
            "model": args.model, "K": args.k, "delay": args.delay,
            "members": N, "rho": args.rho, "alpha_A": args.alpha_a,
            "alpha_obs": args.alpha_obs, "seed": args.seed,
            "reference": "arXiv:2603.06752v2 Algorithm 1, retrofit on frozen latents",
            "additions_vs_paper": "latent Q from A-residuals; optional inflation",
        },
        "gate_linear_dynamics": gate,
        "obs_encoder": {"train_r2": eobs_r2},
        "aggregates": agg,
        "records": records,
        "comparison_vs_envelope": comparison,
    }, indent=1, default=float))
    print(f"[lae] wrote {out} in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
