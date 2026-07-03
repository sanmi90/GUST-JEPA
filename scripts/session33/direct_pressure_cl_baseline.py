"""Direct windowed pressure -> C_L regression: the no-latent baseline.

Session 33 audit. The envelope compares the filter against a STATIC recovery
that goes through the latent (pressure -> state -> probe). The referee question
is simpler: can a plain windowed regression from the taps straight to the lift,
with no latent at all, track the load per encounter? If it tracks at strong
gusts, the filter's load-tracking adds little; if it fails like the via-latent
recovery, the sequential dynamics are doing real work.

Protocol (characterisation, D236 pattern): KRR (Nystroem-RBF + Ridge, the O1
estimator) on causal windows (W = 30) at the qDEIM K = 8 target-blind taps
(matched sensing budget) and, as an upper bound, at all 192 taps. Fit on TRAIN
encounters only; per-encounter R2 over the assimilation window
[t_imp - 24, t_imp + 48] on val / test_b / test_c; the train stratum is scored
out-of-fold (5-fold GroupKFold by encounter). Stratified by |G| next to the
frozen filter and via-latent recovery medians.

Inputs: outputs/session33/t3_latents.npz (p_wall + meta for all 450 encounters);
C_L reloaded from the v2p2 cache. Output:
outputs/session33/direct_pressure_cl_baseline.json.

Run:
    taskset -c 16-23 python -m scripts.session33.direct_pressure_cl_baseline
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

GORDER = ["0.25-0.5", "1", "1.5", "2", "3", "4"]


def bucket_of(g: float) -> str | None:
    edges = {"0.25-0.5": (0.24, 0.5), "1": (0.9, 1.0), "1.5": (1.4, 1.5),
             "2": (1.9, 2.0), "3": (2.9, 3.0), "4": (3.9, 4.0)}
    for name, (lo, hi) in edges.items():
        if lo < g <= hi:
            return name
    return None


def cache_dir(partition: str) -> Path:
    env = os.environ.get("VORTEX_JEPA_CACHE")
    if env:
        return Path(env) / partition
    prevent = os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT"))
    return Path(prevent) / "data" / "processed" / "vortex-jepa" / partition


def windowed(p: np.ndarray, w: int) -> np.ndarray:
    """Causal per-frame window (T, w*K), edge-padded (track_b convention)."""
    t_len, k = p.shape
    out = np.empty((t_len, w * k), dtype=np.float32)
    for t in range(t_len):
        lo = t - w + 1
        if lo < 0:
            win = np.concatenate([np.repeat(p[0:1], -lo, axis=0), p[: t + 1]], axis=0)
        else:
            win = p[lo: t + 1]
        out[t] = win.reshape(-1)
    return out


def per_encounter_r2(yhat, y):
    mu = y.mean()
    den = float(((y - mu) ** 2).sum())
    return float(1.0 - ((yhat - y) ** 2).sum() / den) if den > 0 else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description="direct windowed pressure -> C_L baseline")
    ap.add_argument("--latents-npz", default="outputs/session33/t3_latents.npz")
    ap.add_argument("--qdeim-taps", default="outputs/session32/qdeim_taps_v2p2.json")
    ap.add_argument("--envelope-json", default="outputs/session32/envelope_by_gust.json")
    ap.add_argument("--partition", default="v2p2")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--n-components", type=int, default=300)
    ap.add_argument("--sub", type=int, default=9000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/session33/direct_pressure_cl_baseline.json")
    args = ap.parse_args(argv)

    from sklearn.model_selection import GroupKFold

    from src.evaluation.pressure_infer import fit_pressure_estimator

    d = np.load(REPO_ROOT / args.latents_npz, allow_pickle=True)
    qdeim = json.loads((REPO_ROOT / args.qdeim_taps).read_text())
    taps8 = sorted(int(t) for t in qdeim["K8"])
    n_enc = d["p_wall"].shape[0]
    split = d["split"].astype(str)
    abs_g = np.abs(d["G"].astype(float))
    cdir = cache_dir(args.partition)

    # reload C_L per encounter from the cache (t3 npz stores pressure only)
    n_frames = d["p_wall"].shape[1]
    cl = np.zeros((n_enc, n_frames))
    for e in range(n_enc):
        f = cdir / str(d["case_id"][e]) / f"encounter_{int(d['encounter_index'][e]):02d}.h5"
        with h5py.File(f, "r") as h:
            cl[e] = np.asarray(h["C_L"])[:n_frames]

    rng = np.random.default_rng(args.seed)
    results = {}
    for tap_name, taps in (("K8_qdeim", taps8), ("all192", list(range(192)))):
        # per-encounter windows on the assimilation frames
        feats, targs, encs = [], [], []
        for e in range(n_enc):
            p = d["p_wall"][e][:, taps]
            win = windowed(p, args.window)
            f0, f1 = int(d["f0"][e]), int(d["f1"][e])
            fr = np.arange(f0, f1 + 1)
            feats.append(win[fr])
            targs.append(cl[e][fr])
            encs.append(np.full(len(fr), e))
        X = np.concatenate(feats)
        y = np.concatenate(targs)
        enc_row = np.concatenate(encs)

        yhat = np.full(len(y), np.nan)
        tr_rows = np.isin(enc_row, np.where(split == "train")[0])
        # train stratum: 5-fold OOF within train encounters
        gkf = GroupKFold(n_splits=5)
        Xtr, ytr, gtr = X[tr_rows], y[tr_rows], enc_row[tr_rows]
        oof = np.full(len(ytr), np.nan)
        for tr_i, te_i in gkf.split(Xtr, ytr, gtr):
            sub = rng.choice(len(tr_i), size=min(args.sub, len(tr_i)), replace=False)
            est = fit_pressure_estimator(
                Xtr[tr_i][sub], ytr[tr_i][sub, None],
                n_components=args.n_components, seed=args.seed,
                groups=gtr[tr_i][sub].astype(str),
            )
            oof[te_i] = est.predict(Xtr[te_i]).reshape(-1)
        yhat[tr_rows] = oof
        # held-out splits: fit once on train, predict the rest
        sub = rng.choice(len(Xtr), size=min(args.sub, len(Xtr)), replace=False)
        est = fit_pressure_estimator(
            Xtr[sub], ytr[sub, None], n_components=args.n_components,
            seed=args.seed, groups=gtr[sub].astype(str),
        )
        yhat[~tr_rows] = est.predict(X[~tr_rows]).reshape(-1)

        # per-encounter R2, stratified
        r2_e = np.full(n_enc, np.nan)
        for e in range(n_enc):
            m = enc_row == e
            if m.any():
                r2_e[e] = per_encounter_r2(yhat[m], y[m])
        strata = {}
        for gname in GORDER:
            sel = np.array([bucket_of(g) == gname for g in abs_g]) & np.isfinite(r2_e)
            if sel.any():
                strata[gname] = {
                    "n": int(sel.sum()),
                    "median_r2": float(np.median(r2_e[sel])),
                    "frac_positive": float(np.mean(r2_e[sel] > 0)),
                }
        results[tap_name] = strata
        print(f"[direct-cl] {tap_name}: " + " ".join(
            f"|G|={g}:{strata[g]['median_r2']:+.2f}" for g in GORDER if g in strata),
            flush=True)

    # frozen comparators
    env = json.loads((REPO_ROOT / args.envelope_json).read_text())
    byg = env["models"]["jepa_pool"]["aggregates"]["by_G"]
    comp = {
        g: {
            "filter_CL_median": byg[g]["filter_CL_analysis_r2_impact"]["median"],
            "vialatent_recovery_CL_median": byg[g]["recovery_CL_r2_impact"]["median"],
        }
        for g in GORDER if byg.get(g, {}).get("n", 0)
    }

    payload = {
        "task": "SESSION 33 audit -- direct windowed pressure -> C_L (no latent)",
        "params": {
            "window": args.window,
            "estimator": "KRR Nystroem-RBF (O1/fit_pressure_estimator)",
            "fit": "train only (train stratum scored 5-fold OOF); "
                   "val/test_b/test_c predicted from the train fit (D236 pattern)",
            "scoring": "per-encounter R2 over the assimilation window",
            "note": "R2 basis differs from the envelope lenses (whole assim window "
                    "here vs impact phase there); compare shapes across |G|, "
                    "not decimals",
        },
        "direct": results,
        "frozen_comparators_jepa_pool": comp,
    }
    out = REPO_ROOT / args.out
    out.write_text(json.dumps(payload, indent=2))
    print(f"[direct-cl] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
