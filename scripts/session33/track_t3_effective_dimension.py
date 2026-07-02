"""Track T3 -- effective dimension per gust stratum, and the embedding bound.

SESSION_33_MANUSCRIPT_V3.md Section 11 item 12 (Table T3); HANDOFF D238/D239.

Estimates the encounter attractor's effective dimension d_eff per |G| stratum
from the frozen jepa_pool encoded trajectories, three ways:

  1. Grassberger-Procaccia correlation dimension (primary): pooled stratum
     points, Theiler exclusion (same-encounter pairs with |dt| < 30 frames, one
     shedding period), log-spaced C(r), slope over the flattest local-slope
     window, encounter-resampling bootstrap CI. Reported as a BOUNDED ESTIMATE
     per the addendum caveat, never a fitted constant.
  2. Participation ratio (src.training.diagnostics.participation_ratio).
  3. n_PC(90%) -- smallest number of principal components spanning 90% variance.

The out-of-plane enstrophy fraction column (three-dimensional-onset proxy) is
HARVESTED from outputs/session23/chi3d/chi3d.json (per-case max_chi3d_wz_post,
the paper's Section 2.1 convention), NOT recomputed from mid-plane data. Caveat
recorded: that set is the 84 v2-era cases (the 17 run4 cases, i.e. the G_inv=+4
side of Test C, are absent).

Bound overlay: per stratum, m_needed(K=8) = smallest window W whose stratum
state-recovery R2 (from track_t_recovery_grid.json per_G_window) meets the
target, against the delay-embedding bound 2*d_eff/K; plus the envelope
divergence boundary (envelope_by_gust.json thresholds). Two targets are
reported: relative (90% of the stratum's W=30 recovery) and absolute (0.5).
The |G|=4 stratum is absent from the grid (train+test_b); it is filled by a
CHARACTERISATION-ONLY pass: KRR recovery fit on train windows at (K=8, W),
applied to the test_c encoded latents (D236 precedent; no selection).

Gate T3 (descriptive): consistency of d_eff growth with |G|, m_needed growth,
and the envelope divergence boundary. Consistency, not a fit.

Run (RTX 6000 for encoding; estimator is CPU):
    taskset -c 0-15 python -m scripts.session33.track_t3_effective_dimension --gpu 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

THEILER_DT = 30  # frames; ~one shedding period at St=0.675, dt_tc=0.05
G_ORDER = ("0", "0.25-0.5", "1", "1.5", "2", "3", "4")


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def bucket_of(abs_g: float) -> str | None:
    if abs_g == 0.0:
        return "0"
    if 0.24 < abs_g <= 0.5:
        return "0.25-0.5"
    if 0.9 < abs_g <= 1.0:
        return "1"
    if 1.4 < abs_g <= 1.5:
        return "1.5"
    if 1.9 < abs_g <= 2.0:
        return "2"
    if 2.9 < abs_g <= 3.0:
        return "3"
    if 3.9 < abs_g <= 4.0:
        return "4"
    return None


# --------------------------------------------------- correlation dimension (GP)
def correlation_integral(pts, enc_ids, times, r_grid, *, theiler_dt=THEILER_DT):
    """C(r) over pairs allowed by the Theiler window (same-encounter pairs with
    |dt| < theiler_dt excluded). Chunked pairwise distances, float32."""
    n = pts.shape[0]
    pts = np.ascontiguousarray(pts, dtype=np.float32)
    counts = np.zeros(len(r_grid), dtype=np.int64)
    n_allowed = 0
    chunk = 512
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        # pairs (i, j) with j > i only: restrict to the upper triangle
        d2 = ((pts[i0:i1, None, :] - pts[None, i0:, :]) ** 2).sum(axis=2)
        dist = np.sqrt(d2, dtype=np.float32)
        same = enc_ids[i0:i1, None] == enc_ids[None, i0:]
        close_t = np.abs(times[i0:i1, None] - times[None, i0:]) < theiler_dt
        allowed = ~(same & close_t)
        # upper triangle within the global index space
        rows = np.arange(i0, i1)[:, None]
        cols = np.arange(i0, n)[None, :]
        allowed &= cols > rows
        n_allowed += int(allowed.sum())
        d_ok = dist[allowed]
        for ri, r in enumerate(r_grid):
            counts[ri] += int((d_ok < r).sum())
    C = counts / max(n_allowed, 1)
    return C, n_allowed


def gp_dimension(pts, enc_ids, times, *, theiler_dt=THEILER_DT, n_r=40, seed=0,
                 slope_win=8, c_lo=5e-4, c_hi=0.3):
    """Grassberger-Procaccia d_eff: slope of log C(r) vs log r over the flattest
    local-slope window with C in [c_lo, c_hi]. Returns (d_eff, diag)."""
    rng = np.random.default_rng(seed)
    n = pts.shape[0]
    # distance scale from a subsample of allowed pairs
    m = min(n, 800)
    idx = rng.choice(n, size=m, replace=False)
    d2 = ((pts[idx, None, :] - pts[None, idx, :]) ** 2).sum(axis=2)
    dref = np.sqrt(d2[np.triu_indices(m, k=1)])
    dref = dref[dref > 0]
    r_grid = np.geomspace(np.percentile(dref, 0.5), np.percentile(dref, 99.5), n_r)
    C, n_allowed = correlation_integral(pts, enc_ids, times, r_grid, theiler_dt=theiler_dt)
    ok = C > 0
    logr, logC = np.log(r_grid[ok]), np.log(C[ok])
    if logr.size < slope_win + 2:
        return float("nan"), {"n_allowed": n_allowed, "note": "too few nonzero C(r)"}
    slopes = np.gradient(logC, logr)
    best, best_var = None, np.inf
    for s in range(0, len(slopes) - slope_win):
        seg = slice(s, s + slope_win)
        Cseg = np.exp(logC[seg])
        if Cseg.min() < c_lo or Cseg.max() > c_hi:
            continue
        v = float(np.var(slopes[seg]))
        if v < best_var:
            best_var, best = v, seg
    if best is None:  # fall back to the mid-range window
        s = max(0, len(slopes) // 2 - slope_win // 2)
        best = slice(s, s + slope_win)
        best_var = float(np.var(slopes[best]))
    d_eff = float(np.mean(slopes[best]))
    diag = {
        "n_points": int(n),
        "n_allowed_pairs": int(n_allowed),
        "slope_window_var": best_var,
        "slope_window_r": [float(np.exp(logr[best])[0]), float(np.exp(logr[best])[-1])],
        "slope_window_C": [float(np.exp(logC[best])[0]), float(np.exp(logC[best])[-1])],
        "logr": [float(x) for x in logr],
        "logC": [float(x) for x in logC],
    }
    return d_eff, diag


def stratum_gp(trajs, *, frames_sel, cap=4000, n_boot=150, seed=0):
    """d_eff for one stratum: pooled points from the selected frames of each
    encounter trajectory, with an encounter-resampling bootstrap CI."""
    rng = np.random.default_rng(seed)

    def collect(traj_ids):
        pts, eids, ts = [], [], []
        for e in traj_ids:
            z, f0, f1 = trajs[e]["z"], trajs[e][frames_sel][0], trajs[e][frames_sel][1]
            fr = np.arange(f0, f1 + 1)
            pts.append(z[fr])
            eids.append(np.full(len(fr), e))
            ts.append(fr)
        pts = np.concatenate(pts, axis=0)
        eids = np.concatenate(eids)
        ts = np.concatenate(ts)
        if pts.shape[0] > cap:
            sub = rng.choice(pts.shape[0], size=cap, replace=False)
            pts, eids, ts = pts[sub], eids[sub], ts[sub]
        return pts, eids, ts

    ids = list(range(len(trajs)))
    pts, eids, ts = collect(ids)
    d_eff, diag = gp_dimension(pts, eids, ts, seed=seed)
    boots = []
    for b in range(n_boot):
        pick = [ids[i] for i in rng.integers(0, len(ids), size=len(ids))]
        # re-tag encounters so Theiler still applies within each resampled copy
        p2, e2, t2 = [], [], []
        for j, e in enumerate(pick):
            z, f0, f1 = trajs[e]["z"], trajs[e][frames_sel][0], trajs[e][frames_sel][1]
            fr = np.arange(f0, f1 + 1)
            p2.append(z[fr])
            e2.append(np.full(len(fr), j))
            t2.append(fr)
        p2 = np.concatenate(p2, axis=0)
        e2 = np.concatenate(e2)
        t2 = np.concatenate(t2)
        if p2.shape[0] > cap:
            sub = rng.choice(p2.shape[0], size=cap, replace=False)
            p2, e2, t2 = p2[sub], e2[sub], t2[sub]
        db, _ = gp_dimension(p2, e2, t2, seed=seed + b + 1)
        if np.isfinite(db):
            boots.append(db)
    ci = (
        [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        if boots
        else [float("nan")] * 2
    )
    return {"d_eff": d_eff, "ci95": ci, "n_boot_ok": len(boots), "diag": diag}


# ------------------------------------------------------------ PCA cross-checks
def pca_checks(pts) -> dict:
    x = pts - pts.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    ev = s**2
    pr = float(ev.sum() ** 2 / (ev**2).sum())
    frac = np.cumsum(ev) / ev.sum()
    n90 = int(np.searchsorted(frac, 0.90) + 1)
    return {"participation_ratio": pr, "n_pc_90": n90}


# ----------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Track T3 effective dimension vs |G|")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--model", default="jepa_pool")
    p.add_argument("--run-dir", default="outputs/runs/session31/jepa_pool")
    p.add_argument("--checkpoint", default="checkpoint_iter010000.pt")
    p.add_argument("--grid-json", default="outputs/session33/track_t_recovery_grid.json")
    p.add_argument("--envelope-json", default="outputs/session32/envelope_by_gust.json")
    p.add_argument("--chi3d-json", default="outputs/session23/chi3d/chi3d.json")
    p.add_argument("--qdeim-taps", default="outputs/session32/qdeim_taps_v2p2.json")
    p.add_argument("--train-latents",
                   default="outputs/session31/q1_latents/latents_jepa_pool_train.npz")
    p.add_argument("--latents-npz", default="outputs/session33/t3_latents.npz")
    p.add_argument("--partition", default="v2p2")
    p.add_argument("--pipeline-manifest", default="outputs/data_pipeline/v2p2/manifest.json")
    p.add_argument("--split", default="configs/splits/split_v2p2.json")
    p.add_argument("--windows", default="outputs/session31/windows_v2p2.json")
    p.add_argument("--cap", type=int, default=4000)
    p.add_argument("--n-boot", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=8, help="tap count for the bound 2*d_eff/K")
    p.add_argument("--out", default="outputs/session33/track_t3_effective_dimension.json")
    return p.parse_args(argv)


def encode_all(a, device):
    """Encode all 450 encounters (all splits) with the frozen pooled encoder;
    cache trajectories + pressure to --latents-npz."""
    from src.data.omega_pipeline import OmegaPipeline
    from src.evaluation.rom_eval import load_frozen_model
    from scripts.session32.envelope_by_gust import enumerate_encounters
    from scripts.session32.track_b_pilot import PRE, POST, _cache_dir, encode_encounters

    npz_path = _resolve(a.latents_npz)
    if npz_path.exists():
        d = np.load(npz_path, allow_pickle=True)
        print(f"[t3] reusing {npz_path} ({d['z'].shape[0]} encounters)", flush=True)
        return d

    windows = json.loads(_resolve(a.windows).read_text())["windows"]
    pipe = OmegaPipeline.from_manifest(_resolve(a.pipeline_manifest))
    cache_dir = _cache_dir(a.partition)
    frozen = load_frozen_model(_resolve(a.run_dir), a.checkpoint, device)
    metas = enumerate_encounters(_resolve(a.split))
    zs, ps, rows = [], [], []
    t0 = time.time()
    for i, m in enumerate(metas):
        enc = encode_encounters(
            frozen, [(m["case_id"], m["encounter_index"])], pipe, cache_dir, windows, device
        )[0]
        t_imp = int(enc["t_impact"])
        f0 = max(0, t_imp - PRE)
        f1 = min(enc["n_frames"] - 1, t_imp + POST)
        zs.append(enc["z_gap"].astype(np.float32))
        ps.append(enc["p_wall"].astype(np.float32))
        rows.append(
            (m["case_id"], m["encounter_index"], m["split"], m["G"], m["D"], t_imp, f0, f1)
        )
        if (i + 1) % 50 == 0 or (i + 1) == len(metas):
            print(f"[t3] encoded {i+1}/{len(metas)} ({(i+1)/(time.time()-t0):.2f}/s)", flush=True)
    n_frames = min(z.shape[0] for z in zs)
    z = np.stack([x[:n_frames] for x in zs])
    p = np.stack([x[:n_frames] for x in ps])
    meta = np.array(rows, dtype=object)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        z=z,
        p_wall=p,
        case_id=np.array([r[0] for r in rows]),
        encounter_index=np.array([int(r[1]) for r in rows]),
        split=np.array([r[2] for r in rows]),
        G=np.array([float(r[3]) for r in rows]),
        D=np.array([float(r[4]) for r in rows]),
        t_impact=np.array([int(r[5]) for r in rows]),
        f0=np.array([int(r[6]) for r in rows]),
        f1=np.array([int(r[7]) for r in rows]),
    )
    print(f"[t3] wrote {npz_path}", flush=True)
    del meta
    return np.load(npz_path, allow_pickle=True)


def m_needed_from_curve(curve: dict, target: float):
    """Smallest W whose stratum state R2 >= target; None if never met."""
    for w in sorted(int(k) for k in curve.keys()):
        v = curve[str(w)]
        if v is not None and np.isfinite(v) and v >= target:
            return w
    return None


def testc_recovery_curve(d, taps, ws, *, seed, n_components=400):
    """CHARACTERISATION-ONLY |G|=4 column: KRR recovery fit on TRAIN encounter
    windows, applied to test_c; stratum state R2 over the assimilation window."""
    from src.evaluation.pressure_infer import fit_pressure_estimator
    from scripts.session32.track_b_pilot import windowed_features

    split = d["split"].astype(str)
    tr = np.where(split == "train")[0]
    tc = np.where(split == "test_c")[0]
    rng = np.random.default_rng(seed)
    out = {}
    for w in ws:
        # per-encounter causal windows at the qDEIM K8 taps
        def rows_of(idx_set):
            X, Z, M = [], [], []
            for e in idx_set:
                p_t = d["p_wall"][e][:, np.sort(taps)]
                win = windowed_features(p_t, w)
                f0, f1 = int(d["f0"][e]), int(d["f1"][e])
                fr = np.arange(f0, f1 + 1)
                X.append(win[fr])
                Z.append(d["z"][e][fr])
                M.append(np.full(len(fr), e))
            return np.concatenate(X), np.concatenate(Z), np.concatenate(M)

        Xtr, Ztr, Gtr = rows_of(tr)
        sub = rng.choice(len(Xtr), size=min(9000, len(Xtr)), replace=False)
        est = fit_pressure_estimator(
            Xtr[sub], Ztr[sub], n_components=n_components, seed=seed,
            groups=Gtr[sub].astype(str),
        )
        Xtc, Ztc, _ = rows_of(tc)
        zh = np.asarray(est.predict(Xtc), dtype=np.float64)
        mu = Ztc.mean(axis=0, keepdims=True)
        den = float(((Ztc - mu) ** 2).sum())
        r2 = float(1.0 - ((zh - Ztc) ** 2).sum() / den) if den > 0 else float("nan")
        out[str(w)] = r2
        print(f"[t3] test_c KRR recovery K8 W={w}: state R2={r2:.3f}", flush=True)
    return out


def main(argv=None):
    import torch

    a = parse_args(argv)
    from src.utils.device import require_rtx6000

    device = require_rtx6000(gpu_index=a.gpu)
    gpu_name = torch.cuda.get_device_name(device.index)
    if "RTX" not in gpu_name or "6000" not in gpu_name:
        raise RuntimeError(f"hardware policy: gpu_name={gpu_name!r} is not an RTX 6000")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    d = encode_all(a, device)
    abs_g = np.abs(d["G"].astype(float))
    buckets = np.array([bucket_of(g) or "other" for g in abs_g])

    # ---- effective dimension per stratum (assim window primary; full-traj alt)
    strata = {}
    for b in G_ORDER:
        idx = np.where(buckets == b)[0]
        if idx.size == 0:
            continue
        trajs = [
            {"z": d["z"][e], "assim": (int(d["f0"][e]), int(d["f1"][e])),
             "full": (0, d["z"][e].shape[0] - 1)}
            for e in idx
        ]
        t0 = time.time()
        gp_assim = stratum_gp(trajs, frames_sel="assim", cap=a.cap, n_boot=a.n_boot, seed=a.seed)
        gp_full = stratum_gp(trajs, frames_sel="full", cap=a.cap, n_boot=0, seed=a.seed)
        pts = np.concatenate(
            [d["z"][e][int(d["f0"][e]): int(d["f1"][e]) + 1] for e in idx], axis=0
        )
        strata[b] = {
            "n_encounters": int(idx.size),
            "gp_assim_window": gp_assim,
            "gp_full_trajectory": {k: gp_full[k] for k in ("d_eff", "diag")},
            **pca_checks(pts),
        }
        ci_txt = [round(x, 2) for x in gp_assim["ci95"]]
        print(
            f"[t3] |G|={b}: d_eff={gp_assim['d_eff']:.2f} CI{ci_txt} "
            f"(full-traj {gp_full['d_eff']:.2f}) PR={strata[b]['participation_ratio']:.2f} "
            f"nPC90={strata[b]['n_pc_90']} [{time.time()-t0:.0f}s]",
            flush=True,
        )

    # ---- chi3d column (harvested, per-|G| medians of max_chi3d_wz_post)
    chi = json.loads(_resolve(a.chi3d_json).read_text())
    chi_by_bucket = {}
    for b in G_ORDER:
        vals = [
            r["max_chi3d_wz_post"] for r in chi["records"]
            if bucket_of(abs(float(r["G"]))) == b and not r.get("missing_flag")
        ]
        chi_by_bucket[b] = {
            "median": float(np.median(vals)) if vals else None,
            "n_cases": len(vals),
        }

    # ---- m_needed(K=8) per stratum from the grid curves
    grid = json.loads(_resolve(a.grid_json).read_text())
    ws = sorted(grid["params"]["grid_Ws"])
    k8 = max(grid["params"]["grid_Ks"])
    curves = {}
    for b in G_ORDER:
        curve = {}
        for w in ws:
            cell = grid["cells"].get(f"K{k8}_W{w}", {})
            v = cell.get("per_G_window", {}).get(b, {}).get("state_r2")
            curve[str(w)] = v
        if any(v is not None for v in curve.values()):
            curves[b] = curve
    # |G|=4 characterisation-only column (test_c; grid has no test_c rows)
    qdeim = json.loads(_resolve(a.qdeim_taps).read_text())
    taps = [int(t) for t in qdeim["K8"]]
    if "4" not in curves or all(v is None for v in curves.get("4", {}).values()):
        curves["4"] = testc_recovery_curve(d, taps, ws, seed=a.seed)
        curves["4"]["note"] = "characterisation-only: KRR fit on train, applied to test_c (D236)"

    # ---- bound overlay + gate
    env = json.loads(_resolve(a.envelope_json).read_text())
    env_thr = env["models"][a.model]["thresholds"]
    table = {}
    for b in G_ORDER:
        if b not in strata:
            continue
        curve = {k: v for k, v in curves.get(b, {}).items() if k != "note"}
        sat = curve.get("30")
        d_eff = strata[b]["gp_assim_window"]["d_eff"]
        table[b] = {
            "d_eff": d_eff,
            "d_eff_ci95": strata[b]["gp_assim_window"]["ci95"],
            "participation_ratio": strata[b]["participation_ratio"],
            "n_pc_90": strata[b]["n_pc_90"],
            "chi3d_wz_post_median": chi_by_bucket.get(b, {}).get("median"),
            "recovery_curve_K8": curve,
            "m_needed_rel90": (
                m_needed_from_curve(curve, 0.9 * sat)
                if sat is not None and np.isfinite(sat) else None
            ),
            "m_needed_abs0.5": m_needed_from_curve(curve, 0.5),
            "bound_2deff_over_K": float(2.0 * d_eff / a.k) if np.isfinite(d_eff) else None,
        }

    def _series(key):
        return [(b, table[b][key]) for b in G_ORDER if b in table and table[b][key] is not None]

    deffs = _series("d_eff")
    mneed = _series("m_needed_abs0.5")
    d_eff_grows = bool(
        len(deffs) >= 3 and deffs[-1][1] > deffs[0][1]
    )
    gate = {
        "d_eff_series": deffs,
        "m_needed_abs0.5_series": mneed,
        "d_eff_grows_with_G": d_eff_grows,
        "envelope_divergence_boundary": {
            dn: env_thr[dn]["div_rate_exceeds_50pct_at_absG"] for dn in env_thr
        },
        "verdict": "descriptive; see consistency fields",
    }

    payload = {
        "task": "SESSION 33 Track T3 -- effective dimension vs |G| and the embedding bound",
        "params": {
            "gpu_name": gpu_name,
            "model": a.model,
            "theiler_dt_frames": THEILER_DT,
            "cap_points_per_stratum": a.cap,
            "n_boot_encounters": a.n_boot,
            "seed": a.seed,
            "K_for_bound": a.k,
            "gp_estimator": (
                "Grassberger-Procaccia; slope of log C(r) over the flattest "
                "local-slope window with C in [5e-4, 0.3]; bounded estimate, "
                "not a fitted constant (addendum caveat)"
            ),
            "frames": "primary = assimilation window [t_imp-24, t_imp+48]; alt = full trajectory",
            "chi3d_source": (
                f"{a.chi3d_json} max_chi3d_wz_post per case (Section 2.1 convention); "
                "84 v2-era cases; the 17 run4 cases (G_inv=+4 Test C half) are absent"
            ),
            "m_needed_targets": "rel90 = 0.9 x stratum W=30 recovery; abs0.5 = R2 >= 0.5",
        },
        "strata": strata,
        "chi3d_by_bucket": chi_by_bucket,
        "recovery_curves_K8": curves,
        "table_T3": table,
        "gate_T3": gate,
    }
    out_path = _resolve(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=float))

    print("\n[t3] ===== TABLE T3 =====", flush=True)
    for b, row in table.items():
        print(
            f"  |G|={b:8s} d_eff={row['d_eff']:.2f} CI{[round(x, 2) for x in row['d_eff_ci95']]} "
            f"PR={row['participation_ratio']:.2f} nPC90={row['n_pc_90']} "
            f"chi3d={row['chi3d_wz_post_median']} "
            f"m_needed(abs0.5)={row['m_needed_abs0.5']} m_needed(rel90)={row['m_needed_rel90']} "
            f"bound 2d/K={row['bound_2deff_over_K']:.2f}",
            flush=True,
        )
    print(
        f"[t3] GATE T3 (descriptive): d_eff grows with |G|: {gate['d_eff_grows_with_G']}; "
        f"divergence boundary per D: {gate['envelope_divergence_boundary']}",
        flush=True,
    )
    print(f"[t3] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
