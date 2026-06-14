"""SESSION29 Track I: strictly CAUSAL pressure -> state recovery (referee F6), v2.1.

THE F6 CRITIQUE. The v2.1 sensing headline (scripts/session28/sensing_cf.py) maps
a W=30 wall-pressure window ENDING at the readout instant impact+16 to the readout
latent z(impact+16). That window spans frames [impact-14, impact+16): it extends 16
frames PAST impact, so the estimator sees wall pressure up to and including the
target time. It is therefore NOT a strictly pre-impact sensing claim. This track
re-runs pressure -> state recovery under STRICTLY CAUSAL windows and asks whether
the headline ordering (the predictive JEPA latent is the MOST recoverable state,
jepa state R^2 ~ 0.78 at K=8) SURVIVES a window that ends at or before impact.

REGIME / PROBE DECLARATION (CLAUDE.md probe methodology). The pressure -> latent map
is a STATE-RECOVERY map (same role as sensing_cf): a KernelRidge(RBF) regressor from
the flattened (K*W) wall-pressure window to the readout latent z(impact+H). It is NOT
a (G, D, Y) parameter probe and not a per-frame state-descriptor probe; the parameter
and lift readouts below are reported alongside it as context. The downstream wake
readout is the FROZEN closure-matrix wake probe (a ridge on PER-frame z fitted once on
TRAIN, never refit on pressure) and the wake is the KNOWN NON-RESULT: it is reported
but carries no recovery claim. The map is fitted on TRAIN cases and evaluated once on
test_b with a case-clustered bootstrap CI; the KRR regularisation is selected by
GroupKFold(5)-by-case so no case straddles the tuning folds (the probe-hygiene rule).

WINDOWS COMPARED (all W frames, all at the SAME K=8 qDEIM taps; only the window
POSITION varies, so the comparison isolates causality):
  preimpact_m30_to_m1   frames [impact-30, impact)   STRICTLY pre-impact (causal).
  through_impact_m29_to_0 frames [impact-29, impact+1) ends AT impact (causal-ish).
  readout_m13_to_p16    frames [impact+H-W, impact+H) = the CURRENT sensing_cf
                        window, ending AT the readout (NON-causal baseline).

PLACEMENT. qDEIM (target-blind, Drmac & Gugercin 2016; Manohar et al. 2018) on the
STRICTLY pre-impact window-mean snapshot over TRAIN encounters: target-blind AND
causal, family-independent, computed ONCE and reused for every window so the only
thing that changes between windows is which frames feed the K=8 taps.

TARGETS (each scored as variance-weighted state-recovery R^2 on the multi-output, or
plain R^2 on a scalar; cm.case_clustered_r2_ci on the held-out residuals):
  latent_state  PRIMARY: the d=64 readout latent z(impact+H), variance-weighted R^2.
  params        [G, D, Y] (3 outputs), variance-weighted R^2 (context).
  cl_impact     impact C_L = C_L at the readout frame impact+H (scalar) (context).
  wake          DNS wake_enstrophy at the readout via the FROZEN closure wake probe;
                NON-RESULT, reported with NO recovery claim.

VERDICT. STRONG: the predictive latent (jepa_tf_noc) stays the MOST recoverable
latent_state under the strictly-causal preimpact_m30_to_m1 window -> pressure stays a
MAIN result, now reported with a causal window. WEAK: the predictive-latent ordering
holds only for the non-causal readout window -> pressure moves to the APPENDIX as a
latent-state recoverability result, and the wake non-result stays in the text.

Statistics reuse: scripts/session28/stats_lib (via cm.stats_lib), the v2.1 family
latents + DNS observables via cm._s29_common, the qDEIM selector + KRR map +
variance-weighted R^2 from scripts/session28/sensing_cf (imported, not reimplemented).

CPU-only (numpy + sklearn). Run niced:
    export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
    nice -n 15 python scripts/session29/pressure_causal.py
    nice -n 15 python scripts/session29/pressure_causal.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import _s29_common as cm

REPO = cm.REPO
sys.path.insert(0, str(REPO / "scripts" / "session28"))
import sensing_cf as scf  # noqa: E402  (qDEIM selector + variance-weighted R^2)
from closure_matrix import apply_probe, fit_ridge  # noqa: E402  (frozen wake probe)

OUT_JSON = REPO / "outputs" / "session29" / "pressure_causal.json"
OUT_MD = REPO / "outputs" / "session29" / "pressure_causal.md"

# Families: the predictive JEPA, the reconstructive Fukami AE, and the linear POD
# floor. (bvae optional; not in the v2.1 frozen FAMILY_TAGS, so omitted by default.)
FAMILIES = ["jepa_tf_noc", "fukami", "pod"]
PREDICTIVE_FAMILY = "jepa_tf_noc"

K = 8  # qDEIM taps for the whole comparison (the headline sensing budget)
W = 30  # window length in frames (matches sensing_cf WINDOW)
READOUT_H = 16  # readout horizon past impact (matches the closure-matrix endpoint)
N_PRESSURE = scf.N_PRESSURE  # 192 span-averaged surface taps

# (lo_offset, hi_offset) RELATIVE to the per-encounter impact frame; the window is
# frames [impact + lo_offset, impact + hi_offset) (hi exclusive), W = hi - lo frames.
WINDOWS = {
    "preimpact_m30_to_m1": (-W, 0),  # [impact-30, impact)      strictly pre-impact
    "through_impact_m29_to_0": (-(W - 1), 1),  # [impact-29, impact+1) ends AT impact
    "readout_m13_to_p16": (READOUT_H - W, READOUT_H),  # [impact-14, impact+16) sensing_cf
}
CAUSAL_WINDOW = "preimpact_m30_to_m1"  # the verdict-defining strictly-causal window

KRR_ALPHA_GRID = [0.01, 0.1, 1.0, 10.0]
KRR_GAMMA_GRID = [1e-3, 1e-2, 1e-1]


# --------------------------------------------------------------------- window indices


def window_indices(impact: int, window: str, n_frames: int = 120) -> np.ndarray:
    """Frame indices [impact+lo, impact+hi) for ``window`` (hi exclusive).

    Fails loudly if the window runs off either end of the trajectory (no silent
    left-pad: every v2.1 encounter has impact=40 and 120 frames, so all three
    windows are strictly in-bounds; a violation means a corrupted alignment).
    """
    if window not in WINDOWS:
        raise ValueError(f"unknown window {window!r}; have {list(WINDOWS)}")
    lo_off, hi_off = WINDOWS[window]
    lo, hi = impact + lo_off, impact + hi_off
    if lo < 0 or hi > n_frames:
        raise ValueError(
            f"window {window!r} frames [{lo},{hi}) out of bounds for impact={impact}, "
            f"n_frames={n_frames}"
        )
    return np.arange(lo, hi, dtype=int)


def is_strictly_preimpact(window: str) -> bool:
    """True iff every frame of ``window`` is strictly before the impact frame.

    The window [impact+lo, impact+hi) is strictly pre-impact iff its last frame
    impact+hi-1 < impact, i.e. hi <= 0.
    """
    _, hi_off = WINDOWS[window]
    return hi_off <= 0


# --------------------------------------------------------------------- loaders


def _cache_root() -> Path:
    root = os.environ.get("VORTEX_JEPA_CACHE")
    if not root:
        prevent = os.environ.get("PREVENT_ROOT")
        if not prevent:
            raise EnvironmentError(
                "neither VORTEX_JEPA_CACHE nor PREVENT_ROOT is set; cannot locate "
                "the v2.1 pressure cache"
            )
        root = str(Path(prevent) / "data" / "processed" / "vortex-jepa")
    return Path(root) / "v2p1"


def _dns_maps(split: str):
    """DNS per-frame wake_enstrophy + C_L maps, keyed by (case_id, encounter)."""
    wake, _ = cm.load_dns_observable(split, "wake_enstrophy")
    cl, _ = cm.load_dns_observable(split, "C_L")
    return wake, cl


def load_split(tag: str, split: str, window: str):
    """Assemble one family/split/window bundle.

    Returns a dict with:
      Xwin  (n, W, 192)  the pressure window (all 192 taps; taps sub-selected later)
      z     (n, d)       readout latent z(impact+H)
      z_pf  (n, T, d)    per-frame latents (for the frozen wake probe fit on TRAIN)
      G,D,Y (n,)         gust parameters (frame-independent)
      cl    (n,)         DNS C_L at the readout frame
      wake_rd (n,)       DNS wake_enstrophy at the readout frame
      wake_pf (n, T)     DNS wake per frame (for the frozen wake probe fit)
      case_ids (n,)      case id per encounter (clustering unit)
      enc   (n,)         encounter index
    """
    fam = cm.load_family(tag, split)
    cid = fam["case_ids"]
    enc = fam["encounter_indices"]
    imp = fam["impact_frame"]
    z_full = fam["z_full"].astype(np.float64)  # (n, T, d)
    T = z_full.shape[1]
    npz = np.load(cm.LATENTS / tag / f"{split}.npz", allow_pickle=True)
    G = npz["G"].astype(np.float64)
    D = npz["D"].astype(np.float64)
    Y = npz["Y"].astype(np.float64)

    wake_map, cl_map = _dns_maps(split)
    cache = _cache_root()

    Xwin, zrd, zpf, gg, dd, yy, clrd, wrd, wpf, cc, ee = ([] for _ in range(11))
    for i in range(len(cid)):
        key = (str(cid[i]), int(enc[i]))
        if key not in wake_map or key not in cl_map:
            continue
        impact = int(imp[i])
        readout = impact + READOUT_H
        if readout >= T:
            continue
        idx = window_indices(impact, window, n_frames=T)
        h5 = cache / str(cid[i]) / f"encounter_{int(enc[i]):02d}.h5"
        if not h5.exists():
            raise FileNotFoundError(f"missing pressure cache: {h5}")
        with h5py.File(h5, "r") as f:
            if "p_wall" not in f:
                raise KeyError(f"no p_wall in {h5}")
            pw = np.asarray(f["p_wall"], dtype=np.float64)  # (120, 192)
        Xwin.append(pw[idx])  # (W, 192)
        zrd.append(z_full[i, readout])
        zpf.append(z_full[i])
        gg.append(float(G[i]))
        dd.append(float(D[i]))
        yy.append(float(Y[i]))
        clrd.append(float(cl_map[key][readout]))
        wrd.append(float(wake_map[key][readout]))
        wpf.append(np.asarray(wake_map[key], dtype=np.float64))
        cc.append(str(cid[i]))
        ee.append(int(enc[i]))

    if not Xwin:
        raise ValueError(f"{tag}/{split}/{window}: no aligned encounters")
    bundle = dict(
        Xwin=np.asarray(Xwin),
        z=np.asarray(zrd),
        z_pf=np.asarray(zpf),
        G=np.asarray(gg),
        D=np.asarray(dd),
        Y=np.asarray(yy),
        cl=np.asarray(clrd),
        wake_rd=np.asarray(wrd),
        wake_pf=np.asarray(wpf),
        case_ids=np.asarray(cc),
        enc=np.asarray(ee),
    )
    if not np.isfinite(bundle["Xwin"]).all():
        raise ValueError(f"{tag}/{split}/{window}: non-finite pressure window")
    return bundle


def feats(Xwin: np.ndarray, taps) -> np.ndarray:
    """Flatten the (n, W, K) sub-window into the (n, K*W) regressor vector."""
    return Xwin[:, :, list(taps)].reshape(len(Xwin), -1)


# --------------------------------------------------------------------- placement


def causal_qdeim_taps(Xtr_win: np.ndarray, k: int) -> list[int]:
    """Target-blind, CAUSAL qDEIM placement on the window-mean TRAIN snapshot.

    The snapshot is the per-encounter mean over the window's frames (n_train, 192).
    With the strictly pre-impact window this is also causal, so the placement never
    sees post-impact pressure. Reuses sensing_cf.selector_qdeim verbatim.
    """
    snapshot = Xtr_win.mean(axis=1)  # (n_train, 192)
    return scf.selector_qdeim(snapshot, k)


# --------------------------------------------------------------------- KRR map


def fit_krr_map(Xtr: np.ndarray, Ytr: np.ndarray, groups: np.ndarray, seed: int = 0):
    """KernelRidge(RBF) map with (alpha, gamma) chosen by GroupKFold(5)-by-case.

    Multi-output safe: Ytr may be (n,) or (n, m). Standardises in/out. The grid
    search groups by case so no case straddles the tuning folds (probe hygiene).
    Returns a fitted (estimator, scaler_y) pair plus the selected params.
    """
    Y2 = Ytr.reshape(len(Ytr), -1)
    sy = StandardScaler().fit(Y2)
    Yn = sy.transform(Y2)
    pipe = Pipeline([("scale", StandardScaler()), ("model", KernelRidge(kernel="rbf"))])
    grid = {"model__alpha": KRR_ALPHA_GRID, "model__gamma": KRR_GAMMA_GRID}
    n_groups = len(np.unique(groups))
    cv = GroupKFold(n_splits=min(5, n_groups))
    # GridSearchCV with multioutput r2 averages over outputs; that is the
    # variance-unweighted r2, used ONLY for hyperparameter selection. The reported
    # held-out score is the variance-weighted R^2 below.
    gs = GridSearchCV(pipe, grid, scoring="r2", cv=cv, n_jobs=-1)
    gs.fit(Xtr, Yn, groups=groups)
    return gs.best_estimator_, sy, dict(gs.best_params_)


def krr_predict(model, sy, X: np.ndarray) -> np.ndarray:
    return sy.inverse_transform(model.predict(X).reshape(len(X), -1))


# --------------------------------------------------------------------- scoring


def vw_r2_ci(y_pred: np.ndarray, y_true: np.ndarray, case_ids: np.ndarray, seed: int = 0):
    """Variance-weighted R^2 point + case-clustered bootstrap CI (multi-output).

    For a scalar target this is the ordinary R^2 with a case-clustered CI; for a
    multi-output target it is scf.variance_weighted_r2 (pooled SSE / pooled SST),
    bootstrapped by resampling CASES.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(len(y_pred), -1)
    y_true = np.asarray(y_true, dtype=np.float64).reshape(len(y_true), -1)
    point = scf.variance_weighted_r2(y_pred, y_true)
    rng = np.random.default_rng(seed)
    cases = np.unique(case_ids)
    idx_by_case = {c: np.where(case_ids == c)[0] for c in cases}
    boots = []
    for _ in range(2000):
        pick = rng.choice(cases, size=len(cases), replace=True)
        sel = np.concatenate([idx_by_case[c] for c in pick])
        boots.append(scf.variance_weighted_r2(y_pred[sel], y_true[sel]))
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return float(point), lo, hi


# --------------------------------------------------------------------- verdict


def decide_verdict(state_r2: dict, causal_window: str = CAUSAL_WINDOW) -> dict:
    """Does the predictive-latent ordering survive the strictly-causal window?

    ``state_r2`` is {window: {family: latent_state R^2 on test_b}}. The ordering
    "holds" for a window iff PREDICTIVE_FAMILY has the strictly highest state R^2
    among all families for that window. STRONG iff it holds under the causal
    window; WEAK if it holds only for the non-causal readout window (or not at all
    under the causal window). Pure function -> unit-testable without real latents.
    """
    if causal_window not in state_r2:
        raise ValueError(f"causal window {causal_window!r} absent from state_r2")

    def ordering_holds(win: str) -> bool:
        fams = state_r2[win]
        if PREDICTIVE_FAMILY not in fams:
            return False
        pred = fams[PREDICTIVE_FAMILY]
        return all(pred > v for f, v in fams.items() if f != PREDICTIVE_FAMILY)

    holds = {win: ordering_holds(win) for win in state_r2}
    causal_holds = holds[causal_window]
    branch = "STRONG" if causal_holds else "WEAK"
    if branch == "STRONG":
        recommendation = (
            "Pressure -> state recovery stays a MAIN result, now reported with the "
            f"strictly-causal {causal_window} window."
        )
    else:
        recommendation = (
            "Move pressure to the APPENDIX as a latent-state recoverability result "
            "(the predictive-latent ordering does not survive a strictly-causal "
            "pre-impact window); keep the wake non-result in the text."
        )
    return {
        "predictive_family": PREDICTIVE_FAMILY,
        "causal_window": causal_window,
        "ordering_holds_per_window": holds,
        "ordering_holds_under_causal_window": bool(causal_holds),
        "branch": branch,
        "recommendation": recommendation,
        "note": (
            "STRONG iff the predictive latent (jepa_tf_noc) has the strictly highest "
            "variance-weighted latent-state R^2 (test_b) under the strictly-causal "
            f"{causal_window} window. case-clustered R^2 CIs are wide at 10 test_b "
            "cases; weigh the sign/ordering consistency alongside the point R^2."
        ),
    }


# --------------------------------------------------------------------- main


def run(args) -> dict:
    """Fit the pressure -> {state, params, cl, wake} maps for every family x window."""
    results: dict = {win: {} for win in args.windows}
    # qDEIM placement: target-blind + causal, computed ONCE on the causal window's
    # TRAIN window-mean snapshot, reused for every window so only the window POSITION
    # varies. (The cache pressure is family-independent; use jepa's alignment.)
    place_bundle = load_split(cm.FAMILY_TAGS[PREDICTIVE_FAMILY], "train", CAUSAL_WINDOW)
    taps = causal_qdeim_taps(place_bundle["Xwin"], args.k)
    print(f"[pressure_causal] causal qDEIM K={args.k} taps (from {CAUSAL_WINDOW}): {taps}")

    for win in args.windows:
        for fam in args.families:
            tag = cm.FAMILY_TAGS[fam]
            tr = load_split(tag, "train", win)
            te = load_split(tag, "test_b", win)
            Xtr = feats(tr["Xwin"], taps)
            Xte = feats(te["Xwin"], taps)

            # FROZEN closure wake probe: ridge on TRAIN per-frame z -> per-frame wake,
            # fitted once, never refit on pressure (reused verbatim, never re-tuned).
            zf = tr["z_pf"].reshape(-1, tr["z_pf"].shape[2])
            wake_probe = fit_ridge(zf, tr["wake_pf"].reshape(-1), alpha=1.0)

            fam_rec: dict = {}

            # --- PRIMARY: pressure -> readout latent z(impact+H), vw-R^2 ---
            m, sy, params = fit_krr_map(Xtr, tr["z"], tr["case_ids"], seed=args.seed)
            zhat = krr_predict(m, sy, Xte)
            r2, lo, hi = vw_r2_ci(zhat, te["z"], te["case_ids"], seed=args.seed)
            fam_rec["latent_state"] = dict(
                r2=r2, ci_lo=lo, ci_hi=hi, krr_params=params, d=int(te["z"].shape[1])
            )

            # --- context: pressure -> [G, D, Y], vw-R^2 ---
            P_tr = np.stack([tr["G"], tr["D"], tr["Y"]], axis=1)
            P_te = np.stack([te["G"], te["D"], te["Y"]], axis=1)
            mp, syp, pp = fit_krr_map(Xtr, P_tr, tr["case_ids"], seed=args.seed)
            r2p, lop, hip = vw_r2_ci(
                krr_predict(mp, syp, Xte), P_te, te["case_ids"], seed=args.seed
            )
            fam_rec["params"] = dict(r2=r2p, ci_lo=lop, ci_hi=hip, krr_params=pp)

            # --- context: pressure -> impact C_L (scalar) ---
            mc, syc, pc = fit_krr_map(Xtr, tr["cl"], tr["case_ids"], seed=args.seed)
            r2c, loc, hic = vw_r2_ci(
                krr_predict(mc, syc, Xte).ravel(), te["cl"], te["case_ids"], seed=args.seed
            )
            fam_rec["cl_impact"] = dict(r2=r2c, ci_lo=loc, ci_hi=hic, krr_params=pc)

            # --- NON-RESULT: pressure -> z -> frozen wake probe (reported, no claim) ---
            wake_hat = apply_probe(zhat, wake_probe)
            r2w, low, hiw = vw_r2_ci(wake_hat, te["wake_rd"], te["case_ids"], seed=args.seed)
            fam_rec["wake_via_latent"] = dict(
                r2=r2w,
                ci_lo=low,
                ci_hi=hiw,
                claim=False,
                note="known NON-RESULT; reported but no recovery claim is made.",
            )

            fam_rec["n_train"] = int(len(tr["z"]))
            fam_rec["n_test_b"] = int(len(te["z"]))
            fam_rec["n_cases_test_b"] = int(len(np.unique(te["case_ids"])))
            results[win][fam] = fam_rec
            print(
                f"  {win:24s} {fam:12s} state R2={r2:+.3f} [{lo:+.3f},{hi:+.3f}]  "
                f"params={r2p:+.3f} cl={r2c:+.3f} wake(NR)={r2w:+.3f}"
            )

    state_r2 = {
        win: {fam: results[win][fam]["latent_state"]["r2"] for fam in results[win]}
        for win in results
    }
    verdict = decide_verdict(state_r2, causal_window=CAUSAL_WINDOW)
    return {"results": results, "state_r2_table": state_r2, "verdict": verdict, "taps": taps}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--windows", nargs="+", default=list(WINDOWS), choices=list(WINDOWS))
    ap.add_argument("--k", type=int, default=K, help="qDEIM tap budget.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        print("[dry-run] families:", args.families)
        print("[dry-run] windows:")
        for w in args.windows:
            lo, hi = WINDOWS[w]
            idx = window_indices(40, w)
            print(
                f"  {w:24s} offsets=[{lo},{hi})  impact=40 -> frames "
                f"[{idx[0]},{idx[-1]}]  strictly_preimpact={is_strictly_preimpact(w)}"
            )
        print(f"[dry-run] K={args.k} qDEIM taps, readout horizon H={READOUT_H}, W={W}")
        print(f"[dry-run] causal (verdict) window: {CAUSAL_WINDOW}")
        place = load_split(cm.FAMILY_TAGS[PREDICTIVE_FAMILY], "train", CAUSAL_WINDOW)
        print(
            f"[dry-run] causal-window TRAIN bundle: Xwin{place['Xwin'].shape} "
            f"z{place['z'].shape} cases={len(np.unique(place['case_ids']))}"
        )
        print("[dry-run] qDEIM taps:", causal_qdeim_taps(place["Xwin"], args.k))
        return

    inputs = [cm.LATENTS / cm.FAMILY_TAGS[f] / "train.npz" for f in args.families]
    inputs += [cm.TARGETS / "train.npz", cm.TARGETS / "test_b.npz"]

    out = run(args)
    payload = {
        "_provenance": cm.provenance(inputs, seed=args.seed),
        "config": {
            "question": "F6: does pressure->state recovery survive a strictly causal window?",
            "K": args.k,
            "W": W,
            "readout_horizon_H": READOUT_H,
            "windows": {w: list(WINDOWS[w]) for w in args.windows},
            "causal_window": CAUSAL_WINDOW,
            "placement": "target-blind causal qDEIM on the pre-impact window-mean snapshot",
            "map": "KernelRidge(RBF), (alpha,gamma) by GroupKFold(5)-by-case",
            "primary_target": "latent_state (variance-weighted R^2)",
            "regime": "fit train, eval test_b, case-clustered bootstrap CI (n=2000)",
            "wake": "frozen closure wake probe; reported as a NON-RESULT (no claim)",
        },
        "results": out["results"],
        "state_r2_table": out["state_r2_table"],
        "qdeim_taps": out["taps"],
        "verdict": out["verdict"],
    }
    cm.write_artifact(OUT_JSON, payload)
    _write_md(args, out)
    v = out["verdict"]
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_MD}")
    print(
        f"VERDICT: {v['branch']} | predictive-latent ordering under causal window "
        f"({CAUSAL_WINDOW}): {v['ordering_holds_under_causal_window']}"
    )
    print(f"  {v['recommendation']}")


def _write_md(args, out) -> None:
    res = out["results"]
    v = out["verdict"]
    lines = [
        "# Track I: strictly causal pressure -> state recovery (F6, v2.1, test_b)\n",
        "Pressure window (K={} qDEIM taps, W={} frames) -> readout latent "
        "z(impact+{}); KernelRidge(RBF), (alpha,gamma) by GroupKFold(5)-by-case, fit "
        "on train, evaluated once on test_b with a case-clustered bootstrap CI. The "
        "qDEIM placement is target-blind and computed on the strictly pre-impact "
        "window-mean snapshot (causal), then reused for every window so only the "
        "window POSITION varies.\n".format(args.k, W, READOUT_H),
        "Targets: **latent_state** (PRIMARY, variance-weighted R^2 over d outputs), "
        "params [G,D,Y], impact C_L (context), and wake enstrophy via the frozen "
        "closure probe (a known NON-RESULT, reported with no recovery claim).\n",
        "## latent_state R^2 (PRIMARY) -- family x window, test_b case-clustered CI",
        "",
        "| window | strictly pre-impact | "
        + " | ".join(args.families)
        + " | predictive most recoverable? |",
        "|---|---|" + "---|" * (len(args.families) + 1),
    ]
    for win in args.windows:
        cells = []
        for fam in args.families:
            r = res[win][fam]["latent_state"]
            cells.append(f"{r['r2']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]")
        most = v["ordering_holds_per_window"].get(win, False)
        lines.append(
            f"| `{win}` | {is_strictly_preimpact(win)} | "
            + " | ".join(cells)
            + f" | {'YES' if most else 'no'} |"
        )

    lines += [
        "",
        "## context + non-result targets (test_b, R^2 [CI]) at each window",
        "",
        "| window | family | params [G,D,Y] | impact C_L | wake via latent (NON-RESULT) |",
        "|---|---|---|---|---|",
    ]
    for win in args.windows:
        for fam in args.families:
            rp = res[win][fam]["params"]
            rc = res[win][fam]["cl_impact"]
            rw = res[win][fam]["wake_via_latent"]
            lines.append(
                f"| `{win}` | {fam} | {rp['r2']:+.3f} [{rp['ci_lo']:+.3f},{rp['ci_hi']:+.3f}] "
                f"| {rc['r2']:+.3f} [{rc['ci_lo']:+.3f},{rc['ci_hi']:+.3f}] "
                f"| {rw['r2']:+.3f} [{rw['ci_lo']:+.3f},{rw['ci_hi']:+.3f}] |"
            )

    lines += [
        "",
        "## Verdict",
        "",
        f"Predictive family: `{v['predictive_family']}`. Strictly-causal window: "
        f"`{v['causal_window']}`.",
        "",
        "Predictive-latent ordering (predictive family has the strictly highest "
        "latent_state R^2) holds per window: "
        + ", ".join(f"`{w}`={h}" for w, h in v["ordering_holds_per_window"].items())
        + ".",
        "",
        f"Ordering holds under the strictly-causal window: "
        f"**{v['ordering_holds_under_causal_window']}**.",
        "",
        f"**Verdict: {v['branch']}.** {v['recommendation']}",
        "",
        v["note"],
        "",
        f"qDEIM taps (K={args.k}, causal placement): {out['taps']}.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
