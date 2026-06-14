"""SESSION29 Track G: stronger physical floors vs the predictive latent (v2.1).

Question (referee F5/F6): the central claim is that the predictive JEPA latent
carries wake information beyond what cheap, low-dimensional PHYSICAL descriptors
(gust parameters, force history, a persistence baseline, pre-impact wall
pressure) already encode. If a cheap physical floor predicts held-out wake
enstrophy as well as the latent, the latent's wake advantage shrinks to the
increment over that floor (WEAK). If the latent clears every floor, the central
distinction is strengthened (STRONG).

Regime (matched to Track D so the numbers are comparable): readout frame =
impact+H (H=16 primary), one row per encounter, fit on TRAIN cases, evaluate
once on test_b, group by case_id, Ridge (or KernelRidge-RBF) with the
regularisation selected by GroupKFold(5)-by-case. Score = case-clustered
wake-enstrophy R^2 (cm.case_clustered_r2_ci) plus per-case MAE.

Floors (one feature vector per encounter):
  - gdy           : [G, D, Y] at impact.
  - gdy_history   : [G, D, Y] at impact + C_L, C_D over [impact-10, impact).
  - persistence   : wake_enstrophy at frame=impact (last observed pre-readout
                    frame), i.e. last-observation-carried-forward, single feature.
  - pressure_only : span-mean p_wall over the pre-impact window [impact-20,
                    impact) from the cache, mean-over-window per tap (192 taps).
  - pod_dmd_phase : SKIPPED (POD/DMD phase reconstruction is out of scope here;
                    the pod latent family is already reported as a context row).

Latent reference: jepa_tf_noc's readout wake R^2 computed the SAME way
(cm.readout_xy + the same Ridge/grouped-CV), with fukami and pod for context.

Outputs: outputs/session29/physics_floors.json (+ provenance) and
outputs/session29/physics_floors.md (floor->R^2 table + verdict). CPU-only.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import _s29_common as cm

REPO = cm.REPO
OUT_JSON = REPO / "outputs" / "session29" / "physics_floors.json"
OUT_MD = REPO / "outputs" / "session29" / "physics_floors.md"

FLOORS = ["gdy", "gdy_history", "persistence", "pressure_only"]
LATENT_FAMILIES = ["jepa_tf_noc", "fukami", "pod"]
SKIPPED_FLOORS = {
    "pod_dmd_phase": (
        "POD/DMD phase reconstruction is out of scope for this track; the linear "
        "POD floor is already represented by the 'pod' latent family context row."
    )
}

# Pre-impact / readout window offsets (relative to the per-encounter impact frame).
HISTORY_WINDOW = 10  # gdy_history: C_L/C_D over [impact-10, impact)
PRESSURE_WINDOW = 20  # pressure_only: span-mean p_wall over [impact-20, impact)


def _cache_root() -> Path:
    root = os.environ.get("VORTEX_JEPA_CACHE")
    if not root:
        prevent = os.environ.get("PREVENT_ROOT")
        if not prevent:
            raise EnvironmentError(
                "neither VORTEX_JEPA_CACHE nor PREVENT_ROOT is set; cannot locate "
                "the v2.1 cache for the pressure floor"
            )
        root = str(Path(prevent) / "data" / "processed" / "vortex-jepa")
    return Path(root) / "v2p1"


def _ordered_rows(split: str, horizon: int):
    """Encounter rows in the SAME order/skip-logic as cm.readout_xy.

    Returns a list of dicts with the per-encounter alignment + the target. This is
    the single source of truth that every floor (and the latent reference) is
    aligned against, so X_floor, X_latent and y all index the same encounters.
    """
    obs_full, _ = cm.load_dns_observable(split, "wake_enstrophy")
    # Use the jepa family purely for its (case_id, encounter, impact_frame)
    # alignment + the readout-frame validity test (fr < z.shape[1]); the target y
    # is the DNS observable, not the latent.
    fam = cm.load_family(cm.FAMILY_TAGS["jepa_tf_noc"], split)
    cid, enc, imp = fam["case_ids"], fam["encounter_indices"], fam["impact_frame"]
    n_frames = fam["z_full"].shape[1]
    rows = []
    for i in range(len(cid)):
        impact = int(imp[i])
        fr = impact + horizon
        key = (cid[i], int(enc[i]))
        if key not in obs_full or fr >= n_frames:
            continue
        rows.append(
            {
                "case_id": str(cid[i]),
                "encounter": int(enc[i]),
                "impact": impact,
                "readout_frame": fr,
                "y": float(obs_full[key][fr]),
            }
        )
    if not rows:
        raise ValueError(f"{split}: no valid readout rows at horizon {horizon}")
    return rows


def _per_frame_maps(split: str, observables):
    return {obs: cm.load_dns_observable(split, obs)[0] for obs in observables}


def build_floor_features(floor: str, split: str, horizon: int):
    """Build (X, y, groups) for one floor, aligned to _ordered_rows order.

    Returns (X (n, d_floor), y (n,), groups (n,)). Fails loudly if any feature or
    target is non-finite, or if a required cache file is missing.
    """
    rows = _ordered_rows(split, horizon)
    y = np.asarray([r["y"] for r in rows], dtype=float)
    groups = np.asarray([r["case_id"] for r in rows])

    if floor == "gdy":
        maps = _per_frame_maps(split, ["G", "D", "Y"])
        X = np.asarray(
            [
                [
                    float(maps["G"][(r["case_id"], r["encounter"])][r["impact"]]),
                    float(maps["D"][(r["case_id"], r["encounter"])][r["impact"]]),
                    float(maps["Y"][(r["case_id"], r["encounter"])][r["impact"]]),
                ]
                for r in rows
            ],
            dtype=float,
        )

    elif floor == "gdy_history":
        maps = _per_frame_maps(split, ["G", "D", "Y", "C_L", "C_D"])
        feats = []
        for r in rows:
            cid_enc = (r["case_id"], r["encounter"])
            lo = r["impact"] - HISTORY_WINDOW
            hi = r["impact"]  # exclusive: history strictly before impact
            cl = maps["C_L"][cid_enc][lo:hi]
            cd = maps["C_D"][cid_enc][lo:hi]
            feats.append(
                [
                    float(maps["G"][cid_enc][r["impact"]]),
                    float(maps["D"][cid_enc][r["impact"]]),
                    float(maps["Y"][cid_enc][r["impact"]]),
                    *cl.astype(float).tolist(),
                    *cd.astype(float).tolist(),
                ]
            )
        X = np.asarray(feats, dtype=float)

    elif floor == "persistence":
        wmap = cm.load_dns_observable(split, "wake_enstrophy")[0]
        # Last observed pre-readout frame = impact (LOCF). Single feature.
        X = np.asarray(
            [[float(wmap[(r["case_id"], r["encounter"])][r["impact"]])] for r in rows],
            dtype=float,
        )

    elif floor == "pressure_only":
        X = _pressure_features(rows)

    else:
        raise ValueError(f"unknown floor {floor!r}")

    if not np.isfinite(X).all():
        raise ValueError(f"floor {floor!r} ({split}): non-finite features")
    if not np.isfinite(y).all():
        raise ValueError(f"floor {floor!r} ({split}): non-finite targets")
    return X, y, groups


def _pressure_features(rows):
    """Mean-over-window p_wall per tap (192 taps) from the v2.1 cache.

    p_wall in the cache is already span-meaned to (120, 192) (192 surface taps);
    we average over the pre-impact window [impact-PRESSURE_WINDOW, impact) so the
    feature is a single 192-vector per encounter using only pre-readout info.
    """
    import h5py

    cache = _cache_root()
    feats = []
    for r in rows:
        h5 = cache / r["case_id"] / f"encounter_{r['encounter']:02d}.h5"
        if not h5.exists():
            raise FileNotFoundError(f"pressure floor: missing cache file {h5}")
        lo = r["impact"] - PRESSURE_WINDOW
        hi = r["impact"]  # exclusive: strictly pre-impact
        with h5py.File(h5, "r") as f:
            if "p_wall" not in f:
                raise KeyError(f"pressure floor: no p_wall in {h5}")
            pw = np.asarray(f["p_wall"][lo:hi, :], dtype=float)  # (window, 192)
        feats.append(pw.mean(axis=0))  # mean over the window per tap -> (192,)
    return np.asarray(feats, dtype=float)


def fit_eval(probe, Xtr, ytr, gtr, Xte, yte, gte, seed):
    """Fit on TRAIN (GroupKFold-by-case alpha selection), score on test_b.

    probe in {"ridge", "kernel_ridge_rbf"}. Returns r2 + clustered CI + per-case MAE.
    """
    scaler = ("scale", StandardScaler())
    if probe == "ridge":
        est = Ridge()
        grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
    elif probe == "kernel_ridge_rbf":
        est = KernelRidge(kernel="rbf")
        grid = {
            "model__alpha": [0.01, 0.1, 1.0, 10.0],
            "model__gamma": [1e-3, 1e-2, 1e-1, 1.0],
        }
    else:
        raise ValueError(f"unknown probe {probe!r}")
    pipe = Pipeline([scaler, ("model", est)])
    n_groups = len(np.unique(gtr))
    cv = GroupKFold(n_splits=min(5, n_groups))
    gs = GridSearchCV(pipe, grid, scoring="r2", cv=cv, n_jobs=-1)
    gs.fit(Xtr, ytr, groups=gtr)
    yp = gs.predict(Xte)
    r2, lo, hi = cm.case_clustered_r2_ci(yte, yp, gte, seed=seed)
    if not np.isfinite(r2):
        raise ValueError(f"{probe}: non-finite R^2 on test_b")
    cases = np.unique(gte)
    per_case_mae = float(np.mean([np.mean(np.abs(yte[gte == c] - yp[gte == c])) for c in cases]))
    return dict(
        r2=float(r2),
        ci_lo=float(lo),
        ci_hi=float(hi),
        per_case_mae=per_case_mae,
        best_params={k: v for k, v in gs.best_params_.items()},
        n_features=int(Xtr.shape[1]),
        n_test_b=int(len(yte)),
        n_cases_test_b=int(len(np.unique(gte))),
    )


def _best_over_probes(Xtr, ytr, gtr, Xte, yte, gte, probes, seed):
    """Run each probe class and keep the one with the highest test_b R^2."""
    runs = {p: fit_eval(p, Xtr, ytr, gtr, Xte, yte, gte, seed) for p in probes}
    best = max(runs, key=lambda p: runs[p]["r2"])
    rec = dict(runs[best])
    rec["probe"] = best
    rec["by_probe"] = {p: runs[p]["r2"] for p in probes}
    return rec


def decide_verdict(latent_r2: float, floor_r2: dict) -> dict:
    """Pure verdict logic: predictive latent vs every physical floor.

    STRONG iff the predictive-latent R^2 strictly exceeds EVERY floor R^2; WEAK
    if any floor matches or beats it (latent_r2 <= floor_r2). The latent advantage
    is reported against the single best floor. Pulled out as a pure function so it
    is unit-testable without the heavy real latents.
    """
    if not floor_r2:
        raise ValueError("decide_verdict: no floors provided")
    best_floor = max(floor_r2, key=lambda k: floor_r2[k])
    best_floor_r2 = floor_r2[best_floor]
    cleared = [fl for fl, v in floor_r2.items() if latent_r2 > v]
    not_cleared = [fl for fl, v in floor_r2.items() if latent_r2 <= v]
    return {
        "latent_r2": float(latent_r2),
        "floor_r2": {k: float(v) for k, v in floor_r2.items()},
        "best_floor": best_floor,
        "best_floor_r2": float(best_floor_r2),
        "latent_advantage_over_best_floor": float(latent_r2 - best_floor_r2),
        "floors_cleared": cleared,
        "floors_not_cleared": not_cleared,
        "branch": "STRONG" if not not_cleared else "WEAK",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--floors", nargs="+", default=FLOORS)
    ap.add_argument("--families", nargs="+", default=LATENT_FAMILIES)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument(
        "--probes",
        nargs="+",
        default=["ridge", "kernel_ridge_rbf"],
        help="probe classes; the best test_b R^2 per floor/family is reported.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    inputs = [cm.LATENTS / cm.FAMILY_TAGS[f] / "train.npz" for f in args.families]
    inputs += [cm.TARGETS / "train.npz", cm.TARGETS / "test_b.npz"]

    if args.dry_run:
        print("[dry-run] floors:", args.floors, "families:", args.families)
        print("[dry-run] skipped floors:", list(SKIPPED_FLOORS))
        for sp in ("train", "test_b"):
            rows = _ordered_rows(sp, args.horizon)
            print(f"  {sp}: {len(rows)} rows, {len(set(r['case_id'] for r in rows))} cases")
        for floor in args.floors:
            Xtr, ytr, gtr = build_floor_features(floor, "train", args.horizon)
            Xte, _, gte = build_floor_features(floor, "test_b", args.horizon)
            print(
                f"  floor {floor:14s}: train X{Xtr.shape}, test_b X{Xte.shape} "
                f"({len(np.unique(gte))} cases)"
            )
        for f in args.families:
            tag = cm.FAMILY_TAGS[f]
            Xtr, _, _ = cm.readout_xy(tag, "train", "wake_enstrophy", args.horizon)
            print(f"  latent {f:14s}: train X{Xtr.shape}")
        return

    results = {"floors": {}, "latents": {}}

    # --- physical floors ---
    for floor in args.floors:
        Xtr, ytr, gtr = build_floor_features(floor, "train", args.horizon)
        Xte, yte, gte = build_floor_features(floor, "test_b", args.horizon)
        # persistence is a single feature: a kernel adds nothing, keep ridge only.
        probes = ["ridge"] if floor == "persistence" else args.probes
        rec = _best_over_probes(Xtr, ytr, gtr, Xte, yte, gte, probes, args.seed)
        results["floors"][floor] = rec
        print(
            f"FLOOR  {floor:14s} R2={rec['r2']:+.3f} "
            f"[{rec['ci_lo']:+.3f},{rec['ci_hi']:+.3f}] "
            f"MAE/case={rec['per_case_mae']:.3f} ({rec['probe']}, {rec['n_features']}f)"
        )

    # --- latent reference (same Ridge/grouped-CV regime) ---
    for f in args.families:
        tag = cm.FAMILY_TAGS[f]
        Xtr, ytr, gtr = cm.readout_xy(tag, "train", "wake_enstrophy", args.horizon)
        Xte, yte, gte = cm.readout_xy(tag, "test_b", "wake_enstrophy", args.horizon)
        if len(yte) == 0 or not np.isfinite(ytr).all() or not np.isfinite(yte).all():
            raise ValueError(f"latent {f}: empty/non-finite readout matrix")
        rec = _best_over_probes(Xtr, ytr, gtr, Xte, yte, gte, args.probes, args.seed)
        results["latents"][f] = rec
        print(
            f"LATENT {f:14s} R2={rec['r2']:+.3f} "
            f"[{rec['ci_lo']:+.3f},{rec['ci_hi']:+.3f}] "
            f"MAE/case={rec['per_case_mae']:.3f} ({rec['probe']}, {rec['n_features']}f)"
        )

    # --- verdict: predictive latent (jepa_tf_noc) vs every floor ---
    verdict = None
    if "jepa_tf_noc" in results["latents"] and results["floors"]:
        latent_r2 = results["latents"]["jepa_tf_noc"]["r2"]
        floor_r2 = {fl: results["floors"][fl]["r2"] for fl in results["floors"]}
        verdict = decide_verdict(latent_r2, floor_r2)
        verdict["predictive_latent"] = "jepa_tf_noc"
        verdict["note"] = (
            "STRONG = the predictive latent (jepa_tf_noc) wake-enstrophy R^2 "
            "exceeds EVERY physical floor; the latent advantage is the latent "
            "minus the best floor. WEAK = some cheap floor (most likely "
            "gdy_history or persistence) matches or beats the latent, so the "
            "latent's wake claim shrinks to the increment over that floor. NB: "
            "case-clustered R^2 CIs are wide at 10 test_b cases; weigh the sign "
            "consistency and per-case MAE alongside the point R^2."
        )

    payload = {
        "_provenance": cm.provenance(inputs, seed=args.seed),
        "config": {
            "observable": "wake_enstrophy",
            "horizon": args.horizon,
            "regime": "readout-frame impact+H, fit train, eval test_b, "
            "GroupKFold(5)-by-case, case-clustered CI",
            "probes": args.probes,
            "history_window": HISTORY_WINDOW,
            "pressure_window": PRESSURE_WINDOW,
            "skipped_floors": SKIPPED_FLOORS,
        },
        "results": results,
        "verdict": verdict,
    }
    cm.write_artifact(OUT_JSON, payload)

    _write_md(args, results, verdict)
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_MD}")
    if verdict:
        print(
            f"VERDICT: {verdict['branch']} | latent R2={verdict['latent_r2']:+.3f} "
            f"| best floor {verdict['best_floor']}={verdict['best_floor_r2']:+.3f} "
            f"| advantage {verdict['latent_advantage_over_best_floor']:+.3f}"
        )


def _write_md(args, results, verdict) -> None:
    lines = [
        "# Track G: stronger physical floors vs the predictive latent "
        "(v2.1, wake enstrophy, H={}, test_b)\n".format(args.horizon),
        "Readout-frame fit at impact+H, fit on train, evaluated on test_b, "
        "GroupKFold(5)-by-case alpha selection, case-clustered bootstrap CI. "
        "Per row the best of {} is reported.\n".format(args.probes),
        "| kind | name | R^2 | 95% CI | MAE/case | probe | n_feat |",
        "|---|---|---|---|---|---|---|",
    ]
    for fl, r in results["floors"].items():
        lines.append(
            f"| floor | {fl} | {r['r2']:+.3f} | "
            f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {r['per_case_mae']:.3f} | "
            f"{r['probe']} | {r['n_features']} |"
        )
    for f, r in results["latents"].items():
        tag = "predictive-latent" if f == "jepa_tf_noc" else "latent (context)"
        lines.append(
            f"| {tag} | {f} | {r['r2']:+.3f} | "
            f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {r['per_case_mae']:.3f} | "
            f"{r['probe']} | {r['n_features']} |"
        )
    if SKIPPED_FLOORS:
        lines += ["", "## Skipped floors", ""]
        for name, why in SKIPPED_FLOORS.items():
            lines.append(f"- `{name}`: {why}")
    if verdict:
        lines += [
            "",
            "## Verdict",
            "",
            f"Predictive latent (`{verdict['predictive_latent']}`) R^2 = "
            f"{verdict['latent_r2']:+.3f}. Best floor = `{verdict['best_floor']}` "
            f"at R^2 = {verdict['best_floor_r2']:+.3f}. Latent advantage over the "
            f"best floor = {verdict['latent_advantage_over_best_floor']:+.3f}.",
            "",
            f"Floors the latent clears: {verdict['floors_cleared']}.",
            f"Floors the latent does NOT clear: {verdict['floors_not_cleared']}.",
            "",
            f"**Verdict: {verdict['branch']}.**",
            "",
            verdict["note"],
        ]
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
