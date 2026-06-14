"""SESSION29 Track C-full: per-fold held-out wake closure + 5-fold aggregate.

After cv_full_train.sh has retrained jepa and the matched reconstructive control
(ctrl_recon, fuk_matched recipe) per case-disjoint fold and encoded each fold's
held-out cases, this script fits the canonical wake-enstrophy probe (readout frame
impact+H) on each fold's TRAIN latents and evaluates on that fold's HELD-OUT cases,
then aggregates the five held-out-case wake R^2 values per family. The held-out
distribution across folds is the F3 case-level evidence: every case is predicted by
an encoder that never saw its case.

Canonical wake target: pooled from dns_physical_metrics (train + test_a + test_b),
so any fold's cases resolve. CPU-only.

Outputs outputs/session29/cv_full.json + .md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import _s29_common as cm

REPO = cm.REPO
LAT = REPO / "outputs" / "session28" / "latents" / "cvfull"
OUT_JSON = REPO / "outputs" / "session29" / "cv_full.json"
OUT_MD = REPO / "outputs" / "session29" / "cv_full.md"
# fold-encoded model dirs -> (latent subdir name, label)
MODELS = {
    "jepa": "jepa",
    "ctrl_recon": "fukami",
}  # path label -> encode --baseline subdir


def pooled_canonical(observable: str) -> dict:
    """(case_id, enc) -> per-frame observable, pooled over train+test_a+test_b."""
    d = np.load(cm.DNS_CANON, allow_pickle=True)
    out = {}
    for split in ("train", "test_a", "test_b"):
        cid = np.asarray(d[f"{split}_case_id"]).astype(str)
        enc = np.asarray(d[f"{split}_encounter_index"]).astype(int)
        obs = np.asarray(d[f"{split}_{observable}"])
        for i in range(len(cid)):
            out[(cid[i], int(enc[i]))] = obs[i]
    return out


def fold_xy(npz_path: Path, obs_map: dict, horizon: int):
    d = np.load(npz_path, allow_pickle=True)
    z = d["z_full"]
    cid = np.asarray(d["case_ids"]).astype(str)
    enc = np.asarray(d["encounter_indices"]).astype(int)
    imp = np.asarray(d["impact_frame"]).astype(int)
    X, y, g = [], [], []
    for i in range(len(cid)):
        fr = int(imp[i]) + horizon
        k = (cid[i], int(enc[i]))
        if k in obs_map and fr < z.shape[1]:
            X.append(z[i, fr])
            y.append(float(obs_map[k][fr]))
            g.append(cid[i])
    return np.asarray(X), np.asarray(y), np.asarray(g)


def fit_probe(Xtr, ytr, gtr, Xte):
    pipe = Pipeline([("s", StandardScaler()), ("m", Ridge())])
    cv = GroupKFold(n_splits=min(5, len(np.unique(gtr))))
    gs = GridSearchCV(
        pipe, {"m__alpha": [0.1, 1.0, 10.0, 100.0]}, scoring="r2", cv=cv, n_jobs=-1
    )
    gs.fit(Xtr, ytr, groups=gtr)
    return gs.predict(Xte)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--observable", default="wake_enstrophy")
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="use *_smoke encode dirs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sm = "_smoke" if args.smoke else ""
    obs_map = pooled_canonical(args.observable)
    if args.dry_run:
        print(f"[dry-run] pooled canonical map: {len(obs_map)} (case,enc) keys")
        for k in range(args.folds):
            for lbl, sub in MODELS.items():
                p = LAT / f"fold{k}" / f"{sub}{sm}" / "test_b.npz"
                print(
                    f"  fold{k} {lbl}: {'present' if p.exists() else 'MISSING'} ({p})"
                )
        return

    per_fold = {lbl: [] for lbl in MODELS}
    for k in range(args.folds):
        for lbl, sub in MODELS.items():
            tr = LAT / f"fold{k}" / f"{sub}{sm}" / "train.npz"
            te = LAT / f"fold{k}" / f"{sub}{sm}" / "test_b.npz"
            if not (tr.exists() and te.exists()):
                print(f"fold{k} {lbl}: MISSING latents, skipping")
                continue
            Xtr, ytr, gtr = fold_xy(tr, obs_map, args.horizon)
            Xte, yte, gte = fold_xy(te, obs_map, args.horizon)
            if len(yte) == 0:
                print(f"fold{k} {lbl}: empty holdout")
                continue
            yp = fit_probe(Xtr, ytr, gtr, Xte)
            r2, lo, hi = cm.case_clustered_r2_ci(yte, yp, gte, seed=k)
            per_fold[lbl].append(
                {"fold": k, "r2": r2, "n_holdout_cases": int(len(np.unique(gte)))}
            )
            print(
                f"fold{k} {lbl:11s} held-out wake R2={r2:+.3f} ({len(np.unique(gte))} cases)"
            )

    summary = {}
    for lbl, rows in per_fold.items():
        vals = [r["r2"] for r in rows]
        if vals:
            summary[lbl] = {
                "median": float(np.median(vals)),
                "mean": float(np.mean(vals)),
                "iqr": [float(np.percentile(vals, 25)), float(np.percentile(vals, 75))],
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n_folds": len(vals),
                "per_fold": rows,
            }
    verdict = None
    if "jepa" in summary and "ctrl_recon" in summary:
        gap = summary["jepa"]["median"] - summary["ctrl_recon"]["median"]
        verdict = {
            "jepa_median": summary["jepa"]["median"],
            "ctrl_recon_median": summary["ctrl_recon"]["median"],
            "median_gap": gap,
            "branch": "STRONG" if gap > 0.2 else "WEAK",
            "note": (
                "5-fold held-out-case wake R^2 distribution; STRONG if the predictive "
                "median clears the matched reconstructive control by >0.2 across folds. "
                "Every case predicted by an encoder that never saw its case (F3)."
            ),
        }
    payload = {
        "_provenance": cm.provenance([cm.DNS_CANON], seed=0),
        "config": {
            "observable": args.observable,
            "horizon": args.horizon,
            "smoke": args.smoke,
        },
        "summary": summary,
        "verdict": verdict,
    }
    cm.write_artifact(OUT_JSON, payload)
    lines = [
        "# Track C-full: 5-fold grouped-CV held-out wake closure (canonical, H16)\n",
        "| family | median | mean | IQR | min | max | folds |",
        "|---|---|---|---|---|---|---|",
    ]
    for lbl, s in summary.items():
        iqr = f"[{s['iqr'][0]:+.3f},{s['iqr'][1]:+.3f}]"
        lines.append(
            f"| {lbl} | {s['median']:+.3f} | {s['mean']:+.3f} | "
            f"{iqr} | {s['min']:+.3f} | {s['max']:+.3f} | {s['n_folds']} |"
        )
    if verdict:
        lines += [
            "",
            f"**Verdict: {verdict['branch']}** (jepa median "
            f"{verdict['jepa_median']:+.3f} vs matched-recon "
            f"{verdict['ctrl_recon_median']:+.3f}, gap {verdict['median_gap']:+.3f}).",
            verdict["note"],
        ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
