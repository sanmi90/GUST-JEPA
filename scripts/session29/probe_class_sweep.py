"""SESSION29 Track D: probe-class robustness (reconciled to v2.1).

Question (referee F4): JEPA is architecturally optimised for LINEAR readability,
so the wake advantage is most likely to move when the probe class changes. Does a
nonlinear probe (kernel-ridge, MLP, gradient boosting) let the reconstructive
observable-augmented AE (fukami, the matched wake-head control) close the gap to
the predictive latent on case-clustered held-out wake-enstrophy R^2?

Regime: readout-frame probe at impact+H (H=16 primary), one row per encounter,
fit on TRAIN cases with hyperparameters selected by nested grouped-CV by case_id
(GroupKFold), evaluated once on test_b with a case-clustered bootstrap CI. All
probe classes share this regime so the JEPA-vs-fukami GAP per probe class is the
apples-to-apples quantity. (The paper's headline 0.79 is a pooled-frame ridge fit;
this script's linear column is the readout-frame fit and may differ slightly, by
design; the comparison across probe classes is what Track D decides.)

STRONG branch: the gap holds under every nonlinear probe -> claim "broad probe
class". WEAK branch: a nonlinear probe closes it -> pin claim to LINEAR decodability.

Outputs: outputs/session29/probe_class_matrix.json (+ provenance), and
outputs/session29/probe_class_sweep.md. CPU-only sklearn.
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import _s29_common as cm

REPO = cm.REPO
OUT_JSON = REPO / "outputs" / "session29" / "probe_class_matrix.json"
OUT_MD = REPO / "outputs" / "session29" / "probe_class_sweep.md"

FAMILIES = ["jepa_tf_noc", "fukami", "pod"]
PROBES = ["linear", "kernel_ridge_rbf", "mlp", "gbm"]


def make_estimator(probe: str, seed: int):
    scaler = ("scale", StandardScaler())
    if probe == "linear":
        est = Ridge()
        grid = {"model__alpha": [0.1, 1.0, 10.0, 100.0]}
    elif probe == "kernel_ridge_rbf":
        est = KernelRidge(kernel="rbf")
        grid = {"model__alpha": [0.01, 0.1, 1.0], "model__gamma": [1e-3, 1e-2, 1e-1]}
    elif probe == "mlp":
        est = MLPRegressor(max_iter=2000, early_stopping=True, random_state=seed)
        grid = {
            "model__hidden_layer_sizes": [(32,), (64,), (64, 32)],
            "model__alpha": [1e-3, 1e-2, 1e-1],
        }
    elif probe == "gbm":
        est = GradientBoostingRegressor(random_state=seed)
        grid = {
            "model__n_estimators": [100, 300],
            "model__max_depth": [2, 3],
            "model__learning_rate": [0.03, 0.1],
        }
    else:
        raise ValueError(probe)
    return Pipeline([scaler, ("model", est)]), grid


def fit_eval(probe, Xtr, ytr, gtr, Xte, yte, gte, seed):
    pipe, grid = make_estimator(probe, seed)
    n_groups = len(np.unique(gtr))
    cv = GroupKFold(n_splits=min(5, n_groups))
    gs = GridSearchCV(pipe, grid, scoring="r2", cv=cv, n_jobs=-1)
    gs.fit(Xtr, ytr, groups=gtr)
    yp = gs.predict(Xte)
    r2, lo, hi = cm.case_clustered_r2_ci(yte, yp, gte, seed=seed)
    # per-case MAE
    cases = np.unique(gte)
    per_case_mae = float(
        np.mean([np.mean(np.abs(yte[gte == c] - yp[gte == c])) for c in cases])
    )
    return dict(
        r2=r2,
        ci_lo=lo,
        ci_hi=hi,
        per_case_mae=per_case_mae,
        best_params={k: v for k, v in gs.best_params_.items()},
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--probes", nargs="+", default=PROBES)
    ap.add_argument("--observable", default="wake_enstrophy")
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    inputs = [cm.LATENTS / cm.FAMILY_TAGS[f] / "train.npz" for f in args.families]
    inputs += [cm.TARGETS / "train.npz", cm.TARGETS / "test_b.npz"]
    if args.dry_run:
        print("[dry-run] families:", args.families, "probes:", args.probes)
        for f in args.families:
            tag = cm.FAMILY_TAGS[f]
            Xtr, ytr, gtr = cm.readout_xy(tag, "train", args.observable, args.horizon)
            Xte, yte, gte = cm.readout_xy(tag, "test_b", args.observable, args.horizon)
            print(
                f"  {f}: train X{Xtr.shape} ({len(np.unique(gtr))} cases), "
                f"test_b X{Xte.shape} ({len(np.unique(gte))} cases)"
            )
        return

    matrix = {}
    for f in args.families:
        tag = cm.FAMILY_TAGS[f]
        Xtr, ytr, gtr = cm.readout_xy(tag, "train", args.observable, args.horizon)
        Xte, yte, gte = cm.readout_xy(tag, "test_b", args.observable, args.horizon)
        if len(yte) == 0 or not np.isfinite(ytr).all():
            raise ValueError(f"{f}: empty/non-finite readout matrix")
        matrix[f] = {}
        for probe in args.probes:
            seeds = args.seeds if probe == "mlp" else [args.seeds[0]]
            runs = [fit_eval(probe, Xtr, ytr, gtr, Xte, yte, gte, s) for s in seeds]
            r2s = [r["r2"] for r in runs]
            rec = dict(runs[0])
            rec["r2"] = float(np.mean(r2s))
            rec["r2_seed_sd"] = float(np.std(r2s)) if len(r2s) > 1 else 0.0
            rec["n_test_b"] = int(len(yte))
            rec["n_cases_test_b"] = int(len(np.unique(gte)))
            matrix[f][probe] = rec
            print(
                f"{f:14s} {probe:18s} R2={rec['r2']:+.3f} "
                f"[{rec['ci_lo']:+.3f},{rec['ci_hi']:+.3f}] MAE/case={rec['per_case_mae']:.3f}"
            )

    # verdict: MATCHED probe-class comparison (honest). For each probe class, does
    # the predictive latent beat BOTH reconstructive baselines? A nonlinear probe
    # that closes or reverses the ordering moves the claim to LINEAR decodability.
    verdict = None
    if all(f in matrix for f in ("jepa_tf_noc", "fukami", "pod")):
        per_probe = {}
        for p in args.probes:
            j = matrix["jepa_tf_noc"][p]["r2"]
            fk = matrix["fukami"][p]["r2"]
            pd = matrix["pod"][p]["r2"]
            per_probe[p] = {
                "jepa": j,
                "fukami": fk,
                "pod": pd,
                "gap_vs_fukami": j - fk,
                "gap_vs_pod": j - pd,
                "jepa_best": j >= max(fk, pd),
            }
        wins = [p for p, d in per_probe.items() if d["jepa_best"]]
        reversed_probes = [p for p, d in per_probe.items() if not d["jepa_best"]]
        # STRONG only if the predictive latent leads under EVERY probe class.
        branch = "STRONG" if not reversed_probes else "WEAK"
        verdict = {
            "per_probe": per_probe,
            "probe_classes_jepa_leads": wins,
            "probe_classes_reversed": reversed_probes,
            "branch": branch,
            "note": (
                "STRONG = predictive latent leads under EVERY probe class (broad "
                "readability). WEAK = a nonlinear probe closes/reverses the ordering "
                "=> pin claim to LINEAR decodability + organization. NB: case-clustered "
                "R^2 CIs are very wide at 10 test_b cases, so the robust signal is the "
                "SIGN consistency across probe classes and the per-case MAE, not the "
                "individual R^2 magnitudes."
            ),
        }

    payload = {
        "_provenance": cm.provenance(inputs, seed=args.seeds[0]),
        "config": {
            "observable": args.observable,
            "horizon": args.horizon,
            "regime": "readout-frame, nested GroupKFold-by-case, case-clustered CI",
        },
        "matrix": matrix,
        "verdict": verdict,
    }
    cm.write_artifact(OUT_JSON, payload)

    lines = [
        "# Track D: probe-class robustness (v2.1, wake enstrophy, H=16, test_b)\n",
        "Readout-frame probe, nested grouped-CV by case, case-clustered bootstrap CI.\n",
        "| family | probe | R^2 | 95% CI | MAE/case |",
        "|---|---|---|---|---|",
    ]
    for f in args.families:
        for probe in args.probes:
            r = matrix[f][probe]
            lines.append(
                f"| {f} | {probe} | {r['r2']:+.3f} | "
                f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {r['per_case_mae']:.3f} |"
            )
    if verdict:
        lines += [
            "",
            "## Matched probe-class verdict",
            "",
            "| probe | jepa | fukami | pod | jepa leads? |",
            "|---|---|---|---|---|",
        ]
        for p, d in verdict["per_probe"].items():
            lines.append(
                f"| {p} | {d['jepa']:+.3f} | {d['fukami']:+.3f} | {d['pod']:+.3f} | "
                f"{'yes' if d['jepa_best'] else 'NO (reversed)'} |"
            )
        lines += [
            "",
            f"**Verdict: {verdict['branch']}.** Predictive latent leads under "
            f"{verdict['probe_classes_jepa_leads']}; reversed under "
            f"{verdict['probe_classes_reversed']}.",
            "",
            verdict["note"],
        ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_MD}")
    if verdict:
        print(
            f"VERDICT: {verdict['branch']} | leads {verdict['probe_classes_jepa_leads']} "
            f"| reversed {verdict['probe_classes_reversed']}"
        )


if __name__ == "__main__":
    main()
