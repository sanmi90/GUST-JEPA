"""SESSION29 Track E auxiliary leakage tests (F4), reconciled to v2.1.

Two frozen-encoder sentinels for the wake-readability probe, on the CANONICAL
wake target (dns_physical_metrics), readout frame impact+H, grouped by case:

1. shuffled-label sentinel: refit the probe on TRAIN with the wake labels permuted
   across encounters, evaluate on the REAL test_b labels. A genuine latent->wake
   structure gives ~0 (or negative) here; a high value would mean the probe is
   finding spurious structure. Confirms the real R^2 is not overfitting.
2. real vs shuffled gap per family: real_R2 - shuffled_R2 should be large and
   positive for the families that genuinely encode the wake (jepa, supervised_only).

This does NOT retrain encoders (a training-time shuffled-label control would need a
GPU rerun; the probe-level sentinel is the cheap, frozen version and is what
guards the Track E / Track D readability claim against probe overfitting).

CPU-only. Outputs outputs/session29/auxiliary_leakage.json + .md.
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import _s29_common as cm

REPO = cm.REPO
OUT_JSON = REPO / "outputs" / "session29" / "auxiliary_leakage.json"
OUT_MD = REPO / "outputs" / "session29" / "auxiliary_leakage.md"
FAMILIES = ["jepa_tf_noc", "supervised_only", "fukami", "pod"]


def fit_probe(Xtr, ytr, gtr, Xte):
    pipe = Pipeline([("scale", StandardScaler()), ("model", Ridge())])
    grid = {"model__alpha": [0.1, 1.0, 10.0, 100.0]}
    cv = GroupKFold(n_splits=min(5, len(np.unique(gtr))))
    gs = GridSearchCV(pipe, grid, scoring="r2", cv=cv, n_jobs=-1)
    gs.fit(Xtr, ytr, groups=gtr)
    return gs.predict(Xte)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--observable", default="wake_enstrophy")
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    inputs = [cm.DNS_CANON]
    if args.dry_run:
        print("[dry-run] families:", args.families)
        return

    rows = {}
    for f in args.families:
        tag = cm.FAMILY_TAGS[f]
        Xtr, ytr, gtr = cm.readout_xy(
            tag, "train", args.observable, args.horizon, "canonical"
        )
        Xte, yte, gte = cm.readout_xy(
            tag, "test_b", args.observable, args.horizon, "canonical"
        )
        if len(yte) == 0 or not np.isfinite(ytr).all():
            raise ValueError(f"{f}: empty/non-finite readout matrix")
        # real
        yp = fit_probe(Xtr, ytr, gtr, Xte)
        real_r2, real_lo, real_hi = cm.case_clustered_r2_ci(
            yte, yp, gte, seed=args.seed
        )
        # shuffled-label: permute TRAIN labels across encounters, eval on real test
        yp_sh = fit_probe(Xtr, rng.permutation(ytr), gtr, Xte)
        sh_r2, sh_lo, sh_hi = cm.case_clustered_r2_ci(yte, yp_sh, gte, seed=args.seed)
        rows[f] = {
            "real_r2": real_r2,
            "real_ci": [real_lo, real_hi],
            "shuffled_r2": sh_r2,
            "shuffled_ci": [sh_lo, sh_hi],
            "real_minus_shuffled": real_r2 - sh_r2,
        }
        print(
            f"{f:16s} real={real_r2:+.3f} shuffled={sh_r2:+.3f} "
            f"gap={real_r2 - sh_r2:+.3f}"
        )

    # sentinel passes if shuffled collapses (<= 0.1) for the genuine-encoding
    # families and the real-minus-shuffled gap is large for jepa/supervised_only.
    genuine = [f for f in ("jepa_tf_noc", "supervised_only") if f in rows]
    sentinel_ok = all(rows[f]["shuffled_r2"] <= 0.1 for f in rows) and all(
        rows[f]["real_minus_shuffled"] > 0.3 for f in genuine
    )
    payload = {
        "_provenance": cm.provenance(inputs, seed=args.seed),
        "config": {
            "observable": args.observable,
            "horizon": args.horizon,
            "target_source": "canonical",
            "test": "shuffled_label_probe_sentinel",
        },
        "rows": rows,
        "sentinel_ok": bool(sentinel_ok),
        "note": (
            "shuffled-label probe sentinel: permute TRAIN wake labels, refit, "
            "eval on REAL test_b. shuffled R^2 ~0 confirms the real R^2 is "
            "genuine latent->wake structure, not probe overfitting. A training-"
            "time shuffled-label encoder control is the GPU complement (deferred)."
        ),
    }
    cm.write_artifact(OUT_JSON, payload)
    lines = [
        "# Track E leakage: shuffled-label probe sentinel (canonical wake, H16, test_b)\n",
        "| family | real R^2 | shuffled R^2 | gap |",
        "|---|---|---|---|",
    ]
    for f in args.families:
        r = rows[f]
        lines.append(
            f"| {f} | {r['real_r2']:+.3f} | {r['shuffled_r2']:+.3f} | "
            f"{r['real_minus_shuffled']:+.3f} |"
        )
    lines += [
        "",
        f"**Sentinel {'PASS' if sentinel_ok else 'FAIL'}.** " + payload["note"],
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(
        f"\nSENTINEL {'PASS' if sentinel_ok else 'FAIL'}\nwrote {OUT_JSON}\nwrote {OUT_MD}"
    )


if __name__ == "__main__":
    main()
