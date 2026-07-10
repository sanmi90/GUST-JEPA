"""Track M2b: parameter-only floor re-run on split v2.2.

The Methods diagnostic (section_3_methods.tex, "parameter-only floor")
claims a kernel-ridge map from c = (G, D, Y) alone to each observable is
LOW on wake enstrophy at forecast horizon H = 16 (so the latent's wake
closure is not parameter-explainable) and HIGH at the impact frame (the
just-released gust nearly determines the instantaneous state). The
supporting computation was session-23-era (exp_conditioning_floor_plus.py,
pre-v2.2 split, session17 metrics); no v2.2 artifact existed
(editorial/PROVENANCE.md M2b). This script recomputes both floors on the
v2.2 caches so the sentences rest on the current generation.

Protocol (mirrors the honest-CV variant of the session-23 script and the
M1/closure sampling so floor and closure are same-sample comparable):
- features: (G, D, Y) parsed from the archive case_id (archive sign
  convention used consistently for train and test; RBF distances are
  convention-invariant when applied uniformly);
- regressor: StandardScaler + KernelRidge(RBF), alpha/gamma CV-selected on
  train by GroupKFold over cases (5 folds);
- floor A "impact frame": one sample per encounter, observable at frame 40
  (the partition's fixed impact_frame estimate);
- floor B "H = 16 closure window": the M1 sample set, target frames
  anchor + 16 for sliding anchors (anchor >= 24) with the target row inside
  window_mask; features constant per encounter, so the floor measures what
  parameters alone explain of the pooled in-window variance;
- scored on test_b (pooled R^2, sklearn convention), observables
  C_L, C_D, wake_enstrophy, circulation_pos, circulation_neg from the
  cache targets (model-independent; read from the POD cache).

Run (CPU): python -m scripts.session36.param_floor_v2p2
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.session36.rex_families_m1 import OBSERVABLES, load_split, group  # noqa: E402

CTX = 25
H = 16
IMPACT_FRAME = 40
CASE_RE = re.compile(r"G([+-]\d+\.\d+)_D(\d+\.\d+)_Y([+-]\d+\.\d+)")


def cvec(case_id: str) -> tuple[float, float, float]:
    if case_id == "Baseline":
        # no-gust reference: G = 0 makes (D, Y) physically irrelevant; the
        # placeholders sit inside the training envelope so RBF distances
        # stay well-scaled (noted in the output provenance).
        return (0.0, 1.0, 0.0)
    m = CASE_RE.match(case_id)
    if not m:
        raise ValueError(f"unparseable case_id {case_id}")
    return tuple(float(g) for g in m.groups())


def samples(split: dict, mode: str):
    """Return (C, y_dict, case_labels) for floor mode 'impact' or 'h16'."""
    encs = group(split)
    feats, rows, cases = [], [], []
    for e in encs:
        r = e["rows"]
        frames = split["frame"][r]
        wm = split["window_mask"][r]
        if mode == "impact":
            sel = np.where(frames == IMPACT_FRAME)[0]
        else:
            anchors = np.arange(CTX - 1, len(r) - 1)
            tgt = anchors + H
            ok = (tgt <= len(r) - 1) & wm[np.clip(tgt, 0, len(r) - 1)]
            sel = tgt[ok]
        if sel.size == 0:
            continue
        feats.append(np.tile(cvec(e["case_id"]), (sel.size, 1)))
        rows.append(r[sel])
        cases.append(np.array([e["case_id"]] * sel.size))
    C = np.concatenate(feats)
    rows = np.concatenate(rows)
    y = {o: split[f"target_{o}"][rows] for o in OBSERVABLES}
    return C, y, np.concatenate(cases)


def main() -> int:
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tr = load_split("pod", "train")
    tb = load_split("pod", "test_b")
    git = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()
    out = {
        "_provenance": {
            "script": "scripts/session36/param_floor_v2p2.py",
            "git_commit": git,
            "split": "v2p2 (train fit, test_b score)",
            "targets_from": "outputs/session34/trackc_latents/latents_pod_*.npz "
                            "(model-independent cache targets)",
            "baseline_case": "encoded (G, D, Y) = (0.0, 1.0, 0.0); at G = 0 the "
                             "vortex is absent so (D, Y) are placeholders",
            "regressor": "StandardScaler + KernelRidge(RBF), GroupKFold(5) by case, "
                         "alpha in 10^[-2..2], gamma in 10^[-2..1]",
            "modes": {"impact": f"frame {IMPACT_FRAME}, one sample/encounter",
                      "h16": f"target frames anchor+{H}, anchors >= {CTX - 1}, "
                             "targets in window_mask (M1 sampling)"},
        },
        "floors": {},
    }
    grid = {
        "kernelridge__alpha": np.logspace(-2, 2, 9),
        "kernelridge__gamma": np.logspace(-2, 1, 7),
    }
    for mode in ("impact", "h16"):
        Ctr, ytr, cases_tr = samples(tr, mode)
        Ctb, ytb, _ = samples(tb, mode)
        rec = {}
        for obs in OBSERVABLES:
            gcv = GridSearchCV(
                make_pipeline(StandardScaler(), KernelRidge(kernel="rbf")),
                grid,
                cv=GroupKFold(n_splits=5).split(Ctr, ytr[obs], groups=cases_tr),
                scoring="r2",
                n_jobs=4,
            )
            gcv.fit(Ctr, ytr[obs])
            rec[obs] = {
                "r2_test_b": float(r2_score(ytb[obs], gcv.predict(Ctb))),
                "cv_r2_train": float(gcv.best_score_),
                "alpha": float(gcv.best_params_["kernelridge__alpha"]),
                "gamma": float(gcv.best_params_["kernelridge__gamma"]),
                "n_train": int(Ctr.shape[0]),
                "n_test_b": int(Ctb.shape[0]),
            }
            print(f"[floor {mode}] {obs}: test_b R2 = {rec[obs]['r2_test_b']:+.3f} "
                  f"(cv {rec[obs]['cv_r2_train']:+.3f})", flush=True)
        out["floors"][mode] = rec
    path = REPO_ROOT / "outputs/session36/param_floor_v2p2.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"[floor] -> {path.relative_to(REPO_ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
