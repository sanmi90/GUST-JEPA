"""Session 33 distributed-code analysis on the pooled v2.2 latents (Section 4.6).

Re-run of parts (i) and (ii) of scripts/session28/p5_wake_code.py for the v2.2
manuscript, per SESSION_33_MANUSCRIPT_V3.md Section 11 item 5 ("The
distributed-code gap and energy-information curve, Section 4.6"). Part (iii)
of the session28 script (the spatial saliency footprint) is intentionally NOT
ported: it needs the frozen encoder on GPU and the manuscript v3 plan only
asks for the gap and the curve.

PROBE REGIME DECLARATION (CLAUDE.md probe methodology): every probe here is a
STATE-DESCRIPTOR probe on PER-frame z_gap, toward the SAME-frame DNS wake
enstrophy mirrored into the latent caches (key target_wake_enstrophy). No
parameter (G, D, Y) probe lives here. This differs from the session28 p5
protocol (future wake at H = 16, |Spearman| skill): the Section 11 item 5 spec
asks for the linear-probe R^2 form on the pooled caches, so the v2.1 reference
gap (0.36 predictive vs 0.05/0.04 baselines, Spearman-based) is stored in the
JSON as a QUALITATIVE reference, not a numerically comparable one.

PART (i), full-vs-best-coordinate gap: per family, a linear probe of the wake
enstrophy from the FULL d = 32 latent vs from the single best coordinate (best
chosen on train). Probe = StandardScaler -> RidgeCV, target-scaled
(src.evaluation.represent.fit_linear_probe, the session31 Q1 convention).
Train skill is out-of-fold (OOF) under a 5-fold GroupKFold grouped by the
encounter key case_id/encounter_index; test_b skill is a probe fit on ALL
train rows and scored on test_b. R^2 is scored on the window_mask rows (the
per-encounter analysis window, the session31 Q1 scoring convention); the
all-rows R^2 is kept as a secondary number. gap = full - best_single.

PART (ii), energy-information curve: per family, probe skill vs the number of
leading PCA components k in {1, 2, 4, 8, 16, 32}. For the test_b curve the
PCA is fit on ALL train z_gap rows (the session28 convention); for the train
OOF curve the PCA is refit inside each fold on the fold-train rows so the OOF
number is leakage-free.

Rows in each latents NPZ are per (encounter, frame); the caches carry
z_gap (N, 32), case_id, encounter_index, frame, window_mask and the mirrored
DNS targets. CPU-only (numpy/sklearn); no GPU is touched.

Output: outputs/session33/wake_code_v2p2.json with a params/provenance block
(inputs, probe protocol, v2.1 reference constants) and per-family results
(full/best-coordinate R^2 and gap on train-OOF and test_b, per-coordinate
skills, and the skill-vs-k curve).

Usage:
    export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
    taskset -c 16-23 nice -n 10 .venv/bin/python \\
        scripts/session33/wake_code_v2p2.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.evaluation.represent import fit_linear_probe  # noqa: E402

TARGET = "target_wake_enstrophy"
KS = [1, 2, 4, 8, 16, 32]
N_FOLDS = 5

# Model tag -> per-frame pooled-latent caches (repo-relative).
MODEL_CACHES = {
    # D250 flagship: predictive family is the native-vector jepa_pool_vec.
    "jepa_pool": "outputs/session33/q1_vec_latents/latents_jepa_pool_vec_{split}.npz",
    "supervised_only_pool": (
        "outputs/session32/q1_pool_latents/latents_supervised_only_pool_{split}.npz"
    ),
    "fukami_wake": "outputs/session31/q1_latents/latents_fukami_wake_{split}.npz",
    "pod": "outputs/session31/q1_latents/latents_pod_{split}.npz",
    "regAE_pool": "outputs/session32/q1_pool_latents/latents_regAE_pool_{split}.npz",
}

# v2.1 reference (outputs/session28/p5_code/p5_results.json, part (i)): the
# full-vs-best gap was LARGE for the predictive family and NEAR ZERO for the
# baselines. Those numbers are |Spearman| combo-minus-best-single at H = 16 on
# the v2.1 d = 64 latents, so they are a qualitative ordering reference only.
V2P1_REFERENCE = {
    "gap_predictive_jepa_tf_noc": 0.36,
    "gap_fukami": 0.05,
    "gap_pod": 0.04,
    "definition": (
        "v2.1 session28 p5 part (i): |Spearman| combination-minus-best-single "
        "toward FUTURE wake enstrophy at H = 16, d = 64 latents. The v2.2 "
        "numbers in this file are linear-probe R^2 toward the SAME-frame wake "
        "enstrophy at pooled d = 32, so only the qualitative ordering (large "
        "gap for the predictive family, small for the baselines) transfers."
    ),
}


def load_split(family: str, split: str) -> dict[str, np.ndarray]:
    """Load one family/split per-frame cache: z_gap, target, groups, mask."""
    path = REPO / MODEL_CACHES[family].format(split=split)
    blob = np.load(path, allow_pickle=True)
    cids = np.array([str(c) for c in blob["case_id"]])
    encs = blob["encounter_index"].astype(np.int64)
    groups = np.array([f"{c}/{e}" for c, e in zip(cids, encs)])
    return {
        "z": blob["z_gap"].astype(np.float64),
        "y": blob[TARGET].astype(np.float64),
        "groups": groups,
        "mask": blob["window_mask"].astype(bool),
        "path": str(path.relative_to(REPO)),
    }


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Protocol R^2 (SST about the evaluation-set mean)."""
    sse = float(((y_pred - y_true) ** 2).sum())
    sst = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - sse / max(sst, 1e-12)


def oof_r2(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    mask: np.ndarray,
    pca_k: int | None = None,
) -> dict[str, float]:
    """5-fold GroupKFold OOF R^2 on train, grouped by the encounter key.

    Probes are fit on ALL fold-train rows; the pooled OOF prediction is scored
    on the window_mask rows (primary) and on all rows (secondary). With
    pca_k set, a PCA is refit on the fold-train rows and both sides are
    projected onto its leading pca_k components (leakage-free OOF curve).
    """
    yhat = np.full(y.shape, np.nan)
    gkf = GroupKFold(n_splits=N_FOLDS)
    for tr_idx, va_idx in gkf.split(X, y, groups):
        Xtr, Xva = X[tr_idx], X[va_idx]
        if pca_k is not None:
            pca = PCA(n_components=pca_k).fit(Xtr)
            Xtr, Xva = pca.transform(Xtr), pca.transform(Xva)
        model = fit_linear_probe(Xtr, y[tr_idx])
        yhat[va_idx] = model.predict(Xva)
    assert np.isfinite(yhat).all(), "OOF prediction has uncovered rows"
    return {
        "r2_window": _r2(y[mask], yhat[mask]),
        "r2_allrows": _r2(y, yhat),
    }


def train_to_test_r2(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    mask_te: np.ndarray,
) -> dict[str, float]:
    """Probe fit on ALL train rows, scored on test_b (windowed + all rows)."""
    model = fit_linear_probe(Xtr, ytr)
    yhat = model.predict(Xte)
    return {
        "r2_window": _r2(yte[mask_te], yhat[mask_te]),
        "r2_allrows": _r2(yte, yhat),
    }


def analyse_family(family: str) -> dict:
    """Parts (i) and (ii) for one family."""
    tr = load_split(family, "train")
    tb = load_split(family, "test_b")
    d = tr["z"].shape[1]

    # ---- part (i): full vs best single coordinate --------------------------
    full_oof = oof_r2(tr["z"], tr["y"], tr["groups"], tr["mask"])
    full_tb = train_to_test_r2(tr["z"], tr["y"], tb["z"], tb["y"], tb["mask"])

    per_coord_oof = []
    for k in range(d):
        r = oof_r2(tr["z"][:, [k]], tr["y"], tr["groups"], tr["mask"])
        per_coord_oof.append(r["r2_window"])
    per_coord_oof = np.asarray(per_coord_oof)
    best_k = int(np.argmax(per_coord_oof))
    best_oof = float(per_coord_oof[best_k])
    best_tb = train_to_test_r2(
        tr["z"][:, [best_k]], tr["y"], tb["z"][:, [best_k]], tb["y"], tb["mask"]
    )

    part_i = {
        "full_r2_train_oof": full_oof["r2_window"],
        "full_r2_train_oof_allrows": full_oof["r2_allrows"],
        "full_r2_test_b": full_tb["r2_window"],
        "full_r2_test_b_allrows": full_tb["r2_allrows"],
        "best_coord": best_k,
        "best_coord_r2_train_oof": best_oof,
        "best_coord_r2_test_b": best_tb["r2_window"],
        "best_coord_r2_test_b_allrows": best_tb["r2_allrows"],
        "gap_train_oof": full_oof["r2_window"] - best_oof,
        "gap_test_b": full_tb["r2_window"] - best_tb["r2_window"],
        "per_coord_r2_train_oof": per_coord_oof.tolist(),
    }

    # ---- part (ii): energy-information curve -------------------------------
    ks = [k for k in KS if k <= d]
    pca_full = PCA(n_components=max(ks)).fit(tr["z"])
    s_tr = pca_full.transform(tr["z"])
    s_tb = pca_full.transform(tb["z"])
    ev = pca_full.explained_variance_ratio_
    curve = {"ks": ks, "train_oof_r2": {}, "test_b_r2": {}, "energy_fraction": {}}
    for k in ks:
        curve["train_oof_r2"][k] = oof_r2(
            tr["z"], tr["y"], tr["groups"], tr["mask"], pca_k=k
        )["r2_window"]
        curve["test_b_r2"][k] = train_to_test_r2(
            s_tr[:, :k], tr["y"], s_tb[:, :k], tb["y"], tb["mask"]
        )["r2_window"]
        curve["energy_fraction"][k] = float(ev[:k].sum())

    return {
        "inputs": {"train": tr["path"], "test_b": tb["path"]},
        "d": d,
        "n_rows_train": int(tr["z"].shape[0]),
        "n_rows_train_window": int(tr["mask"].sum()),
        "n_rows_test_b": int(tb["z"].shape[0]),
        "n_rows_test_b_window": int(tb["mask"].sum()),
        "n_encounters_train": int(len(np.unique(tr["groups"]))),
        "full_vs_best_coordinate": part_i,
        "energy_information_curve": curve,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--families",
        nargs="+",
        default=list(MODEL_CACHES),
        choices=list(MODEL_CACHES),
        help="Families to analyse (default: all five pooled d = 32 caches).",
    )
    p.add_argument(
        "--output",
        default=str(REPO / "outputs" / "session33" / "wake_code_v2p2.json"),
        help="Output JSON path.",
    )
    args = p.parse_args()

    t0 = time.time()
    results: dict[str, dict] = {}
    for fam in args.families:
        f0 = time.time()
        results[fam] = analyse_family(fam)
        print(f"[wake_code] {fam}: done in {time.time() - f0:.1f}s")

    out = {
        "params": {
            "script": "scripts/session33/wake_code_v2p2.py",
            "spec": "SESSION_33_MANUSCRIPT_V3.md Section 11 item 5 (Section 4.6)",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "target": TARGET,
            "probe": (
                "StandardScaler -> RidgeCV (alphas logspace(-3, 4, 24)), "
                "target-scaled (src.evaluation.represent.fit_linear_probe)"
            ),
            "train_skill": (
                f"{N_FOLDS}-fold GroupKFold OOF grouped by case_id/encounter_index; "
                "R^2 scored on window_mask rows (r2_allrows secondary)"
            ),
            "test_b_skill": "probe fit on ALL train rows, scored on test_b window_mask rows",
            "pca": (
                "test_b curve: PCA fit on ALL train z_gap; train OOF curve: PCA "
                "refit per fold on fold-train rows (leakage-free)"
            ),
            "ks": KS,
            "model_caches": {f: MODEL_CACHES[f] for f in args.families},
        },
        "v2p1_reference": V2P1_REFERENCE,
        "per_family": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[wake_code] wrote {out_path}")

    # ---- console summary ----------------------------------------------------
    print("\n=== full vs best coordinate (wake enstrophy R^2, windowed rows) ===")
    hdr = (
        f"{'family':22s} {'full OOF':>9s} {'best OOF':>9s} {'gap OOF':>8s}   "
        f"{'full tb':>8s} {'best tb':>8s} {'gap tb':>7s}  best_k"
    )
    print(hdr)
    for fam, r in results.items():
        c = r["full_vs_best_coordinate"]
        print(
            f"{fam:22s} {c['full_r2_train_oof']:9.3f} {c['best_coord_r2_train_oof']:9.3f} "
            f"{c['gap_train_oof']:8.3f}   {c['full_r2_test_b']:8.3f} "
            f"{c['best_coord_r2_test_b']:8.3f} {c['gap_test_b']:7.3f}  {c['best_coord']:d}"
        )
    print("\n=== energy-information curve (test_b R^2 vs k leading train PCs) ===")
    print(f"{'family':22s} " + " ".join(f"k={k:<2d}" for k in KS))
    for fam, r in results.items():
        cur = r["energy_information_curve"]
        vals = " ".join(f"{cur['test_b_r2'][k]:+.2f}" for k in cur["ks"])
        print(f"{fam:22s} {vals}")
    print(f"\n[wake_code] total wall time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
