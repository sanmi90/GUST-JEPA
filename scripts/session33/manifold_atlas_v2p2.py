"""SESSION 33: parametric manifold atlas data + (G, D, Y) probes on v2.2 pooled latents.

SESSION_33_MANUSCRIPT_V3.md Section 11 item 4 (Section 4.6, figure F9): the parametric
manifold atlas [RE-RUN] on the v2.2 pooled d=32 latents. Per family this script (a)
fits a 3-component PCA on the TRAIN per-frame pooled latents (z_gap) and stores the
per-frame PCA coordinates for train + test_b with (case_id, G, D, Y, split, frame,
phase) labels, and (b) recomputes the (G, D, Y) parameter probes with encounter-grouped
cross-validation. The figure itself is produced later from the coords npz; this script
emits data only.

PROBE REGIME (binding rule, CLAUDE.md / Session 16 D118-bis, D120-bis): the (G, D, Y)
parameter probes use IMPACT-FRAME latents, one row per encounter taken at that
encounter's t_impact from outputs/session31/windows_v2p2.json. Per-frame parameter
probes are known to fail (all probe families test_b Y R^2 < 0) and are NOT computed
here. The per-frame latents are used for the atlas PCA coordinates only.

Probe protocol: StandardScaler + Ridge, alpha selected by grouped GridSearchCV
(GroupKFold by case_id). train_cv_r2 is a nested out-of-fold estimate: outer
GroupKFold(5) by case_id, alpha re-selected inside each outer-train split. test_b_r2
fits on all train impact rows and scores on test_b impact rows. A KernelRidge(RBF)
variant is additionally reported for Y (v2.1 found Y needs a nonlinear probe).

Outputs:
    outputs/session33/manifold_atlas_v2p2.json          (probes + PCA explained variance)
    outputs/session33/manifold_atlas_v2p2_coords.npz    (per-frame PCA coords + labels)

CPU-only (numpy/sklearn). Run with a core cap, e.g.:
    taskset -c 16-23 .venv/bin/python scripts/session33/manifold_atlas_v2p2.py
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
SPLIT_JSON = REPO / "configs" / "splits" / "split_v2p2.json"
WINDOWS_JSON = REPO / "outputs" / "session31" / "windows_v2p2.json"
OUT_DIR = REPO / "outputs" / "session33"
OUT_JSON = OUT_DIR / "manifold_atlas_v2p2.json"
OUT_COORDS = OUT_DIR / "manifold_atlas_v2p2_coords.npz"

S31 = REPO / "outputs" / "session31" / "q1_latents"
S32 = REPO / "outputs" / "session32" / "q1_pool_latents"

S33V = REPO / "outputs" / "session33" / "q1_vec_latents"
FAMILIES = {
    # D250 flagship: the predictive family is the native-vector jepa_pool_vec.
    "jepa_pool": {
        "train": S33V / "latents_jepa_pool_vec_train.npz",
        "test_b": S33V / "latents_jepa_pool_vec_test_b.npz",
    },
    "supervised_only_pool": {
        "train": S32 / "latents_supervised_only_pool_train.npz",
        "test_b": S32 / "latents_supervised_only_pool_test_b.npz",
    },
    "fukami": {
        "train": S31 / "latents_fukami_train.npz",
        "test_b": S31 / "latents_fukami_test_b.npz",
    },
    "pod": {
        "train": S31 / "latents_pod_train.npz",
        "test_b": S31 / "latents_pod_test_b.npz",
    },
}

TARGETS = ("G", "D", "Y")
RIDGE_ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
KRR_ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
KRR_GAMMAS = [0.003, 0.01, 0.03, 0.1, 0.3]
SEED = 0

# v2.1 reference values (linear-probe era) for comparison, per the session plan.
V2P1_REFERENCE_TEST_B = {
    "G": 0.83,
    "D": 0.65,
    "Y": -0.03,
    "note": (
        "v2.1 linear-probe test_b R^2; qualitative expectation on v2.2 is "
        "G strong, D medium, Y marginal (Y needed a nonlinear probe in v2.1)"
    ),
}


def load_case_params() -> dict[str, tuple[float, float, float]]:
    """case_id -> (G, D, Y) from the v2.2 split manifest (Baseline has G=D=Y=0)."""
    blob = json.loads(SPLIT_JSON.read_text())
    return {
        cid: (float(c["G"]), float(c["D"]), float(c["Y"]))
        for cid, c in blob["cases"].items()
    }


def load_impact_frames() -> dict[str, int]:
    """'<case_id>/<enc:02d>' -> t_impact from the session31 canonical windows."""
    blob = json.loads(WINDOWS_JSON.read_text())
    return {k: int(v["t_impact"]) for k, v in blob["windows"].items()}


def load_family_split(path: Path, split_name: str, params: dict, impacts: dict) -> dict:
    """Load one latents npz; return per-frame arrays plus per-row phase and t_impact."""
    d = np.load(path, allow_pickle=True)
    z = np.asarray(d["z_gap"], dtype=np.float64)
    cid = np.asarray(d["case_id"]).astype(str)
    enc = np.asarray(d["encounter_index"]).astype(int)
    frame = np.asarray(d["frame"]).astype(int)
    keys = np.array([f"{c}/{e:02d}" for c, e in zip(cid, enc)])
    missing = sorted({k for k in np.unique(keys) if k not in impacts})
    if missing:
        raise KeyError(f"{path.name}: encounters missing from windows_v2p2.json: {missing[:5]}")
    t_imp = np.array([impacts[k] for k in keys], dtype=int)
    gdy = np.array([params[c] for c in cid], dtype=np.float64)
    return {
        "z": z,
        "case_id": cid,
        "encounter_index": enc,
        "frame": frame,
        "phase": frame - t_imp,
        "t_impact": t_imp,
        "G": gdy[:, 0],
        "D": gdy[:, 1],
        "Y": gdy[:, 2],
        "split": np.full(len(cid), split_name),
    }


def impact_rows(fam_split: dict) -> dict:
    """One row per encounter: the frame at t_impact (impact-frame probe regime)."""
    mask = fam_split["frame"] == fam_split["t_impact"]
    n_enc = len(set(zip(fam_split["case_id"].tolist(), fam_split["encounter_index"].tolist())))
    if int(mask.sum()) != n_enc:
        raise RuntimeError(f"impact-row extraction: {int(mask.sum())} rows for {n_enc} encounters")
    return {k: v[mask] for k, v in fam_split.items()}


def ridge_grid(n_groups: int) -> GridSearchCV:
    pipe = Pipeline([("s", StandardScaler()), ("m", Ridge())])
    cv = GroupKFold(n_splits=min(5, n_groups))
    return GridSearchCV(pipe, {"m__alpha": RIDGE_ALPHAS}, scoring="r2", cv=cv, n_jobs=1)


def krr_grid(n_groups: int) -> GridSearchCV:
    pipe = Pipeline([("s", StandardScaler()), ("m", KernelRidge(kernel="rbf"))])
    cv = GroupKFold(n_splits=min(5, n_groups))
    grid = {"m__alpha": KRR_ALPHAS, "m__gamma": KRR_GAMMAS}
    return GridSearchCV(pipe, grid, scoring="r2", cv=cv, n_jobs=1)


def nested_oof_r2(make_grid, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """Nested OOF-CV train R^2: outer GroupKFold(5) by case_id, inner grouped alpha search."""
    outer = GroupKFold(n_splits=5)
    oof = np.full(len(y), np.nan)
    for tr, te in outer.split(X, y, groups):
        gs = make_grid(len(np.unique(groups[tr])))
        gs.fit(X[tr], y[tr], groups=groups[tr])
        oof[te] = gs.predict(X[te])
    assert not np.isnan(oof).any()
    return float(r2_score(y, oof))


def fit_and_score(make_grid, Xtr, ytr, gtr, Xte, yte) -> tuple[float, dict]:
    """Fit grouped-CV grid on all train impact rows; return test_b R^2 and best params."""
    gs = make_grid(len(np.unique(gtr)))
    gs.fit(Xtr, ytr, groups=gtr)
    best = {k.replace("m__", ""): float(v) for k, v in gs.best_params_.items()}
    return float(r2_score(yte, gs.predict(Xte))), best


def run_probes(imp_tr: dict, imp_te: dict) -> dict:
    """Ridge probes for G, D, Y (plus KernelRidge RBF for Y) on impact-frame latents."""
    Xtr, gtr = imp_tr["z"], imp_tr["case_id"]
    Xte = imp_te["z"]
    out = {}
    for tgt in TARGETS:
        ytr, yte = imp_tr[tgt], imp_te[tgt]
        cv_r2 = nested_oof_r2(ridge_grid, Xtr, ytr, gtr)
        te_r2, best = fit_and_score(ridge_grid, Xtr, ytr, gtr, Xte, yte)
        entry = {
            "probe_kind": "ridge",
            "train_cv_r2": cv_r2,
            "test_b_r2": te_r2,
            "best_params": best,
        }
        if tgt == "Y":
            k_cv = nested_oof_r2(krr_grid, Xtr, ytr, gtr)
            k_te, k_best = fit_and_score(krr_grid, Xtr, ytr, gtr, Xte, yte)
            entry["krr_rbf"] = {
                "train_cv_r2": k_cv,
                "test_b_r2": k_te,
                "best_params": k_best,
            }
        out[tgt] = entry
    return out


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    args = ap.parse_args()

    np.random.seed(SEED)
    params = load_case_params()
    impacts = load_impact_frames()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    coords_payload: dict[str, np.ndarray] = {}
    fam_results: dict[str, dict] = {}

    for fam in args.families:
        paths = FAMILIES[fam]
        tr = load_family_split(paths["train"], "train", params, impacts)
        te = load_family_split(paths["test_b"], "test_b", params, impacts)

        # Atlas: PCA(3) fit on TRAIN per-frame z_gap, applied to train + test_b.
        pca = PCA(n_components=3, random_state=SEED)
        pca.fit(tr["z"])
        evr = pca.explained_variance_ratio_.astype(float)
        coords = np.concatenate([pca.transform(tr["z"]), pca.transform(te["z"])], axis=0)
        cat = {
            k: np.concatenate([tr[k], te[k]])
            for k in ("case_id", "encounter_index", "frame", "phase", "split", "G", "D", "Y")
        }
        coords_payload[f"{fam}_coords"] = coords.astype(np.float32)
        coords_payload[f"{fam}_case_id"] = cat["case_id"]
        coords_payload[f"{fam}_encounter_index"] = cat["encounter_index"].astype(np.int16)
        coords_payload[f"{fam}_frame"] = cat["frame"].astype(np.int16)
        coords_payload[f"{fam}_phase"] = cat["phase"].astype(np.int16)
        coords_payload[f"{fam}_split"] = cat["split"]
        for p in ("G", "D", "Y"):
            coords_payload[f"{fam}_{p}"] = cat[p].astype(np.float32)
        coords_payload[f"{fam}_explained_variance_ratio"] = evr

        # Probes: IMPACT-FRAME regime, one row per encounter, grouped by case_id.
        imp_tr, imp_te = impact_rows(tr), impact_rows(te)
        probes = run_probes(imp_tr, imp_te)

        fam_results[fam] = {
            "pca_explained_variance_ratio": evr.tolist(),
            "n_train_frames": int(len(tr["frame"])),
            "n_test_b_frames": int(len(te["frame"])),
            "n_train_impact_rows": int(len(imp_tr["frame"])),
            "n_test_b_impact_rows": int(len(imp_te["frame"])),
            "n_train_cases": int(len(np.unique(imp_tr["case_id"]))),
            "probes": probes,
        }

        pg, pd, py = (probes[t] for t in TARGETS)
        print(
            f"[{fam:>20s}] PCA evr={np.round(evr, 3).tolist()} | "
            f"G cv={pg['train_cv_r2']:+.3f} tb={pg['test_b_r2']:+.3f} | "
            f"D cv={pd['train_cv_r2']:+.3f} tb={pd['test_b_r2']:+.3f} | "
            f"Y cv={py['train_cv_r2']:+.3f} tb={py['test_b_r2']:+.3f} "
            f"(krr Y cv={py['krr_rbf']['train_cv_r2']:+.3f} "
            f"tb={py['krr_rbf']['test_b_r2']:+.3f})"
        )

    np.savez_compressed(OUT_COORDS, **coords_payload)

    payload = {
        "_provenance": {
            "script": "scripts/session33/manifold_atlas_v2p2.py",
            "created_iso": datetime.datetime.now().isoformat(timespec="seconds"),
            "git_head": git_head(),
            "split_manifest": str(SPLIT_JSON.relative_to(REPO)),
            "windows": str(WINDOWS_JSON.relative_to(REPO)),
            "inputs": {
                fam: {k: str(v.relative_to(REPO)) for k, v in FAMILIES[fam].items()}
                for fam in args.families
            },
            "coords_npz": str(OUT_COORDS.relative_to(REPO)),
            "seed": SEED,
        },
        "params": {
            "latent_key": "z_gap",
            "pca_components": 3,
            "pca_fit_on": "train per-frame z_gap",
            "probe_regime": "impact-frame (one row per encounter at windows t_impact)",
            "probe_cv": "nested OOF: outer GroupKFold(5) by case_id, inner grouped alpha search",
            "ridge_alphas": RIDGE_ALPHAS,
            "krr_alphas": KRR_ALPHAS,
            "krr_gammas": KRR_GAMMAS,
        },
        "v2p1_reference_test_b": V2P1_REFERENCE_TEST_B,
        "families": fam_results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_COORDS}")


if __name__ == "__main__":
    main()
