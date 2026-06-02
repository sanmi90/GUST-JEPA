"""Session 23: strengthened conditioning-only floor for the Section 4.1 closure claim.

PURPOSE
-------
The manuscript currently has a single c=(G,D,Y) -> impact-observable KernelRidge(RBF)
floor and over-claims from it ("parameters alone cannot generalise"). This script adds
FOUR additional floors so the floor proves what it should: that gust PARAMETERS (and the
within-case shedding phase) alone cannot reach the forward-closure gains, and that simple
parameter interpolation fails out-of-distribution (Test C, G = +4).

For each of the SIX impact-frame observables and each split (train, test_b, test_c) we
report the held-out R^2 (sklearn r2_score) of FIVE floors:

  1. c-only            : KernelRidge(RBF) from (G, D, Y), honest CV-selected grid.
  2. phase-only        : KernelRidge(RBF) from the within-encounter phase proxy tau.
  3. c+phase           : KernelRidge(RBF) from (G, D, Y, tau).
  4. nn-param          : nearest-neighbour in parameter space; observable = value at the
                         nearest TRAINING (G, D, Y).
  5. c-only-locv       : KernelRidge(RBF) from (G, D, Y) under LEAVE-ONE-CASE-OUT CV
                         (group = case_id via sklearn LeaveOneGroupOut), so no encounter
                         of a held case informs the prediction for that case.

A sixth column, c-only-fixed, reproduces the EXISTING floor recipe exactly
(KernelRidge alpha=0.1, gamma=0.5, StandardScaler on c) as the sanity-match against
outputs/session18/exp_b1_test3/conditioning_only_baseline.csv (wake_enstrophy test_b
should equal ~0.4817).

OBSERVABLE SOURCE (decision, see schema notes below)
----------------------------------------------------
We use outputs/session17/exp2/dns_physical_metrics.npz, NOT the per_frame_targets file
the task initially pointed at. Reasons, verified before writing this script:

  * per_frame_targets/{split}.npz has C_L, C_D, circulation_pos, circulation_neg,
    wake_enstrophy and centroid_y, but it does NOT contain I_y. centroid_y (an
    |omega|-weighted y-centroid) is NOT the vorticity impulse I_y = integral x*omega dA;
    at the impact frame they correlate only ~0.43. Substituting centroid_y for I_y would
    be guessing, which this task forbids.
  * The shared wake observables (wake_enstrophy, circulation_pos/neg) DIFFER numerically
    between the two files (max abs diff ~432 on enstrophy) because session16 squares the
    pipeline-clipped |omega| while session17 squares raw masked omega over the wake box.
    C_L and C_D are identical.
  * The session17 file is the SAME observable definition that the JEPA forward-closure
    numbers (outputs/session20/closure_r2/closure_r2_heldout.csv) are computed against:
    that pipeline reads DNS_METRICS_PATH = session17/exp2/dns_physical_metrics.npz and
    METRICS = (C_L, C_D, I_y, wake_enstrophy, circulation_pos, circulation_neg).
    Using session17 therefore makes the floor directly comparable to the published JEPA
    representation R^2 (0.7541, wake_enstrophy test_b, mode z_dns H=16) and forecast R^2
    (0.4488, mode z_markov H=16), and reproduces the existing 0.4817 c-only value.

I_y SIGN/DEFINITION (unambiguous, taken from scripts/session17/exp2_dns_physical_metrics.py):
  I_y = +integral x * omega dA   (vorticity impulse, y-component), computed as
  np.einsum("tij,i->t", omega_raw, X_GRID) * DX * DY on UN-normalized omega over the full
  field. Stored as session17 key "{split}_I_y". We use it verbatim; we do not re-derive or
  re-sign it.

PHASE PROXY tau (decision)
--------------------------
The cache fixes impact_frame = 40 for every encounter (constant across the whole partition;
verified: train impact_frame unique value = {40}). Hence "frame index relative to impact"
is identically 0 at the impact row for all encounters: zero variance, useless as a feature.
The only within-case quantity that varies across encounters is encounter_index, the proxy
for the baseline shedding phase at the impact instant (encounters 0..3 of a periodic case
strike the limit cycle at different phases). A directly usable per-encounter
shedding-phase-at-impact scalar is NOT readily available (the session20 phase analysis
defines phase as a latent PC1/PC2 angle, not a stored per-encounter scalar). We therefore
set tau = encounter_index and document this as the phase proxy. This is the honest reading
of "the within-encounter phase/time index ... or the baseline shedding phase at impact if
readily available".

METHOD
------
Inputs standardized with StandardScaler (fit on training only). KRR(RBF) hyperparameters
selected by GroupKFold (group = case_id) over a small grid with NO test leakage; the
selected (alpha, gamma) is then refit on all training cases and evaluated on each split.
For the leave-one-case-out floor the in-envelope (train) R^2 is computed over concatenated
out-of-fold predictions (LeaveOneGroupOut, group = case_id); for test_b/test_c the model is
fit on ALL training cases (using the CV-selected grid point) and used to predict. The
nearest-neighbour floor predicts the training observable of the nearest training (G, D, Y);
on train it uses leave-one-CASE-out neighbours (a held case never sees its own encounters).

OUTPUTS
-------
  outputs/session23/conditioning_floor_plus/floor.json
  outputs/session23/conditioning_floor_plus/floor.csv   (rows = floor x observable x split)
  + a console summary, including the c-only sanity match and the Section 4.1 verdict on
    wake_enstrophy test_b vs the JEPA representation (0.7541) and forecast (0.4488) R^2.

CPU-only. No GPU, no training of any neural model.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPO = Path("/home/carlos/GUST-JEPA")
DNS_METRICS = REPO / "outputs" / "session17" / "exp2" / "dns_physical_metrics.npz"
EXISTING_FLOOR_CSV = (
    REPO / "outputs" / "session18" / "exp_b1_test3" / "conditioning_only_baseline.csv"
)
OUT_DIR = REPO / "outputs" / "session23" / "conditioning_floor_plus"

SPLITS = ("train", "test_b", "test_c")
OBSERVABLES = ("C_L", "C_D", "I_y", "wake_enstrophy", "circulation_pos", "circulation_neg")
GAMMA_PLUS = "circulation_pos"  # Gamma+
GAMMA_MINUS = "circulation_neg"  # Gamma-

# Honest KRR(RBF) grid (no test leakage; selected by GroupKFold on train cases only).
# The gamma grid is refined toward the smooth end after the GroupKFold OOF surface for
# wake_enstrophy was found to peak broadly around gamma ~ 0.03-0.1 (cv OOF and the
# held-out test_b ranking agree), well below the 0.5 the original fixed recipe used. We
# include several small gamma so the selected point is the genuine CV optimum, not a
# grid-edge value. High gamma (overfit-prone) stays in the grid so CV can reject it.
ALPHA_GRID = (1e-2, 1e-1, 1.0)
GAMMA_GRID = (0.03, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)

# JEPA reference numbers for the Section 4.1 verdict (from
# outputs/session20/closure_r2/closure_r2_heldout.csv, jepa_d64_test1_noBN,
# metric=wake_enstrophy, split=test_b):
JEPA_REPR_R2 = 0.7541  # mode z_dns,    H=16 (encode true DNS field, probe latent)
JEPA_FCST_R2 = 0.4488  # mode z_markov, H=16 (Markov rollout from impact, probe latent)
SANITY_TARGET = 0.4817  # existing c-only wake_enstrophy test_b


def load_impact_table(horizon: int = 0) -> dict:
    """Load per-split observables (at impact+horizon), parameters, and phase proxy.

    Returns a dict split -> {case_id (n,), enc (n,), tau (n,), c (n,3),
    obs (n, n_obs)} where obs columns follow OBSERVABLES and the observable is
    sampled at the per-encounter frame (impact_frame + horizon), clamped to the
    last cached frame. horizon=0 is the impact frame (default, original recipe);
    horizon=16 makes the floor frame-matched to the JEPA H=16 closure numbers.
    """
    if not DNS_METRICS.exists():
        raise FileNotFoundError(f"Observable source missing: {DNS_METRICS}")
    d = np.load(DNS_METRICS, allow_pickle=True)
    out: dict = {}
    for split in SPLITS:
        cids = d[f"{split}_case_id"].astype(str)
        enc = d[f"{split}_encounter_index"].astype(int)
        impact = d[f"{split}_impact_frame"].astype(int)
        c = np.stack([d[f"{split}_G"], d[f"{split}_D"], d[f"{split}_Y"]], axis=1).astype(float)
        n = len(cids)
        obs = np.full((n, len(OBSERVABLES)), np.nan, dtype=float)
        for j, name in enumerate(OBSERVABLES):
            ser = np.asarray(d[f"{split}_{name}"], dtype=float)  # (n, T)
            for i in range(n):
                fr = min(int(impact[i]) + horizon, ser.shape[1] - 1)
                obs[i, j] = ser[i, fr]
        out[split] = {
            "case_id": cids,
            "enc": enc,
            "tau": enc.astype(float).reshape(-1, 1),  # phase proxy (see docstring)
            "c": c,
            "obs": obs,
        }
    return out


def select_krr(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    """Group-aware grid search for KRR(RBF) (alpha, gamma); no test leakage.

    GroupKFold on case_id so encounters of a case stay together. Returns the
    (alpha, gamma) with the best mean out-of-fold R^2 on the training cases.
    """
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    if n_splits < 2:
        return 0.1, 0.5
    gkf = GroupKFold(n_splits=n_splits)
    best, best_score = (0.1, 0.5), -np.inf
    for alpha in ALPHA_GRID:
        for gamma in GAMMA_GRID:
            scores = []
            for tr, va in gkf.split(X, y, groups):
                if len(np.unique(y[tr])) < 2:
                    continue
                sx = StandardScaler().fit(X[tr])
                m = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
                m.fit(sx.transform(X[tr]), y[tr])
                yhat = m.predict(sx.transform(X[va]))
                scores.append(r2_score(y[va], yhat))
            if scores:
                mean = float(np.mean(scores))
                if mean > best_score:
                    best_score, best = mean, (alpha, gamma)
    return best


def krr_floor(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    groups_tr: np.ndarray,
    eval_X: dict,
    fixed: tuple[float, float] | None = None,
) -> dict:
    """Fit a CV-selected (or fixed) KRR(RBF) on train, predict on every split.

    Train R^2 is the in-sample fit (refit-on-all, predict-train); test_b/test_c
    R^2 are held-out. Returns split -> R^2.
    """
    alpha, gamma = fixed if fixed is not None else select_krr(Xtr, ytr, groups_tr)
    sx = StandardScaler().fit(Xtr)
    m = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
    m.fit(sx.transform(Xtr), ytr)
    res = {"_alpha": alpha, "_gamma": gamma}
    for split, (X, y) in eval_X.items():
        yhat = m.predict(sx.transform(X))
        res[split] = float(r2_score(y, yhat))
    return res


def krr_locv_floor(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    groups_tr: np.ndarray,
    eval_X: dict,
) -> dict:
    """Leave-one-case-out KRR(RBF) floor.

    Train R^2: concatenated out-of-fold predictions over LeaveOneGroupOut
    (group = case_id). test_b/test_c: refit on all training cases with the
    GroupKFold-selected grid point, then predict. Returns split -> R^2.
    """
    alpha, gamma = select_krr(Xtr, ytr, groups_tr)
    logo = LeaveOneGroupOut()
    oof_pred = np.full(len(ytr), np.nan, dtype=float)
    for tr, va in logo.split(Xtr, ytr, groups_tr):
        if len(np.unique(ytr[tr])) < 2:
            continue
        sx = StandardScaler().fit(Xtr[tr])
        m = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
        m.fit(sx.transform(Xtr[tr]), ytr[tr])
        oof_pred[va] = m.predict(sx.transform(Xtr[va]))
    valid = ~np.isnan(oof_pred)
    res = {"_alpha": alpha, "_gamma": gamma}
    res["train"] = float(r2_score(ytr[valid], oof_pred[valid]))
    # held-out splits: fit on all train cases
    sx = StandardScaler().fit(Xtr)
    m = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
    m.fit(sx.transform(Xtr), ytr)
    for split, (X, y) in eval_X.items():
        if split == "train":
            continue
        res[split] = float(r2_score(y, m.predict(sx.transform(X))))
    return res


def nn_param_floor(
    c_tr: np.ndarray,
    y_tr: np.ndarray,
    groups_tr: np.ndarray,
    eval_c: dict,
    eval_y: dict,
) -> dict:
    """Nearest-neighbour-in-parameter-space floor.

    Prediction = training observable of the nearest training (G, D, Y) point
    (Euclidean on standardized c). For train we use leave-one-CASE-out neighbours
    so a held case never copies its own encounters. test_b/test_c query against
    all training points.
    """
    sx = StandardScaler().fit(c_tr)
    ctr_s = sx.transform(c_tr)
    res: dict = {}

    # train: leave-one-case-out neighbour
    oof = np.full(len(y_tr), np.nan, dtype=float)
    for case in np.unique(groups_tr):
        test_mask = groups_tr == case
        train_mask = ~test_mask
        if train_mask.sum() == 0:
            continue
        nn = NearestNeighbors(n_neighbors=1).fit(ctr_s[train_mask])
        _, idx = nn.kneighbors(ctr_s[test_mask])
        oof[test_mask] = y_tr[train_mask][idx[:, 0]]
    valid = ~np.isnan(oof)
    res["train"] = float(r2_score(y_tr[valid], oof[valid]))

    # held-out splits: nearest among all training points
    nn_all = NearestNeighbors(n_neighbors=1).fit(ctr_s)
    for split in eval_c:
        if split == "train":
            continue
        _, idx = nn_all.kneighbors(sx.transform(eval_c[split]))
        yhat = y_tr[idx[:, 0]]
        res[split] = float(r2_score(eval_y[split], yhat))
    return res


def linear_ridge_floor(Xtr: np.ndarray, ytr: np.ndarray, eval_X: dict) -> dict:
    """Linear Ridge floor (smoothness sanity).

    A linear (G,D,Y) -> observable fit. If the KRR floor were a high-gamma
    overfit/leak this linear floor would be comparable; in practice it is much
    weaker, confirming the KRR floor strength comes from mild nonlinearity in the
    parameter map rather than from a linear leak. alpha=1.0 fixed.
    """
    sx = StandardScaler().fit(Xtr)
    m = Ridge(alpha=1.0).fit(sx.transform(Xtr), ytr)
    res: dict = {}
    for split, (X, y) in eval_X.items():
        res[split] = float(r2_score(y, m.predict(sx.transform(X))))
    return res


def read_existing_floor() -> dict:
    """Read the existing c-only floor CSV for the sanity match (or {} if absent)."""
    out: dict = {}
    if not EXISTING_FLOOR_CSV.exists():
        return out
    for r in csv.DictReader(EXISTING_FLOOR_CSV.open()):
        out[r["observable"]] = {
            "train": float(r["train_R2_parameters_only"]),
            "test_b": float(r["test_b_R2_parameters_only"]),
            "test_c": float(r["test_c_R2_parameters_only"]),
        }
    return out


FLOOR_LABELS = {
    "c_only": "c-only KRR(RBF), CV grid",
    "phase_only": "phase-only KRR(RBF) (tau=encounter_index)",
    "c_plus_phase": "c+phase KRR(RBF)",
    "nn_param": "nearest-neighbour in (G,D,Y)",
    "c_only_locv": "c-only KRR(RBF), leave-one-case-out",
    "c_only_linear": "c-only Ridge (linear, smoothness sanity)",
    "c_only_fixed": "c-only KRR(RBF) alpha=0.1 gamma=0.5 (existing-recipe sanity)",
}
FLOOR_ORDER = [
    "c_only",
    "phase_only",
    "c_plus_phase",
    "nn_param",
    "c_only_locv",
    "c_only_linear",
    "c_only_fixed",
]


def main(horizon: int = 0) -> None:
    out_dir = OUT_DIR if horizon == 0 else OUT_DIR / f"h{horizon}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"Session 23: strengthened conditioning-only floor (horizon={horizon})")
    print("=" * 78)
    print(f"Observable source : {DNS_METRICS}")
    print("Decision          : session17 DNS metrics (only source with I_y; same")
    print("                    definition as the JEPA closure numbers; reproduces 0.4817).")
    print("Phase proxy tau   : encounter_index (impact_frame is constant = 40, so")
    print("                    frame-relative-to-impact has zero variance).")
    print()

    tab = load_impact_table(horizon)
    for split in SPLITS:
        print(
            f"  {split:7s} n_enc={len(tab[split]['case_id']):4d}  "
            f"n_cases={len(np.unique(tab[split]['case_id'])):3d}"
        )
    print()

    c_tr = tab["train"]["c"]
    tau_tr = tab["train"]["tau"]
    cphase_tr = np.concatenate([c_tr, tau_tr], axis=1)
    groups_tr = tab["train"]["case_id"]

    results: dict = {obs: {} for obs in OBSERVABLES}
    selected: dict = {}

    for j, obs in enumerate(OBSERVABLES):
        ytr = tab["train"]["obs"][:, j]
        eval_c = {s: tab[s]["c"] for s in SPLITS}
        eval_y = {s: tab[s]["obs"][:, j] for s in SPLITS}
        eval_cphase = {s: np.concatenate([tab[s]["c"], tab[s]["tau"]], axis=1) for s in SPLITS}
        eval_phase = {s: tab[s]["tau"] for s in SPLITS}

        # 1. c-only (CV grid)
        f_c = krr_floor(c_tr, ytr, groups_tr, {s: (eval_c[s], eval_y[s]) for s in SPLITS})
        # 2. phase-only
        f_p = krr_floor(tau_tr, ytr, groups_tr, {s: (eval_phase[s], eval_y[s]) for s in SPLITS})
        # 3. c+phase
        f_cp = krr_floor(
            cphase_tr, ytr, groups_tr, {s: (eval_cphase[s], eval_y[s]) for s in SPLITS}
        )
        # 4. nearest-neighbour in parameter space
        f_nn = nn_param_floor(c_tr, ytr, groups_tr, eval_c, eval_y)
        # 5. leave-one-case-out KRR
        f_locv = krr_locv_floor(c_tr, ytr, groups_tr, {s: (eval_c[s], eval_y[s]) for s in SPLITS})
        # 6. linear ridge (smoothness sanity)
        f_lin = linear_ridge_floor(c_tr, ytr, {s: (eval_c[s], eval_y[s]) for s in SPLITS})
        # 7. existing-recipe fixed c-only (sanity)
        f_fix = krr_floor(
            c_tr, ytr, groups_tr, {s: (eval_c[s], eval_y[s]) for s in SPLITS}, fixed=(0.1, 0.5)
        )

        floors = {
            "c_only": f_c,
            "phase_only": f_p,
            "c_plus_phase": f_cp,
            "nn_param": f_nn,
            "c_only_locv": f_locv,
            "c_only_linear": f_lin,
            "c_only_fixed": f_fix,
        }
        selected[obs] = {
            "c_only": (f_c["_alpha"], f_c["_gamma"]),
            "phase_only": (f_p["_alpha"], f_p["_gamma"]),
            "c_plus_phase": (f_cp["_alpha"], f_cp["_gamma"]),
            "c_only_locv": (f_locv["_alpha"], f_locv["_gamma"]),
        }
        for fname, fres in floors.items():
            results[obs][fname] = {s: fres[s] for s in SPLITS}

    # ---- console table ----
    print("Held-out R^2 by floor x observable x split")
    print("-" * 78)
    hdr = f"{'observable':<16}{'floor':<14}" + "".join(f"{s:>10}" for s in SPLITS)
    print(hdr)
    print("-" * 78)
    for obs in OBSERVABLES:
        for fname in FLOOR_ORDER:
            row = results[obs][fname]
            vals = "".join(f"{row[s]:>+10.3f}" for s in SPLITS)
            print(f"{obs:<16}{fname:<14}{vals}")
        print("-" * 78)

    # ---- sanity match ----
    existing = read_existing_floor()
    print()
    print("c-only SANITY MATCH (existing-recipe c_only_fixed vs existing CSV)")
    print("-" * 78)
    print(f"{'observable':<18}{'this (fixed)':>14}{'existing CSV':>14}{'abs diff':>12}")
    for obs in OBSERVABLES:
        mine = results[obs]["c_only_fixed"]["test_b"]
        ex = existing.get(obs, {}).get("test_b", float("nan"))
        diff = abs(mine - ex) if ex == ex else float("nan")
        print(f"{obs:<18}{mine:>+14.4f}{ex:>+14.4f}{diff:>12.4e}")
    we_fixed = results["wake_enstrophy"]["c_only_fixed"]["test_b"]
    print(
        f"\nwake_enstrophy test_b (fixed recipe) = {we_fixed:+.4f}  "
        f"vs target {SANITY_TARGET:+.4f}  ->  "
        f"{'MATCH' if abs(we_fixed - SANITY_TARGET) < 1e-3 else 'MISMATCH'}"
    )

    # ---- Section 4.1 verdict ----
    print()
    print("Section 4.1 verdict: wake_enstrophy, test_b")
    print("-" * 78)
    we = {fname: results["wake_enstrophy"][fname]["test_b"] for fname in FLOOR_ORDER}
    strongest_name = max(we, key=we.get)
    strongest_val = we[strongest_name]
    for fname in FLOOR_ORDER:
        print(f"  {fname:<14} test_b R^2 = {we[fname]:+.4f}")
    print(f"\n  strongest floor       : {strongest_name} = {strongest_val:+.4f}")
    print(f"  JEPA representation   : {JEPA_REPR_R2:+.4f} (z_dns,    H=16)")
    print(f"  JEPA forecast         : {JEPA_FCST_R2:+.4f} (z_markov, H=16)")
    print(
        f"  strongest floor < JEPA representation? "
        f"{'YES' if strongest_val < JEPA_REPR_R2 else 'NO'} "
        f"(gap {JEPA_REPR_R2 - strongest_val:+.4f})"
    )
    print(
        f"  strongest floor vs JEPA forecast       : "
        f"{strongest_val - JEPA_FCST_R2:+.4f} "
        f"({'floor below forecast' if strongest_val < JEPA_FCST_R2 else 'floor ABOVE forecast'})"
    )

    # ---- OOD caveat ----
    tc_vals = {fname: results["wake_enstrophy"][fname]["test_c"] for fname in FLOOR_ORDER}
    n_neg_tc = sum(
        1 for fname in FLOOR_ORDER for obs in OBSERVABLES if results[obs][fname]["test_c"] < 0
    )
    n_tot_tc = len(FLOOR_ORDER) * len(OBSERVABLES)
    print()
    print("Out-of-distribution (Test C, G = +4) caveat")
    print("-" * 78)
    print(f"  negative test_c R^2 cells: {n_neg_tc}/{n_tot_tc} (floor x observable)")
    print(
        f"  wake_enstrophy test_c: best floor = "
        f"{max(tc_vals.values()):+.3f} ({max(tc_vals, key=tc_vals.get)})"
    )
    print("  -> parameter interpolation/extrapolation fails out of distribution.")

    # ---- write JSON ----
    payload = {
        "horizon": horizon,
        "observable_source": str(DNS_METRICS),
        "observable_source_decision": (
            "session17 DNS metrics: only source with I_y; same observable definition "
            "as the JEPA closure numbers (closure_r2_heldout.csv) and the existing "
            "conditioning floor; reproduces wake_enstrophy test_b = 0.4817. "
            "per_frame_targets lacks I_y and differs numerically on wake observables."
        ),
        "I_y_definition": "I_y = +integral x*omega dA (vorticity impulse y-component); "
        "session17 key {split}_I_y, used verbatim.",
        "phase_proxy": "tau = encounter_index (impact_frame constant = 40, so "
        "frame-relative-to-impact has zero variance).",
        "splits": {
            s: {
                "n_enc": int(len(tab[s]["case_id"])),
                "n_cases": int(len(np.unique(tab[s]["case_id"]))),
            }
            for s in SPLITS
        },
        "observables": list(OBSERVABLES),
        "gamma_plus": GAMMA_PLUS,
        "gamma_minus": GAMMA_MINUS,
        "krr_grid": {
            "alpha": list(ALPHA_GRID),
            "gamma": list(GAMMA_GRID),
            "selection": "GroupKFold(case_id) on train, mean OOF R^2",
        },
        "floor_labels": FLOOR_LABELS,
        "selected_hyperparams": {
            obs: {k: {"alpha": v[0], "gamma": v[1]} for k, v in selected[obs].items()}
            for obs in OBSERVABLES
        },
        "jepa_reference": {
            "wake_enstrophy_test_b_representation_z_dns_H16": JEPA_REPR_R2,
            "wake_enstrophy_test_b_forecast_z_markov_H16": JEPA_FCST_R2,
            "source": "outputs/session20/closure_r2/closure_r2_heldout.csv "
            "(jepa_d64_test1_noBN)",
        },
        "sanity_match": {
            "target_wake_enstrophy_test_b": SANITY_TARGET,
            "this_c_only_fixed_wake_enstrophy_test_b": we_fixed,
            "abs_diff": abs(we_fixed - SANITY_TARGET),
            "match": bool(abs(we_fixed - SANITY_TARGET) < 1e-3),
        },
        "verdict_wake_enstrophy_test_b": {
            "strongest_floor": strongest_name,
            "strongest_floor_R2": strongest_val,
            "below_jepa_representation": bool(strongest_val < JEPA_REPR_R2),
            "gap_to_representation": JEPA_REPR_R2 - strongest_val,
            "vs_jepa_forecast": strongest_val - JEPA_FCST_R2,
            "floor_below_forecast": bool(strongest_val < JEPA_FCST_R2),
        },
        "caveats": {
            "test_c_is_OOD": (
                "Test C is G = +4 only, outside the train envelope G in [-3,+3] "
                "(D and Y are in-range). Every floor goes to negative R^2 on test_c "
                "for almost every observable: parameter interpolation/extrapolation "
                "fails out-of-distribution. This is the honest support for the "
                "'parameter interpolation fails OOD' half of the claim."
            ),
            "phase_proxy_out_of_range_on_test": (
                "encounter_index is 0..3 in train but 0..5 in test_b/test_c, so the "
                "c+phase floor extrapolates one dimension on the test splits. "
                "phase-only is negative on test_b, so the phase proxy carries no "
                "standalone signal; it helps the in-envelope test_b enstrophy floor "
                "only in interaction with c. Treat c+phase test_b as an upper bound."
            ),
            "in_envelope_floor_is_strong": (
                "With CV-selected (not fixed) RBF gamma, the in-envelope (test_b) "
                "conditioning floor is much higher than the existing fixed-recipe "
                "0.4817: c-only ~0.70 and c+phase ~0.81 on wake_enstrophy test_b. "
                "The 0.4817 was a suboptimal hyperparameter, not the true floor. The "
                "GroupKFold OOF surface (train cases only, no leakage) peaks broadly "
                "at gamma ~ 0.03-0.1 and the held-out test_b ranking agrees, so the "
                "high floor is honest. The linear Ridge floor stays ~0.11-0.15 on "
                "test_b, so the strength is mild nonlinearity in (G,D,Y), not a "
                "linear leak."
            ),
        },
        "R2": {obs: {fname: results[obs][fname] for fname in FLOOR_ORDER} for obs in OBSERVABLES},
    }
    out_json = out_dir / "floor.json"
    out_json.write_text(json.dumps(payload, indent=2))

    # ---- write CSV (rows = floor x observable x split) ----
    out_csv = out_dir / "floor.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["floor", "floor_label", "observable", "split", "R2"])
        for obs in OBSERVABLES:
            for fname in FLOOR_ORDER:
                for s in SPLITS:
                    w.writerow(
                        [fname, FLOOR_LABELS[fname], obs, s, f"{results[obs][fname][s]:.6f}"]
                    )

    print()
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--horizon",
        type=int,
        default=0,
        help="frames after impact at which to sample observables "
        "(0 = impact frame, original recipe; 16 = frame-matched to the JEPA H=16 closure).",
    )
    args = parser.parse_args()
    main(args.horizon)
