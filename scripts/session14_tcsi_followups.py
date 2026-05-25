"""Session 14 Thrust 7 follow-ups: TCSI confirmation at K = 2 / 3 / 4.

Implements the four follow-ups that turn the Thrust 7 sensor pilot into a
publishable headline:

  1. TCN proxy learner at K = 2 / 3 / 4 to confirm the TCSI > qDEIM
     ordering survives a stronger model.
  2. Bootstrap-resampled greedy selection at K = 2 / 3 / 4 to test whether
     the LE-cluster sensors (11 and 20) are stable choices.
  3. Per-(G, D, Y) regime stability sweep at K = 2 and K = 4.
  4. Decoded flow-field figure at K = 2 versus the all-192 Ridge
     reconstruction, on two canonical Test B encounters.

This script REUSES helpers from ``scripts/session14_tcsi_pilot.py``
(``build_data_arrays``, ``greedy_forward_selection``, ``selector_qdeim``,
``cv_r2``, ``cv_rmse``, ``_subset_features``, ``load_airfoil_sensor_positions``)
and adds confirmation runs on top of them.

Usage (from the repo root after activating the venv and exporting env)::

    source .venv/bin/activate
    export PREVENT_ROOT=$HOME/PREVENT WANDB_PROJECT=vortex-jepa
    python scripts/session14_tcsi_followups.py --items 1 2 3 4

Use ``--items 2 3`` to run only a subset; item 4 needs a free RTX 6000.
Outputs land under ``outputs/session14/tcsi_pilot/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Limit BLAS thread oversubscription consistent with the pilot.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "4")

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.session14_tcsi_pilot import (  # noqa: E402
    HALF_WINDOW,
    N_SENSORS,
    WINDOW,
    _subset_features,
    _tolist,
    build_data_arrays,
    cv_r2,
    cv_rmse,
    greedy_forward_selection,
    load_airfoil_sensor_positions,
    selector_qdeim,
    selector_uniform,
    selector_random,
)
from src.evaluation.tcn_proxy_learner import TCNConfig, TCNProxyLearner  # noqa: E402


PILOT_OUT_DIR = REPO_ROOT / "outputs" / "session14" / "tcsi_pilot"
LATENT_DIR = REPO_ROOT / "outputs" / "session14" / "latents" / "S12_E_d64"
DECODER_CKPT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "session12"
    / "S12_E_d64"
    / "encoder"
    / "decoder_specloss_recipe"
    / "decoder_iter012000.pt"
)
ENCODER_CKPT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "session12"
    / "S12_E_d64"
    / "encoder"
    / "checkpoint_iter020000.pt"
)
OMEGA_PIPELINE_MANIFEST = (
    REPO_ROOT / "outputs" / "data_pipeline" / "v1" / "manifest.json"
)
K_VALUES_TCN: Tuple[int, ...] = (2, 3, 4)
TCN_SELECTORS: Tuple[str, ...] = ("uniform_K", "random_K_median", "qDEIM", "TCSI")
TCN_FOLDS = 5
BOOT_SEEDS = 50
TCN_DEFAULT_EPOCHS = 200
ITEM4_CASES: Tuple[Tuple[str, int], ...] = (
    ("G+1.00_D1.00_Y+0.10", 0),
    ("G-1.50_D0.50_Y-0.20", 0),
)


# ---------------------------------------------------------------------------
# Item 1: TCN proxy learner at K = 2 / 3 / 4
# ---------------------------------------------------------------------------


def _select_random_median_seed(
    pool_X: np.ndarray,
    pool_Y_z_pc1: np.ndarray,
    K: int,
    n_seeds: int,
) -> Tuple[int, List[int]]:
    """Return the seed whose random-K subset has median ridge ``z_R2`` on the pool.

    Picks the seed with the median (rounded down) ridge ``z_R2`` over
    ``range(n_seeds)``; ties are broken by the smaller seed. This is a
    cheap, deterministic stand-in for "the random sensor set the pilot
    would have called typical" without re-running 50 TCN fits per K.
    """
    r2s: List[Tuple[float, int, List[int]]] = []
    for s in range(n_seeds):
        sensors = selector_random(K, seed=s)
        feats = _subset_features(pool_X, sensors)
        r2 = cv_r2(feats, pool_Y_z_pc1, n_splits=5, alpha=1.0)
        r2s.append((float(r2), s, sensors))
    r2s.sort(key=lambda t: (t[0], t[1]))
    median_idx = len(r2s) // 2
    _, seed, sensors = r2s[median_idx]
    return int(seed), sensors


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Variance-weighted mean R^2 for 1D or 2D targets."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.ndim == 1:
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        if ss_tot == 0.0:
            return float("nan")
        return 1.0 - ss_res / ss_tot
    ss_res = np.sum((yt - yp) ** 2, axis=0)
    ss_tot = np.sum((yt - yt.mean(axis=0, keepdims=True)) ** 2, axis=0)
    out = np.zeros_like(ss_tot)
    mask = ss_tot > 0
    out[mask] = 1.0 - ss_res[mask] / ss_tot[mask]
    out[~mask] = np.nan
    return float(np.nanmean(out))


def _cv_metric_tcn(
    X: np.ndarray,
    y: np.ndarray,
    out_dim: int,
    epochs: int,
    device: str,
    metric: str = "r2",
    seed: int = 0,
) -> float:
    """5-fold CV with the TCN; metric in {"r2", "rmse"}."""
    n = X.shape[0]
    if n < TCN_FOLDS:
        n_splits = max(2, n)
    else:
        n_splits = TCN_FOLDS
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    for fold, (tr, te) in enumerate(kf.split(X)):
        cfg = TCNConfig(epochs=epochs, device=device, seed=seed + fold)
        learner = TCNProxyLearner(out_dim=out_dim, config=cfg)
        learner.fit(X[tr], y[tr])
        y_pred = learner.predict(X[te])
        y_true_all.append(y[te])
        y_pred_all.append(y_pred)
    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    if metric == "r2":
        return _r2_score(y_true, y_pred)
    if metric == "rmse":
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    raise ValueError(f"unknown metric {metric!r}")


def _evaluate_tcn(
    sensors: Sequence[int],
    pool_X: np.ndarray,
    pool_Y_z: np.ndarray,
    pool_Y_CL: np.ndarray,
    pool_Y_phase: np.ndarray,
    epochs: int,
    device: str,
) -> Dict[str, float]:
    """5-fold CV TCN metrics on the selection pool for one sensor subset.

    The pool already contains train + test_a; per the pilot we evaluate
    the TCN with cross-validation on the same pool so the ranking is a
    direct generalisation of the ridge ranking. Numbers from Test B / C
    splits are not retraining-bound and are evaluated by the ridge head
    in item 4 (decoded field reconstruction).
    """
    if len(sensors) == 0:
        return {"z_R2": float("nan"), "CL_R2": float("nan"), "phase_RMSE": float("nan")}
    feats = pool_X[:, list(sensors), :]
    z_r2 = _cv_metric_tcn(feats, pool_Y_z, out_dim=pool_Y_z.shape[1],
                          epochs=epochs, device=device, metric="r2", seed=0)
    cl_r2 = _cv_metric_tcn(feats, pool_Y_CL, out_dim=1,
                           epochs=epochs, device=device, metric="r2", seed=0)
    phase_rmse = _cv_metric_tcn(
        feats, pool_Y_phase.astype(np.float64), out_dim=1,
        epochs=epochs, device=device, metric="rmse", seed=0,
    )
    return {"z_R2": float(z_r2), "CL_R2": float(cl_r2), "phase_RMSE": float(phase_rmse)}


def _evaluate_ridge_pool(
    sensors: Sequence[int],
    pool_X: np.ndarray,
    pool_Y_z: np.ndarray,
    pool_Y_CL: np.ndarray,
    pool_Y_phase: np.ndarray,
) -> Dict[str, float]:
    """5-fold CV Ridge metrics on the selection pool for one sensor subset."""
    if len(sensors) == 0:
        return {"z_R2": float("nan"), "CL_R2": float("nan"), "phase_RMSE": float("nan")}
    feats = _subset_features(pool_X, sensors)
    z_r2 = cv_r2(feats, pool_Y_z, n_splits=TCN_FOLDS, alpha=1.0)
    cl_r2 = cv_r2(feats, pool_Y_CL, n_splits=TCN_FOLDS, alpha=1.0)
    phase_rmse = cv_rmse(
        feats, pool_Y_phase.astype(np.float64), n_splits=TCN_FOLDS, alpha=1.0
    )
    return {"z_R2": float(z_r2), "CL_R2": float(cl_r2), "phase_RMSE": float(phase_rmse)}


def run_item1_tcn_followup(
    pool_X: np.ndarray,
    pool_Y_z: np.ndarray,
    pool_Y_CL: np.ndarray,
    pool_Y_phase: np.ndarray,
    greedy_selections: Dict[int, List[int]],
    epochs: int,
    device: str,
    out_dir: Path,
) -> Dict[str, object]:
    """TCN vs Ridge at K = 2 / 3 / 4, plus the all-192 Ridge baseline."""
    print(f"[item1] TCN epochs={epochs} device={device}", flush=True)
    pool_Y_z_pc1 = PCA(n_components=1, random_state=0).fit_transform(
        pool_Y_z
    ).reshape(-1)

    # Pre-compute qDEIM at each K from the impact-frame pressure matrix.
    P_imp = pool_X[:, :, HALF_WINDOW]
    qdeim_K: Dict[int, List[int]] = {
        K: selector_qdeim(P_imp, K) for K in K_VALUES_TCN
    }
    uniform_K: Dict[int, List[int]] = {K: selector_uniform(K) for K in K_VALUES_TCN}

    # Random-K median selection: pick the seed with median ridge z_R2 on the pool.
    random_K_med: Dict[int, Tuple[int, List[int]]] = {}
    for K in K_VALUES_TCN:
        seed, sensors = _select_random_median_seed(
            pool_X, pool_Y_z_pc1, K, n_seeds=BOOT_SEEDS
        )
        random_K_med[K] = (seed, sensors)
        print(f"[item1] random_K_median K={K}: seed={seed} sensors={sensors}",
              flush=True)

    selectors_by_K: Dict[int, Dict[str, List[int]]] = {}
    for K in K_VALUES_TCN:
        selectors_by_K[K] = {
            "uniform_K": uniform_K[K],
            "random_K_median": random_K_med[K][1],
            "qDEIM": qdeim_K[K],
            "TCSI": list(greedy_selections[K]),
        }

    # all_192 ridge baseline at each K position in the results table.
    all_sensors = list(range(N_SENSORS))
    ridge_all_192 = _evaluate_ridge_pool(
        all_sensors, pool_X, pool_Y_z, pool_Y_CL, pool_Y_phase
    )
    print(f"[item1] all_192 Ridge baseline: {ridge_all_192}", flush=True)

    results: Dict[str, object] = {
        "epochs": int(epochs),
        "device": device,
        "K_values": list(K_VALUES_TCN),
        "selectors": list(TCN_SELECTORS),
        "selection_pool_n": int(pool_X.shape[0]),
        "selectors_by_K": {
            f"K_{K}": {name: [int(s) for s in sel]
                       for name, sel in selectors_by_K[K].items()}
            for K in K_VALUES_TCN
        },
        "random_K_median_seed": {f"K_{K}": int(random_K_med[K][0]) for K in K_VALUES_TCN},
        "ridge_all_192_on_pool": ridge_all_192,
        "tcn_per_K": {},
        "ridge_per_K": {},
    }
    t_start = time.time()
    for K in K_VALUES_TCN:
        tcn_block: Dict[str, Dict[str, float]] = {}
        ridge_block: Dict[str, Dict[str, float]] = {}
        for sel_name in TCN_SELECTORS:
            sensors = selectors_by_K[K][sel_name]
            t_tcn = time.time()
            tcn_metrics = _evaluate_tcn(
                sensors, pool_X, pool_Y_z, pool_Y_CL, pool_Y_phase,
                epochs=epochs, device=device,
            )
            tcn_block[sel_name] = tcn_metrics
            ridge_metrics = _evaluate_ridge_pool(
                sensors, pool_X, pool_Y_z, pool_Y_CL, pool_Y_phase
            )
            ridge_block[sel_name] = ridge_metrics
            print(
                f"[item1]   K={K:1d} {sel_name:18s} "
                f"TCN z_R2={tcn_metrics['z_R2']:+.3f} CL_R2={tcn_metrics['CL_R2']:+.3f} "
                f"phase_RMSE={tcn_metrics['phase_RMSE']:.2f} | "
                f"Ridge z_R2={ridge_metrics['z_R2']:+.3f} CL_R2={ridge_metrics['CL_R2']:+.3f} "
                f"phase_RMSE={ridge_metrics['phase_RMSE']:.2f} "
                f"({time.time() - t_tcn:.1f}s)",
                flush=True,
            )
        results["tcn_per_K"][f"K_{K}"] = tcn_block
        results["ridge_per_K"][f"K_{K}"] = ridge_block
    results["wall_time_seconds"] = float(time.time() - t_start)

    out_path = out_dir / "tcn_followup_K234.json"
    with open(out_path, "w") as f:
        json.dump(_tolist(results), f, indent=2)
    print(f"[item1] wrote {out_path}", flush=True)

    fig_path = out_dir / "tcn_vs_ridge.png"
    _render_tcn_vs_ridge_figure(results, fig_path)
    print(f"[item1] wrote {fig_path}", flush=True)
    return results


def _render_tcn_vs_ridge_figure(
    results: Dict[str, object], fig_path: Path
) -> None:
    """Grouped-bar figure: TCSI/qDEIM x Ridge/TCN x K=2/3/4 on the z_R2 metric."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    K_list = results["K_values"]
    n_k = len(K_list)
    x_pos = np.arange(n_k)
    bar_w = 0.18
    # Four bars per K: TCSI Ridge, TCSI TCN, qDEIM Ridge, qDEIM TCN.
    series_specs = [
        ("TCSI Ridge", "ridge_per_K", "TCSI", "tab:red", "//"),
        ("TCSI TCN", "tcn_per_K", "TCSI", "tab:red", None),
        ("qDEIM Ridge", "ridge_per_K", "qDEIM", "tab:orange", "//"),
        ("qDEIM TCN", "tcn_per_K", "qDEIM", "tab:orange", None),
    ]
    n_ser = len(series_specs)
    for s_idx, (label, table, sel, color, hatch) in enumerate(series_specs):
        y = []
        for K in K_list:
            entry = results[table].get(f"K_{K}", {}).get(sel, {})
            y.append(entry.get("z_R2", float("nan")))
        ax.bar(
            x_pos + (s_idx - (n_ser - 1) / 2) * bar_w,
            y,
            width=bar_w,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            hatch=hatch,
        )
    all192 = results["ridge_all_192_on_pool"]["z_R2"]
    ax.axhline(all192, color="black", linestyle=":", linewidth=0.8,
               label=f"all_192 Ridge (pool CV) = {all192:.3f}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"K={K}" for K in K_list])
    ax.set_ylabel(r"$z\;R^2$ (5-fold CV on selection pool)")
    ax.set_title("Item 1: TCN vs Ridge for TCSI and qDEIM at small K")
    ax.set_ylim(-0.05, 1.0)
    ax.axhline(0, color="black", linewidth=0.4)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Item 2: Bootstrap stability of greedy selection
# ---------------------------------------------------------------------------


def run_item2_bootstrap_stability(
    pool_X: np.ndarray,
    pool_Y_z_pc1: np.ndarray,
    out_dir: Path,
    n_seeds: int = BOOT_SEEDS,
) -> Dict[str, object]:
    """Bootstrap-resample the selection pool and re-run greedy K = 2 / 3 / 4."""
    print(f"[item2] bootstrap stability: {n_seeds} seeds at K in {K_VALUES_TCN}",
          flush=True)
    n_pool = pool_X.shape[0]
    selections: Dict[int, List[List[int]]] = {K: [] for K in K_VALUES_TCN}
    t0 = time.time()
    for s in range(n_seeds):
        rng = np.random.default_rng(s)
        idx = rng.integers(0, n_pool, size=n_pool)
        Xb = pool_X[idx]
        yb = pool_Y_z_pc1[idx]
        # Greedy at the largest K (4) is a prefix of greedy at smaller K
        # for the same target, so compute once and slice.
        chain = greedy_forward_selection(Xb, yb, K=max(K_VALUES_TCN))
        for K in K_VALUES_TCN:
            selections[K].append(list(chain[:K]))
        if (s + 1) % 10 == 0 or s + 1 == n_seeds:
            print(f"[item2]  {s + 1}/{n_seeds} seeds ({time.time() - t0:.1f}s)",
                  flush=True)

    # Reference greedy on the full (non-bootstrapped) pool for the figure overlay.
    ref_chain_K4 = greedy_forward_selection(
        pool_X, pool_Y_z_pc1, K=max(K_VALUES_TCN)
    )

    freq_per_K: Dict[int, Dict[int, float]] = {}
    top10_per_K: Dict[int, List[Tuple[int, float]]] = {}
    for K in K_VALUES_TCN:
        counts: Counter = Counter()
        for sel in selections[K]:
            for j in sel:
                counts[int(j)] += 1
        freq_per_K[K] = {
            int(j): float(c / n_seeds) for j, c in counts.items()
        }
        top10 = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:10]
        top10_per_K[K] = [(int(j), float(c / n_seeds)) for j, c in top10]
        print(
            f"[item2]  K={K} top10 (sensor, freq): "
            f"{[(j, f'{f:.2f}') for j, f in top10_per_K[K]]}",
            flush=True,
        )

    # Stability score per sensor at K = 4 (the user's headline question:
    # "do sensors 11/20 survive bootstrapping").
    stability_K4 = {
        int(j): float(freq_per_K[max(K_VALUES_TCN)].get(int(j), 0.0))
        for j in range(N_SENSORS)
    }

    results: Dict[str, object] = {
        "n_seeds": int(n_seeds),
        "n_pool": int(n_pool),
        "K_values": list(K_VALUES_TCN),
        "selections_per_K": {
            f"K_{K}": [[int(j) for j in sel] for sel in selections[K]]
            for K in K_VALUES_TCN
        },
        "frequency_per_K": {
            f"K_{K}": freq_per_K[K] for K in K_VALUES_TCN
        },
        "top10_per_K": {
            f"K_{K}": [list(t) for t in top10_per_K[K]] for K in K_VALUES_TCN
        },
        "stability_score_K4": stability_K4,
        "reference_chain_K4_full_pool": [int(j) for j in ref_chain_K4],
    }
    # Sensors-of-interest summary
    for j in (11, 20, 44, 5):
        results[f"sensor_{j}_freq_at_K4"] = float(freq_per_K[4].get(j, 0.0))
        results[f"sensor_{j}_freq_at_K2"] = float(freq_per_K[2].get(j, 0.0))
    print(
        f"[item2] sensor 11 freq K=2/K=4: "
        f"{results['sensor_11_freq_at_K2']:.2f} / {results['sensor_11_freq_at_K4']:.2f}",
        flush=True,
    )
    print(
        f"[item2] sensor 20 freq K=2/K=4: "
        f"{results['sensor_20_freq_at_K2']:.2f} / {results['sensor_20_freq_at_K4']:.2f}",
        flush=True,
    )

    out_path = out_dir / "bootstrap_stability_K234.json"
    with open(out_path, "w") as f:
        json.dump(_tolist(results), f, indent=2)
    print(f"[item2] wrote {out_path}", flush=True)

    fig_path = out_dir / "bootstrap_stability.png"
    _render_bootstrap_figure(
        selections, freq_per_K, ref_chain_K4, fig_path, n_seeds=n_seeds,
    )
    print(f"[item2] wrote {fig_path}", flush=True)
    return results


def _render_bootstrap_figure(
    selections: Dict[int, List[List[int]]],
    freq_per_K: Dict[int, Dict[int, float]],
    ref_chain_K4: Sequence[int],
    fig_path: Path,
    n_seeds: int,
) -> None:
    """Heatmap of sensor index vs bootstrap iteration at K = 4, with a frequency strip."""
    K_main = max(K_VALUES_TCN)
    H = np.zeros((n_seeds, N_SENSORS), dtype=np.float32)
    for i, sel in enumerate(selections[K_main]):
        for j in sel:
            H[i, int(j)] = 1.0

    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_h = fig.add_subplot(gs[0])
    ax_h.imshow(H, aspect="auto", cmap="Reds", interpolation="nearest")
    for j in ref_chain_K4:
        ax_h.axvline(int(j), color="tab:blue", linestyle="--", linewidth=0.8,
                     alpha=0.7)
    ax_h.set_ylabel(f"bootstrap iter (n={n_seeds})")
    ax_h.set_title(
        f"Item 2: bootstrap-resampled greedy K={K_main} sensor selections "
        "(red = picked; blue dashed = full-pool reference)"
    )
    ax_h.set_xticks([])

    # Frequency bar strip (top-10 per K stacked colours)
    ax_b = fig.add_subplot(gs[1], sharex=ax_h)
    K_colors = {2: "tab:orange", 3: "tab:green", 4: "tab:red"}
    for K, color in K_colors.items():
        x = np.arange(N_SENSORS)
        y = np.array([freq_per_K[K].get(j, 0.0) for j in x])
        ax_b.bar(x, y, width=1.0, color=color, alpha=0.5, edgecolor=None,
                 label=f"K={K}")
    ax_b.set_xlabel("sensor index (0..191; LE near 0)")
    ax_b.set_ylabel("freq.")
    ax_b.set_xlim(-0.5, N_SENSORS - 0.5)
    ax_b.legend(loc="upper right", fontsize=8)
    ax_b.grid(axis="y", alpha=0.3)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Item 3: Per-(G, D, Y) regime stability
# ---------------------------------------------------------------------------


def _build_regimes(
    G: np.ndarray, D: np.ndarray, Y: np.ndarray, source_groups: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Return boolean index masks over the selection pool per regime."""
    return {
        "high_abs_G_ge_1p5": np.abs(G) >= 1.5,
        "low_abs_G_le_0p5": np.abs(G) <= 0.5,
        "run3_only": source_groups == "run3",
        "periodic_only": source_groups == "periodic",
        "D_eq_1p0": np.isclose(D, 1.0),
        "D_le_0p5": D <= 0.5,
    }


def _resolve_source_groups(case_ids: Iterable[str]) -> np.ndarray:
    """Look up source_group from the split manifest for each case_id."""
    split_path = REPO_ROOT / "configs" / "splits" / "split_v1.json"
    with open(split_path) as f:
        manifest = json.load(f)
    cases = manifest["cases"]
    out = np.array(
        [cases.get(str(cid), {}).get("source_group", "unknown") for cid in case_ids],
        dtype=object,
    )
    return out


def run_item3_regime_stability(
    pool_X: np.ndarray,
    pool_Y_z_pc1: np.ndarray,
    pool_G: np.ndarray,
    pool_D: np.ndarray,
    pool_Y_coord: np.ndarray,
    pool_case_ids: np.ndarray,
    out_dir: Path,
) -> Dict[str, object]:
    """Run greedy K = 2 and K = 4 on six (G, D, Y) regime subsets."""
    print("[item3] per-regime greedy K = 2 and K = 4 on selection pool",
          flush=True)
    source_groups = _resolve_source_groups(pool_case_ids)
    regimes = _build_regimes(pool_G, pool_D, pool_Y_coord, source_groups)

    le_cluster = {11, 20}
    per_regime: Dict[str, Dict[str, object]] = {}
    for name, mask in regimes.items():
        n_sel = int(mask.sum())
        if n_sel < 8:
            per_regime[name] = {
                "n": int(n_sel),
                "K2_sensors": [],
                "K4_sensors": [],
                "le_cluster_picked_K4": False,
                "skipped_reason": "fewer than 8 encounters in this regime",
            }
            print(f"[item3]  {name}: n={n_sel} -> SKIPPED", flush=True)
            continue
        Xr = pool_X[mask]
        yr = pool_Y_z_pc1[mask]
        chain = greedy_forward_selection(Xr, yr, K=4)
        K2 = chain[:2]
        K4 = chain[:4]
        le_picked = bool(le_cluster.issubset(set(int(j) for j in K4)))
        per_regime[name] = {
            "n": int(n_sel),
            "K2_sensors": [int(j) for j in K2],
            "K4_sensors": [int(j) for j in K4],
            "le_cluster_picked_K4": le_picked,
        }
        print(
            f"[item3]  {name:24s} n={n_sel:3d} K2={K2} K4={K4} "
            f"LE_cluster_K4={le_picked}",
            flush=True,
        )

    n_regimes_with_le = sum(
        1 for v in per_regime.values()
        if isinstance(v.get("le_cluster_picked_K4"), bool) and v["le_cluster_picked_K4"]
    )
    n_regimes_total = sum(
        1 for v in per_regime.values() if "le_cluster_picked_K4" in v
        and "skipped_reason" not in v
    )
    results = {
        "regimes": list(regimes.keys()),
        "per_regime": per_regime,
        "le_cluster_definition": "sensors {11, 20}",
        "n_regimes_picking_le_cluster_at_K4": int(n_regimes_with_le),
        "n_regimes_evaluated": int(n_regimes_total),
    }
    print(
        f"[item3] LE-cluster (sensors 11 and 20) appears in K=4 selection "
        f"in {n_regimes_with_le}/{n_regimes_total} regimes",
        flush=True,
    )

    out_path = out_dir / "regime_stability_K2K4.json"
    with open(out_path, "w") as f:
        json.dump(_tolist(results), f, indent=2)
    print(f"[item3] wrote {out_path}", flush=True)

    fig_path = out_dir / "regime_stability.png"
    _render_regime_figure(per_regime, fig_path)
    print(f"[item3] wrote {fig_path}", flush=True)
    return results


def _render_regime_figure(
    per_regime: Dict[str, Dict[str, object]], fig_path: Path
) -> None:
    """Airfoil polygon with per-regime sensor markers."""
    airfoil, sensor_xy = load_airfoil_sensor_positions()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, K_label in zip(axes, ("K2_sensors", "K4_sensors")):
        poly = Polygon(airfoil, closed=True, facecolor="0.85", edgecolor="black",
                       linewidth=0.7, zorder=1)
        ax.add_patch(poly)
        ax.scatter(sensor_xy[:, 0], sensor_xy[:, 1], s=3, color="0.5",
                   alpha=0.35, zorder=2)
        cmap = plt.get_cmap("tab10")
        markers = ["o", "s", "^", "D", "v", "P"]
        for idx, (name, info) in enumerate(per_regime.items()):
            sensors = info.get(K_label, [])
            if not sensors:
                continue
            color = cmap(idx % 10)
            marker = markers[idx % len(markers)]
            ax.scatter(
                sensor_xy[sensors, 0], sensor_xy[sensors, 1],
                s=65, marker=marker, color=color, edgecolor="black",
                linewidth=0.4, zorder=3 + idx, label=f"{name} (n={info['n']})",
            )
        ax.set_aspect("equal")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.10, 0.10)
        ax.set_xlabel("x / c")
        ax.set_title(f"Per-regime greedy selection: {K_label.replace('_sensors', '')}")
    axes[0].set_ylabel("y / c")
    axes[1].legend(loc="upper right", fontsize=7, ncol=1)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Item 4: K = 2 decoded flow-field figure
# ---------------------------------------------------------------------------


def _omega_to_pixel(xy: np.ndarray, H: int = 192, W: int = 96) -> np.ndarray:
    x_min, x_max = -1.5, 4.5
    y_min, y_max = -1.5, 1.5
    px_x = (xy[:, 0] - x_min) * (H - 1) / (x_max - x_min)
    px_y = (xy[:, 1] - y_min) * (W - 1) / (y_max - y_min)
    return np.stack([px_x, px_y], axis=-1)


def _load_airfoil_polygon_for_omega() -> np.ndarray:
    """Closed airfoil polygon in pixel coords for the omega image axes."""
    prevent = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
    with h5py.File(prevent / "data" / "raw" / "periodic" / "Baseline.h5", "r") as f:
        xy = np.asarray(f["airfoil_xy"], dtype=np.float64)
    if not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[0:1]])
    return _omega_to_pixel(xy)


def run_item4_decoded_flow_field(
    data: Dict[str, Dict[str, np.ndarray]],
    out_dir: Path,
    selected_K2_sensors: Sequence[int],
    selected_K192_sensors: Optional[Sequence[int]] = None,
    gpu_index: int = 0,
) -> Dict[str, object]:
    """Decode the K=2 Ridge-predicted z into omega_hat and compare to DNS / K=192.

    Pipeline:
      1. Use train + test_a as the supervised pool: fit
         ``Ridge(window -> z_impact_64d)`` on the K=2 sensors and on all 192.
      2. For each chosen Test B encounter:
         (a) extract the K=2 / K=192 windows;
         (b) predict z_hat (64-d);
         (c) decode z_hat through the SL decoder on the impact frame;
         (d) save dns / k2 / k192 omega panels with the standard +/-3 colorbar.
    """
    print("[item4] decoded flow-field reconstruction at K=2", flush=True)

    if selected_K192_sensors is None:
        selected_K192_sensors = list(range(N_SENSORS))

    # Build supervised pool (train + test_a).
    X_pool = np.concatenate(
        [data["train"]["X_window"], data["test_a"]["X_window"]], axis=0
    )
    Z_pool = np.concatenate(
        [data["train"]["Y_z"], data["test_a"]["Y_z"]], axis=0
    )
    print(f"[item4] supervised pool n={X_pool.shape[0]} z_dim={Z_pool.shape[1]}",
          flush=True)

    # Fit two ridge heads (pressure window -> z_impact 64-d).
    feats_K2_pool = _subset_features(X_pool, selected_K2_sensors)
    feats_full_pool = _subset_features(X_pool, selected_K192_sensors)
    ridge_K2 = Ridge(alpha=1.0).fit(feats_K2_pool, Z_pool)
    ridge_full = Ridge(alpha=1.0).fit(feats_full_pool, Z_pool)

    # Load decoder + airfoil mask + omega pipeline + encoder for unnormalising.
    import torch  # local import keeps the script import-time cheap when items 1-3
                  # run on CPU.

    from src.data.omega_pipeline import OmegaPipeline
    from src.models.encoder import HybridCNNViTEncoder
    from src.models.lap_film_decoder import LapFiLMDecoder
    from src.utils.device import NoRTX6000Error, require_rtx6000

    try:
        device = require_rtx6000(gpu_index=gpu_index)
    except NoRTX6000Error as err0:
        print(f"[item4] gpu_index={gpu_index} unavailable ({err0}); trying the other card",
              flush=True)
        device = require_rtx6000(gpu_index=1 - gpu_index)
    print(f"[item4] device={device} ({torch.cuda.get_device_name(device.index)})",
          flush=True)

    # Decoder. Session 9 trainer stored args.d as None because the latent_dim
    # comes from the loaded encoder rather than from the decoder CLI; fall back
    # to Z_pool.shape[1] (encoder latent dimensionality of the training pool).
    blob = torch.load(DECODER_CKPT, map_location="cpu", weights_only=False)
    saved_args = blob.get("args", {})
    bc = int(saved_args.get("decoder_base_ch") or 64)
    channels = (bc, bc, int(bc * 0.75), int(bc * 0.5), int(bc * 0.375))
    raw_d = saved_args.get("d")
    latent_dim = int(raw_d) if raw_d is not None else int(Z_pool.shape[1])
    raw_fb = saved_args.get("decoder_fourier_bands")
    fourier_bands = int(raw_fb) if raw_fb is not None else 4
    dec = LapFiLMDecoder(
        latent_dim=latent_dim,
        channels=channels,
        resblocks_per_level=int(saved_args.get("decoder_resblocks_per_level") or 2),
        upsample=saved_args.get("decoder_upsample") or "pixelshuffle",
        fourier_bands=fourier_bands,
        use_film=bool(saved_args.get("decoder_use_film", True)),
    )
    dec.load_state_dict(blob["decoder_state_dict"])
    dec.eval().to(device)
    print(f"[item4] decoder loaded from {DECODER_CKPT.name} "
          f"(d={dec.latent_dim}, channels={channels})", flush=True)

    pipe = OmegaPipeline.from_manifest(OMEGA_PIPELINE_MANIFEST)

    airfoil_px = _load_airfoil_polygon_for_omega()
    out_arr_dir = out_dir / "k2_decoded"
    out_arr_dir.mkdir(parents=True, exist_ok=True)

    test_b = data["test_b"]
    fig, axes = plt.subplots(3, len(ITEM4_CASES) + 1, figsize=(11, 8),
                             gridspec_kw={"width_ratios":
                                          [1] * len(ITEM4_CASES) + [0.06]})

    case_records: List[Dict[str, object]] = []
    impact_frame_default = 40

    for col, (case_id, enc_k) in enumerate(ITEM4_CASES):
        # Locate this encounter in test_b.
        mask = (
            (np.asarray(test_b["case_id"]) == case_id)
            & (test_b["encounter_index"] == enc_k)
        )
        if not mask.any():
            print(f"[item4]  WARNING: {case_id} enc {enc_k:02d} not in test_b; "
                  "trying test_a", flush=True)
            test_a = data["test_a"]
            mask = (
                (np.asarray(test_a["case_id"]) == case_id)
                & (test_a["encounter_index"] == enc_k)
            )
            if not mask.any():
                msg = f"missing {case_id} enc {enc_k:02d} in test_b or test_a; skipping"
                print(f"[item4]  {msg}", flush=True)
                case_records.append({"case_id": case_id, "encounter_index": enc_k,
                                     "status": "missing"})
                continue
            source_split_name = "test_a"
            source_split = test_a
        else:
            source_split_name = "test_b"
            source_split = test_b

        idx = int(np.argmax(mask))
        impact_frame = int(source_split["Y_phase"][idx])  # already an int from build_data
        # CLAUDE.md: impact_frame_estimate ~40. Honour the per-encounter value if reasonable.
        if not (10 <= impact_frame <= 110):
            impact_frame = impact_frame_default

        X_win = source_split["X_window"][idx]  # (192, 17)
        feats_K2 = X_win[list(selected_K2_sensors), :].reshape(1, -1)
        feats_full = X_win[list(selected_K192_sensors), :].reshape(1, -1)
        z_hat_K2 = ridge_K2.predict(feats_K2).reshape(-1)
        z_hat_full = ridge_full.predict(feats_full).reshape(-1)

        # Load DNS omega at impact frame (raw cache).
        from scripts.session14_tcsi_pilot import resolve_cache_root
        cache_root = resolve_cache_root()
        h5_path = cache_root / case_id / f"encounter_{int(enc_k):02d}.h5"
        with h5py.File(h5_path, "r") as f:
            omega_full = np.asarray(f["omega_z"], dtype=np.float32)
            stored_impact = int(f.attrs.get("impact_frame_estimate", impact_frame_default))
        # Use the cached attr in preference to the phase-derived one for decoder eval.
        impact_frame = stored_impact if 10 <= stored_impact <= 110 else impact_frame
        omega_at_impact_raw = omega_full[impact_frame]
        # The decoder operates in normalised space, so we put DNS through the same
        # pipeline for a fair visual comparison.
        omega_norm = pipe.preprocess_raw(omega_full, case_id, int(enc_k))
        omega_norm_t = torch.from_numpy(omega_norm)
        omega_norm_t = pipe.normalize(omega_norm_t)
        dns_norm_impact = omega_norm_t[impact_frame].numpy()

        # Decode the two ridge-predicted latents.
        with torch.no_grad():
            for z_hat, key in [(z_hat_K2, "k2"), (z_hat_full, "k192")]:
                z_t = torch.from_numpy(z_hat).to(device, dtype=torch.float32)
                z_t = z_t.unsqueeze(0)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    dec_out = dec(z_t)
                omega_hat = dec_out["pred"].float().squeeze().cpu().numpy()
                if key == "k2":
                    omega_k2 = omega_hat
                else:
                    omega_k192 = omega_hat

        # Save numpy arrays for paper-figure reuse (DNS-normalised, k2, k192).
        out_npy_dns = out_arr_dir / f"{case_id}_enc{int(enc_k):02d}_dns.npy"
        out_npy_k2 = out_arr_dir / f"{case_id}_enc{int(enc_k):02d}_k2.npy"
        out_npy_k192 = out_arr_dir / f"{case_id}_enc{int(enc_k):02d}_k192.npy"
        np.save(out_npy_dns, dns_norm_impact)
        np.save(out_npy_k2, omega_k2)
        np.save(out_npy_k192, omega_k192)

        # Quality metrics (in normalised space; the decoder's native scale).
        ssr_k2 = float(np.mean((dns_norm_impact - omega_k2) ** 2))
        ssr_k192 = float(np.mean((dns_norm_impact - omega_k192) ** 2))
        cos_k2 = float(np.sum(dns_norm_impact * omega_k2) /
                       (np.linalg.norm(dns_norm_impact) * np.linalg.norm(omega_k2) + 1e-9))
        cos_k192 = float(np.sum(dns_norm_impact * omega_k192) /
                         (np.linalg.norm(dns_norm_impact) * np.linalg.norm(omega_k192) + 1e-9))
        print(
            f"[item4]  {case_id} enc {enc_k:02d} (source={source_split_name}, "
            f"impact={impact_frame})  "
            f"K=2  MSE={ssr_k2:.3f}  cos={cos_k2:.3f}  |  "
            f"K=192 MSE={ssr_k192:.3f}  cos={cos_k192:.3f}",
            flush=True,
        )

        case_records.append({
            "case_id": case_id,
            "encounter_index": enc_k,
            "source_split": source_split_name,
            "impact_frame": int(impact_frame),
            "K2_sensors": [int(s) for s in selected_K2_sensors],
            "K192_sensor_count": int(len(selected_K192_sensors)),
            "K2_decode_mse_norm": ssr_k2,
            "K192_decode_mse_norm": ssr_k192,
            "K2_decode_cosine": cos_k2,
            "K192_decode_cosine": cos_k192,
            "npy_dns": str(out_npy_dns),
            "npy_k2": str(out_npy_k2),
            "npy_k192": str(out_npy_k192),
        })

        # Render the three rows.
        vlim = 3.0
        row_specs = [("DNS (normalised)", dns_norm_impact),
                     (f"K=2 reconstruction (sensors {list(selected_K2_sensors)})",
                      omega_k2),
                     ("K=192 reconstruction (Ridge)", omega_k192)]
        for row_idx, (label, arr) in enumerate(row_specs):
            ax = axes[row_idx, col]
            im = ax.imshow(arr.T, origin="lower", cmap="RdBu_r",
                           vmin=-vlim, vmax=vlim)
            ax.add_patch(
                Polygon(airfoil_px, closed=True, facecolor="black",
                        edgecolor="black", linewidth=0.7, zorder=10)
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(
                    f"{case_id}\nenc {enc_k:02d} (impact frame {impact_frame})",
                    fontsize=9,
                )
            if col == 0:
                ax.set_ylabel(label, fontsize=9)

    # Single shared colour bar in the last column (row 0 spans for the full height).
    cax = axes[0, -1]
    fig.colorbar(im, cax=cax, extend="both").set_label(r"$\omega_z$ (norm.)", fontsize=9)
    for r in (1, 2):
        axes[r, -1].axis("off")

    fig.suptitle(
        "Item 4: K=2 sensor -> Ridge -> SL decoder reconstruction on two Test B "
        "encounters\n"
        "Standard fixed-colorbar +/-3 normalised omega panels. "
        "Airfoil overlaid as solid polygon.",
        y=1.02, fontsize=10,
    )
    fig.tight_layout()
    fig_path = out_dir / "k2_decoded_flow_field.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[item4] wrote {fig_path}", flush=True)

    results = {
        "K2_sensors": [int(s) for s in selected_K2_sensors],
        "K192_sensor_count": int(len(selected_K192_sensors)),
        "decoder_checkpoint": str(DECODER_CKPT),
        "encoder_checkpoint": str(ENCODER_CKPT),
        "cases": case_records,
        "figure_path": str(fig_path),
    }
    out_path = out_dir / "k2_decoded_flow_field.json"
    with open(out_path, "w") as f:
        json.dump(_tolist(results), f, indent=2)
    print(f"[item4] wrote {out_path}", flush=True)
    return results


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--items", nargs="+", type=int, default=[1, 2, 3, 4],
        choices=[1, 2, 3, 4], help="Which follow-up items to run.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=PILOT_OUT_DIR,
        help="Output directory for JSON / figures.",
    )
    p.add_argument(
        "--tcn-epochs", type=int, default=TCN_DEFAULT_EPOCHS,
        help="Epochs per TCN fit (default 200; scale down to 100 if needed).",
    )
    p.add_argument(
        "--tcn-device", type=str, default="cpu",
        help="Torch device for TCN training (default cpu; pass cuda:0 for speedup).",
    )
    p.add_argument(
        "--bootstrap-seeds", type=int, default=BOOT_SEEDS,
        help="Bootstrap seed count for item 2.",
    )
    p.add_argument(
        "--gpu", type=int, default=0,
        help="0-indexed RTX 6000 selector for item 4.",
    )
    return p.parse_args()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_global = time.time()

    print(f"[driver] building data arrays (W={WINDOW}, n_sensors={N_SENSORS})",
          flush=True)
    data = build_data_arrays()
    pool_X = np.concatenate(
        [data["train"]["X_window"], data["test_a"]["X_window"]], axis=0
    )
    pool_Y_z = np.concatenate(
        [data["train"]["Y_z"], data["test_a"]["Y_z"]], axis=0
    )
    pool_Y_CL = np.concatenate(
        [data["train"]["Y_CL_imp"], data["test_a"]["Y_CL_imp"]], axis=0
    )
    pool_Y_phase = np.concatenate(
        [data["train"]["Y_phase"], data["test_a"]["Y_phase"]], axis=0
    ).astype(np.float64)
    pool_case_ids = np.concatenate(
        [data["train"]["case_id"], data["test_a"]["case_id"]], axis=0
    )
    # Per-(G, D, Y) lookup from the latent NPZ (faster than re-parsing case_id).
    train_npz = np.load(LATENT_DIR / "train.npz", allow_pickle=True)
    testa_npz = np.load(LATENT_DIR / "test_a.npz", allow_pickle=True)
    pool_G = np.concatenate([train_npz["G"], testa_npz["G"]], axis=0)
    pool_D = np.concatenate([train_npz["D"], testa_npz["D"]], axis=0)
    pool_Y_coord = np.concatenate([train_npz["Y"], testa_npz["Y"]], axis=0)
    train_npz.close()
    testa_npz.close()
    # NaN drop alignment (build_data_arrays filters NaN encounters; we must
    # apply the same filter to the GDY arrays). The two arrays may already
    # be exactly aligned because build_data_arrays preserves order, but to
    # be safe we re-build pool_G/pool_D/pool_Y_coord from the NPZ + filter.
    if pool_G.shape[0] != pool_X.shape[0]:
        # Align by (case_id, encounter_index).
        train_npz = np.load(LATENT_DIR / "train.npz", allow_pickle=True)
        testa_npz = np.load(LATENT_DIR / "test_a.npz", allow_pickle=True)
        all_case = np.concatenate([train_npz["case_id"], testa_npz["case_id"]])
        all_enc = np.concatenate([train_npz["encounter_index"], testa_npz["encounter_index"]])
        all_G = np.concatenate([train_npz["G"], testa_npz["G"]])
        all_D = np.concatenate([train_npz["D"], testa_npz["D"]])
        all_Y = np.concatenate([train_npz["Y"], testa_npz["Y"]])
        key_to_GDY = {
            (str(c), int(e)): (float(g), float(d), float(y))
            for c, e, g, d, y in zip(all_case, all_enc, all_G, all_D, all_Y)
        }
        pool_enc = np.concatenate(
            [data["train"]["encounter_index"], data["test_a"]["encounter_index"]]
        )
        pool_G = np.array(
            [key_to_GDY[(str(c), int(e))][0]
             for c, e in zip(pool_case_ids, pool_enc)]
        )
        pool_D = np.array(
            [key_to_GDY[(str(c), int(e))][1]
             for c, e in zip(pool_case_ids, pool_enc)]
        )
        pool_Y_coord = np.array(
            [key_to_GDY[(str(c), int(e))][2]
             for c, e in zip(pool_case_ids, pool_enc)]
        )
        train_npz.close()
        testa_npz.close()
    pool_Y_z_pc1 = PCA(n_components=1, random_state=0).fit_transform(
        pool_Y_z
    ).reshape(-1)
    print(f"[driver] pool n={pool_X.shape[0]}  z_dim={pool_Y_z.shape[1]}",
          flush=True)

    # Greedy selections at K = 2 / 3 / 4 (reproducible on the unstratified pool)
    greedy_selections: Dict[int, List[int]] = {}
    chain: List[int] = []
    for K in K_VALUES_TCN:
        chain = greedy_forward_selection(
            pool_X, pool_Y_z_pc1, K=K, initial=chain,
        )
        greedy_selections[K] = list(chain)
    print(f"[driver] greedy K=2..4 from pool: {greedy_selections}",
          flush=True)

    # Merge with any previous followups_summary.json so a partial re-run
    # (--items 1 only, then --items 4 only) preserves earlier item results.
    summary_path = out_dir / "followups_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary: Dict[str, object] = json.load(f)
        except (OSError, json.JSONDecodeError):
            summary = {}
    else:
        summary = {}
    summary.update({
        "selection_pool_n": int(pool_X.shape[0]),
        "greedy_selections_pool_K234": {
            f"K_{K}": [int(s) for s in greedy_selections[K]] for K in K_VALUES_TCN
        },
        "items_last_run": list(args.items),
    })
    if "items_history" not in summary or not isinstance(summary.get("items_history"), list):
        summary["items_history"] = []
    summary["items_history"].append({
        "items": list(args.items),
        "wall_time_seconds": None,  # filled in below
    })

    if 1 in args.items:
        item1 = run_item1_tcn_followup(
            pool_X=pool_X, pool_Y_z=pool_Y_z, pool_Y_CL=pool_Y_CL,
            pool_Y_phase=pool_Y_phase,
            greedy_selections=greedy_selections,
            epochs=args.tcn_epochs, device=args.tcn_device,
            out_dir=out_dir,
        )
        summary["item1_tcn_followup"] = {
            "wall_time_seconds": item1["wall_time_seconds"],
            "epochs": item1["epochs"],
            "device": item1["device"],
            "tcn_per_K_z_R2_TCSI": {
                f"K_{K}": item1["tcn_per_K"][f"K_{K}"]["TCSI"]["z_R2"]
                for K in K_VALUES_TCN
            },
            "tcn_per_K_z_R2_qDEIM": {
                f"K_{K}": item1["tcn_per_K"][f"K_{K}"]["qDEIM"]["z_R2"]
                for K in K_VALUES_TCN
            },
        }

    if 2 in args.items:
        item2 = run_item2_bootstrap_stability(
            pool_X=pool_X, pool_Y_z_pc1=pool_Y_z_pc1,
            out_dir=out_dir, n_seeds=args.bootstrap_seeds,
        )
        summary["item2_bootstrap"] = {
            "sensor_11_freq_K2": item2["sensor_11_freq_at_K2"],
            "sensor_11_freq_K4": item2["sensor_11_freq_at_K4"],
            "sensor_20_freq_K2": item2["sensor_20_freq_at_K2"],
            "sensor_20_freq_K4": item2["sensor_20_freq_at_K4"],
            "sensor_44_freq_K4": item2["sensor_44_freq_at_K4"],
            "sensor_5_freq_K4": item2["sensor_5_freq_at_K4"],
        }

    if 3 in args.items:
        item3 = run_item3_regime_stability(
            pool_X=pool_X, pool_Y_z_pc1=pool_Y_z_pc1,
            pool_G=pool_G, pool_D=pool_D, pool_Y_coord=pool_Y_coord,
            pool_case_ids=pool_case_ids,
            out_dir=out_dir,
        )
        summary["item3_regime"] = {
            "n_regimes_picking_le_cluster_K4": item3["n_regimes_picking_le_cluster_at_K4"],
            "n_regimes_evaluated": item3["n_regimes_evaluated"],
        }

    if 4 in args.items:
        try:
            item4 = run_item4_decoded_flow_field(
                data=data, out_dir=out_dir,
                selected_K2_sensors=greedy_selections[2],
                gpu_index=args.gpu,
            )
            summary["item4_decoded"] = {
                "figure_path": item4["figure_path"],
                "n_cases": len(item4["cases"]),
            }
        except Exception as err:  # noqa: BLE001
            print(f"[item4] FAILED: {err}", flush=True)
            summary["item4_decoded"] = {"status": "failed", "error": repr(err)}

    summary["wall_time_seconds_total"] = float(time.time() - t_global)
    if summary["items_history"]:
        summary["items_history"][-1]["wall_time_seconds"] = float(time.time() - t_global)
    with open(summary_path, "w") as f:
        json.dump(_tolist(summary), f, indent=2)
    print(f"[driver] summary -> {summary_path}", flush=True)
    print(f"[driver] total wall time: {time.time() - t_global:.1f}s", flush=True)

    # ---- final report ----
    print("\n[REPORT] ============================================", flush=True)
    if 1 in args.items:
        print("[REPORT] Item 1 (TCN vs Ridge) z_R2 on selection-pool CV:", flush=True)
        for K in K_VALUES_TCN:
            t = summary["item1_tcn_followup"]["tcn_per_K_z_R2_TCSI"][f"K_{K}"]
            q = summary["item1_tcn_followup"]["tcn_per_K_z_R2_qDEIM"][f"K_{K}"]
            delta = t - q
            print(
                f"[REPORT]   K={K} TCSI TCN z_R2={t:+.3f} | qDEIM TCN z_R2={q:+.3f} "
                f"| TCSI - qDEIM = {delta:+.3f}",
                flush=True,
            )
    if 2 in args.items:
        b = summary["item2_bootstrap"]
        print(
            f"[REPORT] Item 2 (bootstrap K=4 freq): "
            f"sensor 11={b['sensor_11_freq_K4']:.2f}  20={b['sensor_20_freq_K4']:.2f}  "
            f"44={b['sensor_44_freq_K4']:.2f}  5={b['sensor_5_freq_K4']:.2f}",
            flush=True,
        )
    if 3 in args.items:
        r = summary["item3_regime"]
        print(
            f"[REPORT] Item 3 (regime stability): LE cluster picked at K=4 in "
            f"{r['n_regimes_picking_le_cluster_K4']}/{r['n_regimes_evaluated']} regimes",
            flush=True,
        )
    if 4 in args.items:
        d = summary.get("item4_decoded", {})
        print(f"[REPORT] Item 4 (decoded field): {d}", flush=True)
    print("[REPORT] ============================================\n", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
