#!/usr/bin/env python
"""Regenerate the three v2.1 Section 4.6 quantities the manuscript still quotes from v2.

The manuscript Section 4.6 still cites three numbers measured on the OLD v2 /
session17 analyses. Phase D forbids hand-typed numbers, so this script remeasures
all three on v2.1 and emits an eval_all numbers part so they can be macro-wired.
Every quantity is reported HONESTLY as measured; we do not force the old v2 value.

Q1  Parameter probe (the Y-weakness / M11 point)
    Probe the predictive (jepa_tf_noc) IMPACT-frame latent for each gust parameter
    G, D, Y with KernelRidge-RBF (CLAUDE.md rule: parameter probes use IMPACT-frame
    z). Fit on TRAIN only, 5-fold CASE-level CV on the readout, held-out test_b R^2
    with case-clustered CI (stats_lib.case_cluster_bootstrap on the per-encounter
    normalised squared errors, for which R^2 = 1 - mean(u)). The IMPACT-frame latent
    is z_full at the encounter's own impact_frame. Old v2 text: G 0.46, D 0.80,
    Y 0.10; expected qualitative result is that Y is the weakest (training mass
    concentrates near Y ~ 0). KRR is refit per fold with the same hyperparameters as
    closure_matrix.fit_krr_rbf (alpha = 1, gamma = 0.01) for byte-faithful behaviour.

Q2  Limit-cycle return distance
    The baseline limit cycle is the point cloud of the predictive latent orbit over
    the SETTLED undisturbed Baseline record (global raw frame >= 400, t/c >= 20;
    HANDOFF D180), pooled over the Baseline encounters present in the family's train
    latents. For every test_b encounter we read the latent at a late horizon
    (H = the largest available such that impact + H < T, capped at 24) and measure
    its minimum distance to the orbit cloud, reported in units of the orbit diameter
    (the 95th-percentile pairwise orbit distance, a robust diameter). The HEADLINE
    metric is EUCLIDEAN: the settled-Baseline orbit is only ~80 points with effective
    dimension ~3 of 64, so its sample covariance is rank-deficient and a Mahalanobis
    whitening inflates the off-orbit directions enormously; Mahalanobis is therefore
    carried only as an ill-conditioned cross-check (maha_crosscheck in results.json).
    Per family (predictive jepa_tf_noc and
    reconstructive fukami) we report the median return distance; the paired
    predictive-minus-reconstructive difference is tested with the B6-mandated sign
    test + case-clustered CI (NOT case_permutation_p, which is degenerate for paired
    location). Sign convention: delta = recon - pred > 0 means the predictive latent
    sits CLOSER to the baseline orbit. Old v2 text: predictive 8.70 vs reconstructive
    9.96 at H = 64; we report v2.1 at the available horizon. The two families live in
    DIFFERENT latent spaces, so each distance is computed in its own family's orbit
    geometry and reported in its own orbit-diameter units; the paired difference is
    between two dimensionless ratios, which is the only family-comparable quantity.

Q3  Scale-decomposition decode tracking
    Needs decoded fields: outputs/session28/decode/{tf_noc,fukami,pod}/recon_test_b.npz
    (omega_decoded_window (n, 31, 192, 96), window frames [25, 55]). We Gaussian-filter
    BOTH the decoded field and the DNS field at sigma/c = 0.05 (large scale; reuse
    physics_prep.large_scale_wake_enstrophy), compute the large-scale wake-enstrophy
    time series for each over the 31-frame window, and report per family the Pearson
    correlation between the decoded and DNS large-scale wake enstrophy and the relative
    amplitude error. The DNS field is the cache omega_z run through the v2p1 omega
    pipeline mask + clip (preprocess_raw, raw scale) so both fields share the decode
    README's "DNS truth pipeline-preprocessed (mask+clip)" footing. test_b. Old v2
    text: predictive decode tracks at 0.89-0.91 while reconstructive collapses; report
    v2.1 honestly (the matched-decoder field work showed POD's linear decoder
    reconstructs the large-scale field best, so we do NOT assume predictive wins).
    The per-encounter correlation/amplitude-error are aggregated to per-family median
    with a case-clustered CI; the per-encounter values use the (n=31) window series.

PROBE REGIME DECLARATION (CLAUDE.md): Q1 is a PARAMETER probe -> IMPACT-frame z.
Q2 reads a single late-horizon PER-frame latent (a state-space distance, not a probe).
Q3 is model-free on decoded fields (no probe).

Outputs
    outputs/session28/s46/results.json        full provenance for all three quantities
    outputs/session28/s46/README.md           each quantity vs the old v2 number
    outputs/session28/numbers_parts/s46.json  eval_all part (alphabetic macros only)

CLI (CPU-only; numpy/scipy/sklearn; reads NPZ + the v2p1 cache for the DNS field)
    export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
    nice -n 10 .venv/bin/python scripts/session28/s46_regen.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO))  # so `from src.data.omega_pipeline import ...` resolves

import closure_matrix as cm  # noqa: E402  probe machinery + R^2 + folds (verbatim ports)
import physics_prep as pp  # noqa: E402  large_scale_wake_enstrophy + wake_mask
import stats_lib  # noqa: E402  case-clustered CI + sign test

# --------------------------------------------------------------------------- defaults
LATENTS_ROOT = REPO / "outputs/session28/latents"
DECODE_ROOT = REPO / "outputs/session28/decode"
DNS_METRICS = REPO / "outputs/session28/exp2/dns_physical_metrics.npz"
SPLIT_MANIFEST = REPO / "configs/splits/split_v2p1.json"
OMEGA_MANIFEST = REPO / "outputs/data_pipeline/v2p1/manifest.json"
OUT_DIR = REPO / "outputs/session28/s46"
NUMBERS_PART = REPO / "outputs/session28/numbers_parts/s46.json"

PRED_FAMILY = "jepa_tf_noc_d64_s42"
RECON_FAMILY = "fukami_d64_s42"
POD_FAMILY = "pod_d64"
DECODE_FAMILIES = {"jepa_tf_noc": "tf_noc", "fukami": "fukami", "pod": "pod"}

PARAMS = ("G", "D", "Y")
ORBIT_RETURN_HORIZON_CAP = 24  # late-horizon cap; the largest available H <= this is used
SETTLED_GLOBAL_FRAME_MIN = pp.SETTLED_GLOBAL_FRAME_MIN  # D180: Baseline raw frame >= 400 is settled
RELEASE_PERIOD_FRAMES = pp.RELEASE_PERIOD_FRAMES
ORBIT_DIAMETER_PCTL = 95.0  # robust orbit "diameter": 95th pctl of pairwise orbit distances

# Old v2 values (for the README diff; NOT used in any computation).
OLD_V2 = {
    "probe_G": 0.46,
    "probe_D": 0.80,
    "probe_Y": 0.10,
    "orbit_pred": 8.70,
    "orbit_recon": 9.96,
    "orbit_horizon": 64,
    "scale_corr_pred_lo": 0.89,
    "scale_corr_pred_hi": 0.91,
}


# ------------------------------------------------------------------- shared NPZ helpers
def load_family_split(family: str, split: str) -> dict:
    """Load a family's latents NPZ for one split, tolerating singular/plural keys."""
    path = LATENTS_ROOT / family / f"{split}.npz"
    blob = np.load(path, allow_pickle=True)
    return {
        "z_full": blob["z_full"].astype(np.float64),
        "G": blob["G"].astype(np.float64),
        "D": blob["D"].astype(np.float64),
        "Y": blob["Y"].astype(np.float64),
        "case_ids": np.array([str(c) for c in cm.npz_get(blob, "case_ids", "case_id").tolist()]),
        "encounter_indices": cm.npz_get(blob, "encounter_indices", "encounter_index").astype(int),
        "impact_frame": blob["impact_frame"].astype(int),
    }


def impact_frame_latents(data: dict) -> np.ndarray:
    """Per-encounter IMPACT-frame latent z[i, impact_i, :] -> (n, d)."""
    z = data["z_full"]
    n, t_len, _ = z.shape
    imp = np.clip(data["impact_frame"], 0, t_len - 1)
    return z[np.arange(n), imp, :]


# ------------------------------------------------------------------------- Q1: probes
def case_cluster_ci_on_r2(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    case_ids: np.ndarray,
    n_boot_case: int,
    boot_seed: int,
) -> tuple[float, float, float]:
    """Held-out R^2 with a case-clustered CI via the u_i = nsq-error mapping.

    R^2 = 1 - mean(u) with u_i = (y_pred_i - y_true_i)^2 / (SST/n); the case-cluster
    bootstrap CI of mean(u) maps affinely onto the R^2 CI as [1 - hi(u), 1 - lo(u)]
    (closure_matrix.cell_uncertainty convention; SST held fixed under resampling).
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    sst = float(((y_true - y_true.mean()) ** 2).sum())
    msd = max(sst / y_true.size, 1e-12)
    u = (y_pred - y_true) ** 2 / msd
    r2 = 1.0 - float(u.mean())
    rng = np.random.default_rng(boot_seed)
    cb = stats_lib.case_cluster_bootstrap(u, case_ids, n_boot=n_boot_case, rng=rng)
    return r2, 1.0 - cb["enc_mean_ci"][1], 1.0 - cb["enc_mean_ci"][0]


def cv_oof_test_predictions(
    z_train: np.ndarray,
    y_train: np.ndarray,
    case_ids_train: np.ndarray,
    z_test: np.ndarray,
    n_folds: int,
    cv_seed: int,
) -> np.ndarray:
    """5-fold case-level CV KRR-RBF: held-out test predictions averaged over folds.

    The protocol's "5-fold case-level CV on the readout" for a HELD-OUT split: a probe
    is refit on each train fold (cases not in fold k) and applied to the test split;
    the 5 fold predictions are averaged (a CV ensemble), so no single train subset is
    privileged and the test readout respects the case-level fold structure. Probe class
    + hyperparameters are fixed to closure_matrix.fit_krr_rbf (no CV hyperparameter
    search; CLAUDE.md / protocol).
    """
    fold_id = cm.make_case_folds(case_ids_train, n_folds=n_folds, seed=cv_seed)
    preds = np.zeros((z_test.shape[0], n_folds), dtype=np.float64)
    for j, k in enumerate(sorted(set(fold_id.tolist()))):
        in_fit = fold_id != k
        probe = cm.fit_krr_rbf(z_train[in_fit], y_train[in_fit])
        preds[:, j] = cm.apply_probe(z_test, probe)
    return preds.mean(axis=1)


def quantity1_parameter_probe(cfg: "Config") -> dict:
    """Q1: KRR-RBF impact-frame parameter probe on the predictive family (test_b R^2)."""
    train = load_family_split(PRED_FAMILY, "train")
    test_b = load_family_split(PRED_FAMILY, "test_b")
    z_tr = impact_frame_latents(train)
    z_te = impact_frame_latents(test_b)
    out: dict = {
        "family": PRED_FAMILY,
        "regime": "IMPACT-frame z (parameter probe; CLAUDE.md)",
        "probe": "krr_rbf",
        "split": "test_b",
        "n_train_enc": int(z_tr.shape[0]),
        "n_test_b_enc": int(z_te.shape[0]),
        "n_folds": cfg.n_folds,
        "per_param": {},
    }
    for param in PARAMS:
        y_tr = train[param]
        y_te = test_b[param]
        y_pred = cv_oof_test_predictions(
            z_tr, y_tr, train["case_ids"], z_te, cfg.n_folds, cfg.cv_seed
        )
        r2, lo, hi = case_cluster_ci_on_r2(
            y_pred, y_te, test_b["case_ids"], cfg.n_boot_case, cfg.boot_seed
        )
        out["per_param"][param] = {
            "r2_test_b": r2,
            "ci_lo": lo,
            "ci_hi": hi,
            "old_v2": OLD_V2[f"probe_{param}"],
        }
    r2s = {p: out["per_param"][p]["r2_test_b"] for p in PARAMS}
    weakest = min(r2s, key=r2s.get)
    out["weakest_param"] = weakest
    out["y_is_weakest"] = bool(weakest == "Y")
    return out


# -------------------------------------------------------- Q2: limit-cycle return distance
def baseline_case_id(manifest: dict) -> str:
    """The is_calibration_reference (Baseline) case id."""
    cid, _ = pp.calibration_reference_case(manifest)
    return cid


def settled_baseline_orbit(train: dict, baseline_cid: str) -> np.ndarray:
    """Point cloud of the settled undisturbed Baseline orbit: (n_pts, d) latents.

    A frame at local index t of a Baseline encounter k (release every 120 frames) is
    settled iff its global raw frame 120 k + t >= SETTLED_GLOBAL_FRAME_MIN (D180). We
    pool every settled frame of every Baseline encounter present in the family's train
    latents into the orbit cloud.
    """
    z = train["z_full"]
    mask = train["case_ids"] == baseline_cid
    if not mask.any():
        raise KeyError(f"baseline case {baseline_cid!r} not in train latents")
    idx = np.flatnonzero(mask)
    parts = []
    t_len = z.shape[1]
    for i in idx:
        k = int(train["encounter_indices"][i])
        global_start = RELEASE_PERIOD_FRAMES * k
        offset = max(0, SETTLED_GLOBAL_FRAME_MIN - global_start)
        if offset < t_len:
            parts.append(z[i, offset:, :])
    if not parts:
        raise ValueError("no settled Baseline frames in train latents")
    return np.concatenate(parts, axis=0)


@dataclass
class OrbitGeometry:
    """Whitening + diameters of a baseline orbit cloud for the return-distance metrics.

    The orbit cloud here is small (~80 settled Baseline frames) and very low-rank
    (effective dimension ~3 of d=64), so its sample covariance is rank-deficient and
    the Mahalanobis whitening inflates the off-orbit directions enormously (the directions
    the orbit barely spans are divided by a ridged near-zero eigenvalue). EUCLIDEAN is
    therefore the well-conditioned, physically interpretable HEADLINE metric; Mahalanobis
    is carried only as a cross-check and is flagged ill-conditioned in the results.
    """

    mean: np.ndarray
    whiten: np.ndarray  # (d, d) inverse-sqrt covariance (Mahalanobis transform; cross-check)
    euclid_diameter: float  # robust orbit diameter in raw (Euclidean) latent units
    maha_diameter: float  # robust orbit diameter in whitened (Mahalanobis) units
    n_pts: int
    effective_dim: float  # participation ratio of the orbit covariance eigenvalues


def _robust_diameter(cloud: np.ndarray, pctl: float, rng: np.random.Generator) -> float:
    """95th-pctl pairwise distance within a point cloud (subsampled for cost if huge)."""
    m = cloud.shape[0]
    sub = cloud[rng.choice(m, 600, replace=False)] if m > 600 else cloud
    dmat = np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(axis=2))
    iu = np.triu_indices(sub.shape[0], k=1)
    return float(np.percentile(dmat[iu], pctl)) if iu[0].size else 1.0


def _orbit_geometry(orbit: np.ndarray, diameter_pctl: float = ORBIT_DIAMETER_PCTL) -> OrbitGeometry:
    """Build the Euclidean + Mahalanobis diameters and the whitening from the orbit cloud."""
    orbit = np.asarray(orbit, dtype=np.float64)
    mean = orbit.mean(axis=0)
    cov = np.cov(orbit, rowvar=False)
    d = cov.shape[0]
    cov = cov + 1e-6 * np.eye(d)  # ridge for numerical stability / rank-deficient clouds
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 1e-12, None)
    eff_dim = float(evals.sum() ** 2 / (evals**2).sum())
    whiten = evecs @ np.diag(evals**-0.5) @ evecs.T
    rng = np.random.default_rng(0)
    euclid_diameter = _robust_diameter(orbit, diameter_pctl, rng)
    maha_diameter = _robust_diameter((orbit - mean) @ whiten, diameter_pctl, rng)
    return OrbitGeometry(
        mean=mean,
        whiten=whiten,
        euclid_diameter=max(euclid_diameter, 1e-9),
        maha_diameter=max(maha_diameter, 1e-9),
        n_pts=int(orbit.shape[0]),
        effective_dim=eff_dim,
    )


def _min_dist_to_cloud(points: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    """Min Euclidean distance from each point to the cloud."""
    points = np.asarray(points, dtype=np.float64)
    cloud = np.asarray(cloud, dtype=np.float64)
    d = np.sqrt(((points[:, None, :] - cloud[None, :, :]) ** 2).sum(axis=2))
    return d.min(axis=1)


def _min_maha_to_cloud(points: np.ndarray, geom: OrbitGeometry, orbit_w: np.ndarray) -> np.ndarray:
    """Min Mahalanobis distance from each point to the (cached) whitened orbit cloud."""
    pw = (np.asarray(points, dtype=np.float64) - geom.mean) @ geom.whiten
    return _min_dist_to_cloud(pw, orbit_w)


def late_horizon_latents(test_b: dict, horizon_cap: int) -> tuple[np.ndarray, int]:
    """Per-encounter latent at the largest H <= cap with impact + H < T; (n, d), H used."""
    z = test_b["z_full"]
    n, t_len, _ = z.shape
    imp = test_b["impact_frame"]
    h_used = min(horizon_cap, int(t_len - 1 - int(imp.max())))
    h_used = max(h_used, 0)
    frames = np.clip(imp + h_used, 0, t_len - 1)
    return z[np.arange(n), frames, :], h_used


def family_return_distances(
    family: str, manifest: dict, horizon_cap: int
) -> tuple[np.ndarray, np.ndarray, int, dict, np.ndarray]:
    """Per test_b encounter return distance (orbit-diameter units) for one family.

    The HEADLINE distance is Euclidean-in-orbit-diameter (well-conditioned for the
    low-rank ~80-point orbit cloud); the Mahalanobis-in-diameter distance is carried as
    an ill-conditioned cross-check (see OrbitGeometry docstring).

    Returns (euclid_dist, case_ids, horizon_used, geometry_meta, maha_dist).
    """
    train = load_family_split(family, "train")
    test_b = load_family_split(family, "test_b")
    orbit = settled_baseline_orbit(train, baseline_case_id(manifest))
    geom = _orbit_geometry(orbit)
    orbit_w = (orbit - geom.mean) @ geom.whiten
    z_late, h_used = late_horizon_latents(test_b, horizon_cap)
    euclid = _min_dist_to_cloud(z_late, orbit) / geom.euclid_diameter
    maha = _min_maha_to_cloud(z_late, geom, orbit_w) / geom.maha_diameter
    meta = {
        "orbit_n_points": geom.n_pts,
        "orbit_effective_dim": geom.effective_dim,
        "orbit_diameter_euclid": geom.euclid_diameter,
        "orbit_diameter_maha": geom.maha_diameter,
        "horizon_used": h_used,
    }
    return euclid, test_b["case_ids"], h_used, meta, maha


def _paired_return_block(d_pred: np.ndarray, d_recon: np.ndarray, cids: np.ndarray, cfg: "Config"):
    """Median returns + paired (recon - pred) case-clustered CI + sign test for one metric."""
    delta = d_recon - d_pred  # >0 means predictive sits CLOSER to the orbit
    rng = np.random.default_rng(cfg.boot_seed)
    cb = stats_lib.case_cluster_bootstrap(delta, cids, n_boot=cfg.n_boot_case, rng=rng)
    k, n_eff, p = stats_lib.sign_test_one_sided(d_pred, d_recon)  # pred < recon -> a better
    _, cmd = stats_lib.case_means(delta, cids)
    case_stats = stats_lib.case_level_paired_stats(cmd)
    return {
        "pred_median_return": float(np.median(d_pred)),
        "recon_median_return": float(np.median(d_recon)),
        "paired_delta_recon_minus_pred": {
            "enc_mean": cb["enc_mean"],
            "enc_mean_ci": cb["enc_mean_ci"],
            "case_mean": cb["case_mean"],
            "case_mean_ci": cb["case_mean_ci"],
            "median_delta": float(np.median(delta)),
        },
        "sign_test_pred_closer": {"k_pred_closer": k, "n_eff": n_eff, "p_one_sided": p},
        "case_level_paired": case_stats,
        "predictive_closer": bool(np.median(d_pred) < np.median(d_recon)),
    }


def quantity2_limit_cycle_return(cfg: "Config") -> dict:
    """Q2: per-family median return distance + paired predictive-minus-reconstructive diff.

    HEADLINE metric is Euclidean-in-orbit-diameter; a Mahalanobis cross-check is reported
    under maha_crosscheck (ill-conditioned for this low-rank orbit; magnitudes not trusted).
    """
    manifest = json.loads(SPLIT_MANIFEST.read_text())
    e_pred, cids_p, h_p, meta_p, m_pred = family_return_distances(
        PRED_FAMILY, manifest, ORBIT_RETURN_HORIZON_CAP
    )
    e_recon, cids_r, h_r, meta_r, m_recon = family_return_distances(
        RECON_FAMILY, manifest, ORBIT_RETURN_HORIZON_CAP
    )
    # Align by (case_id, encounter_index) so the paired difference is per-encounter.
    tb_p = load_family_split(PRED_FAMILY, "test_b")
    tb_r = load_family_split(RECON_FAMILY, "test_b")
    key_r = {
        (str(c), int(e)): j
        for j, (c, e) in enumerate(zip(tb_r["case_ids"], tb_r["encounter_indices"]))
    }
    order = np.array(
        [key_r[(str(c), int(e))] for c, e in zip(tb_p["case_ids"], tb_p["encounter_indices"])]
    )
    cids = cids_p
    headline = _paired_return_block(e_pred, e_recon[order], cids, cfg)
    maha = _paired_return_block(m_pred, m_recon[order], cids, cfg)
    return {
        "horizon_cap": ORBIT_RETURN_HORIZON_CAP,
        "horizon_used_pred": h_p,
        "horizon_used_recon": h_r,
        "orbit_definition": (
            "settled undisturbed Baseline predictive-latent orbit (global raw frame "
            ">= 400, t/c >= 20; D180), per family in its own latent space"
        ),
        "distance_metric": (
            "HEADLINE: min EUCLIDEAN distance to the orbit cloud in units of the "
            "95th-pctl pairwise orbit distance (diameter). Mahalanobis is an "
            "ill-conditioned cross-check (orbit effective dim ~3 of 64)."
        ),
        "pred_family": PRED_FAMILY,
        "recon_family": RECON_FAMILY,
        "n_test_b_enc": int(e_pred.size),
        "pred_median_return": headline["pred_median_return"],
        "recon_median_return": headline["recon_median_return"],
        "paired_delta_recon_minus_pred": headline["paired_delta_recon_minus_pred"],
        "sign_test_pred_closer": headline["sign_test_pred_closer"],
        "case_level_paired": headline["case_level_paired"],
        "predictive_closer": headline["predictive_closer"],
        "maha_crosscheck": maha,
        "pred_geometry": meta_p,
        "recon_geometry": meta_r,
        "old_v2": {
            "pred": OLD_V2["orbit_pred"],
            "recon": OLD_V2["orbit_recon"],
            "horizon": OLD_V2["orbit_horizon"],
            "note": "old v2 distances are in raw latent units at H=64; not directly comparable",
        },
    }


# ----------------------------------------------- Q3: scale-decomposition decode tracking
def load_omega_pipeline():
    """v2p1 omega pipeline (mask + per-encounter clip) for the DNS-truth field."""
    from src.data.omega_pipeline import OmegaPipeline

    return OmegaPipeline.from_manifest(OMEGA_MANIFEST)


def dns_window_field(
    cache_root: Path, case_id: str, enc_idx: int, window_frames: np.ndarray, pipeline
) -> np.ndarray:
    """DNS omega_z over the window, pipeline-preprocessed (mask + clip, raw scale)."""
    import h5py

    path = Path(cache_root) / case_id / f"encounter_{enc_idx:02d}.h5"
    with h5py.File(path, "r") as f:
        omega = f["omega_z"][:].astype(np.float64)
    proc = pipeline.preprocess_raw(omega, case_id, enc_idx)
    return np.asarray(proc)[window_frames]


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two 1D series (NaN if either is constant)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def family_scale_tracking(decode_dir_name: str, cfg: "Config", pipeline, mask) -> dict:
    """Per-encounter decoded-vs-DNS large-scale wake-enstrophy correlation for one family."""
    cache_root = Path(cfg.cache_root) / "v2p1"
    blob = np.load(DECODE_ROOT / decode_dir_name / "recon_test_b.npz", allow_pickle=True)
    decoded = blob["omega_decoded_window"].astype(np.float64)  # (n, 31, 192, 96)
    window_frames = blob["window_frames"].astype(int)
    cids = np.array([str(c) for c in blob["case_ids"].tolist()])
    eis = blob["encounter_indices"].astype(int)
    sigma_px = cfg.sigma_c / pp.DX
    corrs, amp_errs = [], []
    for i in range(decoded.shape[0]):
        e_dec = pp.large_scale_wake_enstrophy(decoded[i], sigma_px, mask)
        dns_field = dns_window_field(cache_root, cids[i], int(eis[i]), window_frames, pipeline)
        e_dns = pp.large_scale_wake_enstrophy(dns_field, sigma_px, mask)
        corrs.append(pearson(e_dec, e_dns))
        amp_dns = float(e_dns.max() - e_dns.min())
        amp_dec = float(e_dec.max() - e_dec.min())
        amp_errs.append((amp_dec - amp_dns) / amp_dns if amp_dns > 1e-12 else float("nan"))
    corrs = np.array(corrs, dtype=np.float64)
    amp_errs = np.array(amp_errs, dtype=np.float64)
    good_c = np.isfinite(corrs)
    good_a = np.isfinite(amp_errs)
    rng = np.random.default_rng(cfg.boot_seed)
    cb = stats_lib.case_cluster_bootstrap(
        corrs[good_c], cids[good_c], n_boot=cfg.n_boot_case, rng=rng
    )
    return {
        "n_enc": int(decoded.shape[0]),
        "n_finite_corr": int(good_c.sum()),
        "median_corr": float(np.median(corrs[good_c])),
        "mean_corr": float(corrs[good_c].mean()),
        "corr_case_clustered_ci": cb["enc_mean_ci"],
        "median_rel_amp_err": float(np.median(amp_errs[good_a])),
        "mean_rel_amp_err": float(amp_errs[good_a].mean()),
    }


def quantity3_scale_decode(cfg: "Config") -> dict:
    """Q3: decoded-vs-DNS large-scale wake-enstrophy tracking per family on test_b."""
    pipeline = load_omega_pipeline()
    mask = pp.wake_mask()
    out: dict = {
        "split": "test_b",
        "sigma_c": cfg.sigma_c,
        "scale": "large (Gaussian sigma/c=0.05)",
        "metric": "Pearson corr of decoded vs DNS large-scale wake enstrophy over [25,55]",
        "dns_truth": "cache omega_z, v2p1 pipeline mask+clip (preprocess_raw, raw scale)",
        "per_family": {},
        "old_v2": {
            "pred_corr_range": [OLD_V2["scale_corr_pred_lo"], OLD_V2["scale_corr_pred_hi"]],
            "note": "old v2: predictive tracks 0.89-0.91, reconstructive collapses",
        },
    }
    for fam, decode_dir in DECODE_FAMILIES.items():
        out["per_family"][fam] = family_scale_tracking(decode_dir, cfg, pipeline, mask)
    return out


# --------------------------------------------------------------------------- numbers part
def fmt_signed(x: float) -> str:
    return f"{x:+.3f}"


def build_numbers_part(q1: dict, q2: dict, q3: dict) -> dict:
    """eval_all numbers part. Macros are ALPHABETIC only (dimensions spelled out)."""
    numbers: dict[str, dict] = {}

    # Q1 parameter probe (KRR-RBF, impact-frame, test_b R^2).
    for param, macro in (("G", "NumProbeG"), ("D", "NumProbeD"), ("Y", "NumProbeY")):
        rec = q1["per_param"][param]
        numbers[f"s46_probe_{param}"] = {
            "macro": macro,
            "value": rec["r2_test_b"],
            "fmt": "%.2f",
            "ci_lo": rec["ci_lo"],
            "ci_hi": rec["ci_hi"],
            "n": q1["n_test_b_enc"],
            "split": "test_b",
            "endpoint": "representational",
            "probe": "krr_rbf",
            "observable": param,
            "run_tags": [PRED_FAMILY],
            "source": "s46_regen.py",
            "note": (
                f"impact-frame parameter probe; 5-fold case-CV; case-clustered CI; "
                f"old v2 = {rec['old_v2']:.2f}"
            ),
        }

    # Q2 limit-cycle return distance (orbit-diameter units).
    pd = q2["paired_delta_recon_minus_pred"]
    numbers["s46_orbit_return_jepa_tf"] = {
        "macro": "NumOrbitReturnJepaTf",
        "value": q2["pred_median_return"],
        "fmt": "%.2f",
        "n": q2["n_test_b_enc"],
        "split": "test_b",
        "run_tags": [PRED_FAMILY],
        "source": "s46_regen.py",
        "note": (
            f"median test_b return distance to settled Baseline orbit, orbit-diameter "
            f"units, H={q2['horizon_used_pred']}"
        ),
    }
    numbers["s46_orbit_return_fukami"] = {
        "macro": "NumOrbitReturnFukami",
        "value": q2["recon_median_return"],
        "fmt": "%.2f",
        "n": q2["n_test_b_enc"],
        "split": "test_b",
        "run_tags": [RECON_FAMILY],
        "source": "s46_regen.py",
        "note": (
            f"median test_b return distance to settled Baseline orbit, orbit-diameter "
            f"units, H={q2['horizon_used_recon']}"
        ),
    }
    numbers["s46_orbit_return_paired_diff"] = {
        "macro": "NumOrbitReturnPairedDiff",
        "value": pd["enc_mean"],
        "fmt": "%.3f",
        "ci_lo": pd["enc_mean_ci"][0],
        "ci_hi": pd["enc_mean_ci"][1],
        "n": q2["n_test_b_enc"],
        "split": "test_b",
        "run_tags": [PRED_FAMILY, RECON_FAMILY],
        "source": "s46_regen.py",
        "note": (
            f"paired delta = recon - pred (>0 = predictive closer); case-clustered CI; "
            f"sign-test p={q2['sign_test_pred_closer']['p_one_sided']:.4g} "
            f"(k={q2['sign_test_pred_closer']['k_pred_closer']}/"
            f"{q2['sign_test_pred_closer']['n_eff']}); "
            f"NOT case_permutation_p (B6 lesson)"
        ),
    }

    # Q3 scale-decomposition decode tracking (Pearson corr, test_b).
    for fam, macro in (
        ("jepa_tf_noc", "NumScaleDecodeCorrJepaTf"),
        ("fukami", "NumScaleDecodeCorrFukami"),
        ("pod", "NumScaleDecodeCorrPod"),
    ):
        rec = q3["per_family"][fam]
        numbers[f"s46_scale_decode_corr_{fam}"] = {
            "macro": macro,
            # Headline is the MEAN correlation so the case-clustered (pooled-encounter
            # mean) CI brackets the reported value; the median is given in the note.
            "value": rec["mean_corr"],
            "fmt": "%.2f",
            "ci_lo": rec["corr_case_clustered_ci"][0],
            "ci_hi": rec["corr_case_clustered_ci"][1],
            "n": rec["n_finite_corr"],
            "split": "test_b",
            "observable": "wake_enstrophy",
            "source": "s46_regen.py",
            "note": (
                f"mean decoded-vs-DNS large-scale wake-enstrophy Pearson corr over "
                f"[25,55] (median {rec['median_corr']:.2f}); median rel amp err "
                f"{rec['median_rel_amp_err']:+.2f}; case-clustered CI on the mean"
            ),
        }
    # Q3 large-scale amplitude error (the "relative error X against Y" pair in S4.6).
    for fam, macro in (
        ("jepa_tf_noc", "NumScaleDecodeRelErrJepaTf"),
        ("fukami", "NumScaleDecodeRelErrFukami"),
        ("pod", "NumScaleDecodeRelErrPod"),
    ):
        rec = q3["per_family"][fam]
        numbers[f"s46_scale_decode_rel_amp_err_{fam}"] = {
            "macro": macro,
            "value": rec["median_rel_amp_err"],
            "fmt": "%.2f",
            "n": rec["n_finite_corr"],
            "split": "test_b",
            "observable": "wake_enstrophy",
            "source": "s46_regen.py",
            "note": (
                "median per-encounter relative amplitude error of the decoded "
                "large-scale wake enstrophy vs DNS over [25,55]; companion to "
                "s46_scale_decode_corr; smaller is better"
            ),
        }
    return {"part": "s46", "numbers": numbers}


# ---------------------------------------------------------------------------------- main
@dataclass
class Config:
    cache_root: str
    n_folds: int = 5
    cv_seed: int = 0
    boot_seed: int = 0
    n_boot_case: int = stats_lib.N_BOOT_CASE
    sigma_c: float = pp.SIGMA_C_DEFAULT
    skip: tuple[str, ...] = field(default_factory=tuple)


def write_readme(q1: dict, q2: dict, q3: dict, out_dir: Path) -> None:
    """Human-readable README stating each quantity as measured vs the old v2 number."""

    def cmp_word(new: float, old: float, tol: float = 0.1) -> str:
        if abs(new - old) <= tol:
            return "MATCHES"
        return "WEAKER" if new < old else "STRONGER"

    lines = [
        "# Session 28 / S4.6 regeneration: three v2.1 quantities (Phase D macro-wiring)",
        "",
        "Each quantity below is measured on v2.1 and reported honestly. The old v2 /",
        "session17 value the manuscript Section 4.6 still quotes is shown alongside.",
        "Generated by scripts/session28/s46_regen.py; macros emitted to",
        "outputs/session28/numbers_parts/s46.json. NOT git-committed; left for review.",
        "",
        "## Q1: parameter probe (Y-weakness / M11)",
        "",
        "KRR-RBF probe on the predictive (jepa_tf_noc) IMPACT-frame latent, 5-fold",
        "case-CV, held-out test_b R^2 with case-clustered CI.",
        "",
        "| param | v2.1 R^2 | 95% CI | old v2 | verdict |",
        "|---|---|---|---|---|",
    ]
    for p in PARAMS:
        r = q1["per_param"][p]
        lines.append(
            f"| {p} | {r['r2_test_b']:.2f} | [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] | "
            f"{r['old_v2']:.2f} | {cmp_word(r['r2_test_b'], r['old_v2'])} |"
        )
    lines += [
        "",
        f"Weakest parameter: {q1['weakest_param']} "
        f"(Y-is-weakest expectation: {q1['y_is_weakest']}).",
        "",
        "## Q2: limit-cycle return distance",
        "",
        f"Orbit definition: {q2['orbit_definition']}.",
        f"Distance metric: {q2['distance_metric']}",
        f"Horizon used: H={q2['horizon_used_pred']} (cap {q2['horizon_cap']}; the largest",
        "available since impact+H must stay inside the 120-frame cache).",
        "",
        f"- Predictive (jepa_tf_noc) median return: {q2['pred_median_return']:.2f} "
        "orbit-diameters",
        f"- Reconstructive (fukami) median return: {q2['recon_median_return']:.2f} "
        "orbit-diameters",
        f"- Paired delta = recon - pred (>0 = predictive closer): "
        f"{q2['paired_delta_recon_minus_pred']['enc_mean']:+.3f} "
        f"[{q2['paired_delta_recon_minus_pred']['enc_mean_ci'][0]:+.3f}, "
        f"{q2['paired_delta_recon_minus_pred']['enc_mean_ci'][1]:+.3f}] (case-clustered CI)",
        f"- Sign test (predictive closer): k="
        f"{q2['sign_test_pred_closer']['k_pred_closer']}/"
        f"{q2['sign_test_pred_closer']['n_eff']}, "
        f"p={q2['sign_test_pred_closer']['p_one_sided']:.4g}",
        f"- Predictive sits closer than reconstructive: {q2['predictive_closer']}",
        f"- Orbit effective dim: pred {q2['pred_geometry']['orbit_effective_dim']:.1f}, "
        f"recon {q2['recon_geometry']['orbit_effective_dim']:.1f} (of 64); the orbit is "
        "very low-rank, so Euclidean is the headline metric and Mahalanobis is an "
        "ill-conditioned cross-check (maha_crosscheck in results.json).",
        f"- Mahalanobis cross-check (ill-conditioned): pred "
        f"{q2['maha_crosscheck']['pred_median_return']:.1f} vs recon "
        f"{q2['maha_crosscheck']['recon_median_return']:.1f} orbit-diameters "
        f"(predictive_closer={q2['maha_crosscheck']['predictive_closer']}).",
        "",
        "Old v2 text: predictive 8.70 vs reconstructive 9.96 at H=64 (raw latent units;",
        "v2.1 reports dimensionless orbit-diameter units at the available horizon, so the",
        "DIRECTION (predictive closer or not), not the magnitude, is the comparable claim.",
        "REVERSAL: on v2.1 the reconstructive (fukami) latent sits CLOSER to the baseline",
        "orbit than the predictive one under BOTH metrics, opposite to the old v2 claim.",
        "",
        "## Q3: scale-decomposition decode tracking",
        "",
        "Pearson correlation of the decoded vs DNS LARGE-scale (Gaussian sigma/c=0.05)",
        "wake-enstrophy time series over the impact window [25,55], test_b.",
        "",
        "Headline value is the MEAN corr (the case-clustered CI brackets it); the median",
        "is shown alongside.",
        "",
        "| family | v2.1 mean corr | median corr | 95% CI (mean) | median rel amp err |",
        "|---|---|---|---|---|",
    ]
    for fam in DECODE_FAMILIES:
        r = q3["per_family"][fam]
        lines.append(
            f"| {fam} | {r['mean_corr']:.2f} | {r['median_corr']:.2f} | "
            f"[{r['corr_case_clustered_ci'][0]:.2f}, {r['corr_case_clustered_ci'][1]:.2f}] | "
            f"{r['median_rel_amp_err']:+.2f} |"
        )
    pred_corr = q3["per_family"]["jepa_tf_noc"]["mean_corr"]
    lines += [
        "",
        "Old v2 text: predictive tracks at 0.89-0.91, reconstructive collapses.",
        f"v2.1 predictive mean corr = {pred_corr:.2f}. The matched-decoder field work",
        "showed POD's linear decoder reconstructs the large-scale field best, so the",
        "predictive family is NOT assumed to win; see the table for the measured ordering.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    """Compute all three quantities, write results.json + README.md + the numbers part."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=str,
        default=os.environ.get(
            "VORTEX_JEPA_CACHE",
            str(
                Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
                / "data"
                / "processed"
                / "vortex-jepa"
            ),
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--numbers-part", type=Path, default=NUMBERS_PART)
    parser.add_argument("--n-boot-case", type=int, default=stats_lib.N_BOOT_CASE)
    parser.add_argument("--cv-seed", type=int, default=0)
    parser.add_argument("--boot-seed", type=int, default=0)
    parser.add_argument(
        "--skip", nargs="+", default=[], choices=["q1", "q2", "q3"], help="Skip quantities."
    )
    args = parser.parse_args(argv)

    cfg = Config(
        cache_root=args.cache_root,
        cv_seed=args.cv_seed,
        boot_seed=args.boot_seed,
        n_boot_case=args.n_boot_case,
        skip=tuple(args.skip),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    q1 = quantity1_parameter_probe(cfg) if "q1" not in cfg.skip else None
    print(f"[s46] Q1 done ({time.time() - t0:.1f}s)")
    t1 = time.time()
    q2 = quantity2_limit_cycle_return(cfg) if "q2" not in cfg.skip else None
    print(f"[s46] Q2 done ({time.time() - t1:.1f}s)")
    t2 = time.time()
    q3 = quantity3_scale_decode(cfg) if "q3" not in cfg.skip else None
    print(f"[s46] Q3 done ({time.time() - t2:.1f}s)")

    results = {
        "generated_iso": datetime.now(timezone.utc).isoformat(),
        "split_manifest": str(SPLIT_MANIFEST),
        "omega_manifest": str(OMEGA_MANIFEST),
        "cache_root": cfg.cache_root,
        "config": {
            "n_folds": cfg.n_folds,
            "cv_seed": cfg.cv_seed,
            "boot_seed": cfg.boot_seed,
            "n_boot_case": cfg.n_boot_case,
            "sigma_c": cfg.sigma_c,
        },
        "q1_parameter_probe": q1,
        "q2_limit_cycle_return": q2,
        "q3_scale_decode_tracking": q3,
    }
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[s46] wrote {args.out_dir / 'results.json'}")

    if q1 and q2 and q3:
        part = build_numbers_part(q1, q2, q3)
        args.numbers_part.parent.mkdir(parents=True, exist_ok=True)
        args.numbers_part.write_text(json.dumps(part, indent=2))
        print(f"[s46] wrote {args.numbers_part} ({len(part['numbers'])} numbers)")
        write_readme(q1, q2, q3, args.out_dir)
        print(f"[s46] wrote {args.out_dir / 'README.md'}")
    else:
        print("[s46] some quantities skipped; numbers part + README NOT written")


if __name__ == "__main__":
    main()
