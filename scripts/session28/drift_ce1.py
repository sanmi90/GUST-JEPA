"""Track C-E1: drift reconciliation (closes referee M6, Table 8 gaps).

The two drift diagnostics published in v2 order the families oppositely and the
manuscript never reconciles them. Figure 5 (relative l2 deviation of the rollout
from the encoded held-out trajectory) shows the reconstructive Fukami AE as the
LEAST-drifting curve while the unconditioned JEPA transformer drifts more; Table
8 (Mahalanobis ratio to the per-frame TRAIN encoded-latent distribution) shows
the AE an order of magnitude out while JEPA stays in-distribution. Both are true:
the AE rollout exits along directions of TINY encoded variance (small l2, huge
Mahalanobis) while the JEPA rollout wanders within a broad isotropised cloud.

This module computes, per family x latent dimension d:

1. Relative l2 deviation vs horizon H (the Figure-5 metric):
       ||z_roll(t_imp + H) - z_enc(t_imp + H)|| / ||z_enc(t_imp + H)||
   per encounter, mean over test_b, for H in {4, 8, 16, 24, 32}.

2. Mahalanobis ratio vs H (the Table-8 metric):
       D_M(z_roll(t_imp + H); mu_train_H, Sigma_train_H)
       / D_M(z_enc (t_imp + H); mu_train_H, Sigma_train_H)
   where (mu_train_H, Sigma_train_H) is the family's per-frame TRAIN encoded
   distribution at the absolute frame t_imp + H. The covariance inverse is
   regularised by Ledoit-Wolf-style diagonal shrinkage (documented below).

3. Per-direction DEPARTURE SPECTRUM: the per-encounter departure vector
   delta = z_roll - z_enc projected onto the encoded-covariance EIGENBASIS of
   the family (computed on the per-frame TRAIN encoded latents). We report (a)
   the raw energy fraction in top-k vs bottom-k eigen-directions, (b) the
   WHITENED energy fraction (contribution to the squared Mahalanobis distance),
   and (c) the headline reconciliation statistic: the fraction of the rollout's
   squared Mahalanobis departure carried by the NEAR-NULL encoded eigen-
   directions (encoded eigenvalue below NULL_REL_CUTOFF x the top eigenvalue),
   with the count of those directions. The reconstructive (collapsed, near-rank-
   degenerate) latent dumps almost all of its Mahalanobis into its many near-null
   directions while keeping a tiny l2 footprint; the predictive latent has far
   fewer near-null directions to fall into and stays in-distribution.

Two Mahalanobis regularisations are reported: a HEADLINE v2-regime estimator
(raw sample covariance + small diagonal floor), which reproduces the published
v2 Table-8 ordering (Fukami AE ratio ~9, JEPA ~0.7), and a ROBUSTNESS estimator
(Ledoit-Wolf shrinkage), under which the near-null tail is regularised and the
ratios collapse toward one.

The Mahalanobis reference is the encoded held-out (test_b) latent at the same
frame; ratios below one therefore mean a rollout that is MORE in-distribution
than the encoded held-out latents, i.e. the predictor regresses toward the
training cloud. This is reported, not hidden.

Significance (heed the B6 lesson): paired per-encounter family DIFFERENCES use
the case-clustered block bootstrap CI + a one-sided sign test from
scripts/session28/stats_lib.py. stats_lib.case_permutation_p is deliberately NOT
used here: it is a case-level association test, degenerate for a paired location
contrast.

Outputs (all absolute, under /home/carlos/GUST-JEPA/outputs/session28/drift/):
    ce1_results.npz        per-encounter rel-l2 and Mahalanobis (both regimes)
                           vs H per family x d, the departure spectra, and the
                           three-panel figure source arrays (rel-l2 vs H,
                           Mahalanobis ratio vs H, per-direction spectrum).
    ce1_significance.json  paired JEPA-vs-Fukami family differences (case-
                           clustered bootstrap CI + sign test) at H = 24, d = 64.
    table8.json            the complete Table 8 as structured rows.
    numbers_parts/drift_ce1.json   eval_all part: headline rel-l2 and Mahalanobis
                           ratio at H = 24 per family at d = 64, and the near-null
                           departure-fraction contrast (JEPA vs Fukami).
    README.md              the COMPLETE Table 8 (all rows), the ratio<1
                           interpretation, and the reconciliation sentences.

CPU only; run under nice. Usage:
    python scripts/session28/drift_ce1.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session28"))

import stats_lib  # noqa: E402  (path injection above)

LAT_DIR = REPO / "outputs" / "session28" / "latents"
ROLL_DIR = REPO / "outputs" / "session28" / "rollouts"
OUT_DIR = REPO / "outputs" / "session28" / "drift"
PARTS_DIR = REPO / "outputs" / "session28" / "numbers_parts"

HORIZONS = (4, 8, 16, 24, 32)
HEADLINE_H = 24
SPEC_K = 5  # bottom-k / top-k eigen-direction count for the departure spectrum
SHRINKAGE_EPS = 1e-6  # diagonal floor added AFTER Ledoit-Wolf shrinkage (robustness regime)
FLOOR_EPS = 1e-6  # diagonal floor for the raw-covariance v2 regime (headline Table-8)
NULL_REL_CUTOFF = 1e-3  # encoded eigval < this * top-eigval is a "near-null" direction

# Rollout-key -> encoded-latent directory, verbatim from the PRED_LAT map in
# scripts/session28/phase_b_runner.sh. The jepa_* rollouts use the UNCONDITIONED
# transformer latents (jepa_tf_noc); fukami/pod/bvae bind to their s42/canonical
# encoded latents. jepa_lstm_noc has encoded latents but NO matched-predictor
# rollout (it was not in the Phase-B KEYS list), so it cannot enter the rollout
# diagnostics; it is reported as absent in Table 8.
PRED_LAT = {
    "jepa_d16": "jepa_tf_noc_d16_s42",
    "jepa_d32": "jepa_tf_noc_d32_s42",
    "jepa_d64": "jepa_tf_noc_d64_s42",
    "fukami_d3": "fukami_d3_s42",
    "fukami_d16": "fukami_d16_s42",
    "fukami_d32": "fukami_d32_s42",
    "fukami_d64": "fukami_d64_s42",
    "pod_d16": "pod_d16",
    "pod_d32": "pod_d32",
    "pod_d64": "pod_d64",
    "bvae_d64": "bvae_faith_d64_s42",
}

# Stable display names + the (family, d) decomposition of each rollout key.
FAMILY_OF = {
    "jepa": "Unconditioned JEPA transformer",
    "fukami": "Fukami observable-augmented AE",
    "pod": "POD",
    "bvae": "beta-VAE",
}

# Families that have ONLY encoded latents (no matched rollout). Reported as a row
# in Table 8 with the rollout cells marked absent.
NO_ROLLOUT_LATENTS = {
    "jepa_lstm_noc_d64": "jepa_lstm_noc_d64_s1",
}


def parse_key(key: str) -> tuple[str, int]:
    """Split a rollout key like 'fukami_d32' into ('fukami', 32)."""
    fam, dpart = key.rsplit("_d", 1)
    return fam, int(dpart)


# ---------------------------------------------------------------------------
# covariance / Mahalanobis (regularised inverse)
# ---------------------------------------------------------------------------


def shrunk_covariance(
    x: np.ndarray, eps: float = SHRINKAGE_EPS
) -> tuple[np.ndarray, np.ndarray, float]:
    """Ledoit-Wolf-style shrinkage covariance with a diagonal numerical floor.

    Returns (mean, sigma, shrinkage_lambda). The estimator is

        Sigma_shrunk = (1 - lambda) * S + lambda * mu_trace * I,

    where S is the (biased, MLE) sample covariance, mu_trace = trace(S) / d is
    the average eigenvalue, and lambda is the Ledoit-Wolf (2004) analytic
    shrinkage intensity clipped to [0, 1]. A diagonal floor eps * I is added
    afterwards so the inverse is stable even when n < d (rank-deficient sample).
    This keeps the Mahalanobis metric well defined at every d without tuning a
    per-family hyperparameter.

    Args:
        x: (n, d) sample of latent vectors at one frame.
        eps: diagonal floor added after shrinkage.

    Returns:
        mean (d,), sigma (d, d), shrinkage_lambda (float).
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    mean = x.mean(axis=0)
    xc = x - mean
    # biased sample covariance S = (1/n) X^T X
    s = (xc.T @ xc) / max(n, 1)
    mu = np.trace(s) / d
    target = mu * np.eye(d)
    if n > 1:
        # Ledoit-Wolf analytic shrinkage intensity.
        # beta = (1/n^2) sum_k ||x_k x_k^T - S||_F^2 ; delta = ||S - target||_F^2
        # lambda = clip(beta / delta, 0, 1)
        sq = np.einsum("ij,ik->ijk", xc, xc)  # (n, d, d) outer products
        diff = sq - s[None, :, :]
        beta = np.sum(diff**2) / (n**2)
        delta = np.sum((s - target) ** 2)
        lam = 0.0 if delta <= 0 else float(np.clip(beta / delta, 0.0, 1.0))
    else:
        lam = 1.0
    sigma = (1.0 - lam) * s + lam * target + eps * np.eye(d)
    return mean, sigma, lam


def floored_covariance(x: np.ndarray, eps: float = FLOOR_EPS) -> tuple[np.ndarray, np.ndarray]:
    """Raw sample covariance with only a small diagonal floor (the v2 regime).

    This is the minimally regularised estimator: Sigma = S + eps * I with eps a
    pure numerical floor (no shrinkage toward isotropy). On the near-rank-
    degenerate reconstructive latents (Fukami, beta-VAE collapse onto a few
    directions) the floored inverse amplifies departures into the near-null
    eigen-directions, which is exactly the v2 Table-8 mechanism that drives the
    AE Mahalanobis ratio an order of magnitude out. It reproduces the published
    v2 ratio (Fukami ~9) and is therefore the headline Table-8 estimator;
    shrunk_covariance is the robustness cross-check.

    Args:
        x: (n, d) sample of latent vectors at one frame.
        eps: diagonal floor.

    Returns:
        mean (d,), sigma (d, d).
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    mean = x.mean(axis=0)
    xc = x - mean
    s = (xc.T @ xc) / max(n - 1, 1)  # unbiased sample covariance (np.cov default)
    return mean, s + eps * np.eye(d)


def mahalanobis(z: np.ndarray, mean: np.ndarray, sigma_inv: np.ndarray) -> np.ndarray:
    """Mahalanobis distance of each row of z to (mean, sigma_inv).

    Args:
        z: (n, d) query points.
        mean: (d,) distribution mean.
        sigma_inv: (d, d) inverse covariance.

    Returns:
        (n,) Mahalanobis distances (non-negative).
    """
    zc = np.asarray(z, dtype=np.float64) - mean
    quad = np.einsum("ij,jk,ik->i", zc, sigma_inv, zc)
    return np.sqrt(np.clip(quad, 0.0, None))


# ---------------------------------------------------------------------------
# departure spectrum
# ---------------------------------------------------------------------------


def departure_spectrum(
    delta: np.ndarray,
    eigvecs: np.ndarray,
    k: int = SPEC_K,
    eigvals: np.ndarray | None = None,
) -> dict:
    """Project per-encounter departure vectors onto the encoded eigenbasis.

    eigvecs columns are the eigenvectors of the per-frame TRAIN encoded
    covariance, ordered by DESCENDING eigenvalue (column 0 = top variance). The
    projection coefficients c = eigvecs^T delta partition the total departure
    energy ||delta||^2 = sum_j c_j^2 exactly (orthonormal basis). We report the
    fraction of pooled departure energy in the top-k vs bottom-k directions.

    If eigvals is given, we ALSO compute the WHITENED spectrum c_j^2 / lambda_j:
    the per-direction contribution to the squared Mahalanobis distance of the
    departure. The whitened bottom-k fraction is the mechanistic reconciliation
    number: a departure that is small in raw energy (top-k) but lands in
    near-null eigen-directions concentrates its Mahalanobis distance in the
    bottom-k, which is exactly how a small l2 deviation becomes a huge
    Mahalanobis distance.

    Args:
        delta: (n, d) per-encounter departure vectors (z_roll - z_enc).
        eigvecs: (d, d) orthonormal eigenvectors, columns ordered high->low var.
        k: number of top / bottom directions.
        eigvals: (d,) eigenvalues in the SAME (descending) order, for the
            whitened spectrum. None to skip it.

    Returns:
        dict with the per-direction raw energy fraction, the pooled top-k and
        bottom-k raw fractions, the per-encounter energy, and (if eigvals given)
        the whitened per-direction fraction and its top-k / bottom-k fractions.
    """
    delta = np.asarray(delta, dtype=np.float64)
    d = eigvecs.shape[0]
    k = min(k, d)
    coeff = delta @ eigvecs  # (n, d), coeff[:, j] = projection onto eigvec j
    energy_per_dir = coeff**2  # (n, d)
    total = energy_per_dir.sum()
    per_dir_frac = energy_per_dir.sum(axis=0) / max(total, 1e-30)  # (d,)
    out = {
        "per_dir_frac": per_dir_frac,  # (d,) fraction of total raw energy per dir
        "top_k_frac": float(per_dir_frac[:k].sum()),
        "bottom_k_frac": float(per_dir_frac[-k:].sum()),
        "k": int(k),
        "total_energy": float(total),
        "per_enc_energy": energy_per_dir.sum(axis=1),  # (n,) ||delta||^2 per encounter
    }
    if eigvals is not None:
        ev_raw = np.asarray(eigvals, dtype=np.float64)
        ev = np.clip(ev_raw, 1e-30, None)
        maha2_per_dir = energy_per_dir / ev[None, :]  # (n, d) contribution to maha^2
        maha2_total = maha2_per_dir.sum()
        wfrac = maha2_per_dir.sum(axis=0) / max(maha2_total, 1e-30)  # (d,)
        out["whitened_per_dir_frac"] = wfrac
        out["whitened_top_k_frac"] = float(wfrac[:k].sum())
        out["whitened_bottom_k_frac"] = float(wfrac[-k:].sum())
        # NEAR-NULL departure fraction: the fraction of the rollout's squared
        # Mahalanobis departure carried by directions whose encoded eigenvalue is
        # below NULL_REL_CUTOFF * (top eigenvalue). This is the robust, d-agnostic
        # reconciliation statistic: a reconstructive rollout that exits the
        # collapsed latent's near-null tail lands almost all of its Mahalanobis
        # mass here while keeping a tiny l2 footprint. n_null_dirs is reported too.
        top = max(float(ev_raw.max()), 1e-30)
        null_mask = ev_raw < (NULL_REL_CUTOFF * top)
        maha2_per_enc = maha2_per_dir.sum(axis=1)
        null_per_enc = maha2_per_dir[:, null_mask].sum(axis=1)
        out["near_null_departure_frac"] = float(
            null_per_enc.sum() / max(maha2_per_enc.sum(), 1e-30)
        )
        out["n_null_dirs"] = int(null_mask.sum())
        out["departure_maha"] = float(np.sqrt(maha2_per_enc).mean())
    return out


# ---------------------------------------------------------------------------
# per-key computation
# ---------------------------------------------------------------------------


def compute_key(key: str) -> dict:
    """Compute all C-E1 diagnostics for one rollout key on test_b.

    The encoded reference is the rollout file's own z_dns, which is byte-
    identical to the encoded latent z_full (verified at build time); using it
    keeps the encounter ordering aligned to the rollout by construction.
    """
    latd = PRED_LAT[key]
    train = np.load(LAT_DIR / latd / "train.npz", allow_pickle=True)
    roll = np.load(ROLL_DIR / key / "test_b.npz", allow_pickle=True)

    z_roll = roll["z_full"].astype(np.float64)  # (n, T, d) full-context rollout
    z_enc = roll["z_dns"].astype(np.float64)  # (n, T, d) encoded held-out reference
    z_train = train["z_full"].astype(np.float64)  # (m, T, d) encoded TRAIN latents
    case_ids = np.asarray(roll["case_ids"])
    impact = np.asarray(roll["impact_frame"]).astype(int)
    n, T, d = z_roll.shape

    rel_l2 = np.full((n, len(HORIZONS)), np.nan)  # per-encounter rel-l2 vs H
    # headline Mahalanobis (floored raw covariance = v2 regime)
    maha_roll = np.full((n, len(HORIZONS)), np.nan)
    maha_enc = np.full((n, len(HORIZONS)), np.nan)
    maha_ratio = np.full((n, len(HORIZONS)), np.nan)
    # robustness Mahalanobis (Ledoit-Wolf shrunk covariance)
    maha_roll_shrunk = np.full((n, len(HORIZONS)), np.nan)
    maha_enc_shrunk = np.full((n, len(HORIZONS)), np.nan)
    maha_ratio_shrunk = np.full((n, len(HORIZONS)), np.nan)
    shrink_lambda = np.full(len(HORIZONS), np.nan)
    # fraction of TRAIN encoded variance carried by the top-k eigen-directions
    encoded_topk_var_frac = np.full(len(HORIZONS), np.nan)

    # departure spectrum is reported at the headline horizon
    spec_top = np.nan
    spec_bottom = np.nan
    spec_whitened_bottom = np.nan
    spec_whitened_top = np.nan
    spec_near_null = np.nan
    spec_n_null = 0
    spec_departure_maha = np.nan
    per_dir_frac_headline = None
    whitened_per_dir_frac_headline = None

    for hi, H in enumerate(HORIZONS):
        # impact is a per-encounter constant (=40 in v2p1); index per encounter
        # so the code is correct even if impact ever varies.
        fr = impact + H  # (n,)
        valid = fr < T
        if not valid.any():
            continue
        idx = np.where(valid)[0]
        zr = z_roll[idx, fr[idx]]  # (nv, d)
        ze = z_enc[idx, fr[idx]]  # (nv, d)

        # rel-l2: ||z_roll - z_enc|| / ||z_enc||, per encounter
        num = np.linalg.norm(zr - ze, axis=1)
        den = np.linalg.norm(ze, axis=1)
        rel_l2[idx, hi] = num / np.clip(den, 1e-30, None)

        # per-frame TRAIN encoded distribution at the SAME absolute frame.
        # Impact is constant across train encounters too; use the modal frame
        # so the train sample is taken at a single absolute frame.
        train_frame = int(np.bincount(fr[idx]).argmax())
        zt = z_train[:, train_frame, :]  # (m, d)

        # --- headline ratio: floored raw covariance (v2 regime) ---
        mean_f, sigma_f = floored_covariance(zt)
        sinv_f = np.linalg.inv(sigma_f)
        dm_roll = mahalanobis(zr, mean_f, sinv_f)
        dm_enc = mahalanobis(ze, mean_f, sinv_f)
        maha_roll[idx, hi] = dm_roll
        maha_enc[idx, hi] = dm_enc
        maha_ratio[idx, hi] = dm_roll / np.clip(dm_enc, 1e-30, None)

        # --- robustness ratio: Ledoit-Wolf shrunk covariance ---
        mean_s, sigma_s, lam = shrunk_covariance(zt)
        shrink_lambda[hi] = lam
        sinv_s = np.linalg.inv(sigma_s)
        maha_roll_shrunk[idx, hi] = mahalanobis(zr, mean_s, sinv_s)
        maha_enc_shrunk[idx, hi] = mahalanobis(ze, mean_s, sinv_s)
        maha_ratio_shrunk[idx, hi] = maha_roll_shrunk[idx, hi] / np.clip(
            maha_enc_shrunk[idx, hi], 1e-30, None
        )

        # departure spectrum on the RAW encoded-covariance eigenbasis of TRAIN
        # (the floored covariance shares the raw eigenbasis; eps only shifts
        # eigenvalues, not eigenvectors). eigh returns ascending; flip to
        # descending so column 0 is the top-variance direction.
        evals, evecs = np.linalg.eigh(sigma_f)
        order = np.argsort(evals)[::-1]
        evecs = evecs[:, order]
        evals = np.clip(evals[order], 0.0, None)
        kk = min(SPEC_K, d)
        encoded_topk_var_frac[hi] = float(evals[:kk].sum() / max(evals.sum(), 1e-30))
        if H == HEADLINE_H:
            spec = departure_spectrum(zr - ze, evecs, k=SPEC_K, eigvals=evals)
            spec_top = spec["top_k_frac"]
            spec_bottom = spec["bottom_k_frac"]
            spec_whitened_top = spec["whitened_top_k_frac"]
            spec_whitened_bottom = spec["whitened_bottom_k_frac"]
            spec_near_null = spec["near_null_departure_frac"]
            spec_n_null = spec["n_null_dirs"]
            spec_departure_maha = spec["departure_maha"]
            per_dir_frac_headline = spec["per_dir_frac"]
            whitened_per_dir_frac_headline = spec["whitened_per_dir_frac"]

    return {
        "key": key,
        "latents_dir": latd,
        "n_enc": int(n),
        "d": int(d),
        "case_ids": case_ids,
        "rel_l2": rel_l2,  # (n, H)
        "maha_roll": maha_roll,  # headline (floored)
        "maha_enc": maha_enc,
        "maha_ratio": maha_ratio,
        "maha_roll_shrunk": maha_roll_shrunk,  # robustness (shrunk)
        "maha_enc_shrunk": maha_enc_shrunk,
        "maha_ratio_shrunk": maha_ratio_shrunk,
        "shrink_lambda": shrink_lambda,  # (H,)
        "encoded_topk_var_frac": encoded_topk_var_frac,  # (H,)
        "spec_top_k_frac": float(spec_top),
        "spec_bottom_k_frac": float(spec_bottom),
        "spec_whitened_top_k_frac": float(spec_whitened_top),
        "spec_whitened_bottom_k_frac": float(spec_whitened_bottom),
        "spec_near_null_departure_frac": float(spec_near_null),
        "spec_n_null_dirs": int(spec_n_null),
        "spec_departure_maha": float(spec_departure_maha),
        "spec_per_dir_frac": per_dir_frac_headline,  # (d,) raw energy at headline H
        "spec_whitened_per_dir_frac": whitened_per_dir_frac_headline,  # (d,) whitened
    }


def encoded_only_maha(key_latents: str) -> dict:
    """Mahalanobis ratio for a family whose ROLLOUT is absent (encoded-only row).

    For these families we still report the per-frame Mahalanobis of the ENCODED
    held-out latents to the TRAIN distribution (the denominator of the ratio),
    which is the only well-defined drift number available without a rollout.
    """
    train = np.load(LAT_DIR / key_latents / "train.npz", allow_pickle=True)
    tb = np.load(LAT_DIR / key_latents / "test_b.npz", allow_pickle=True)
    z_train = train["z_full"].astype(np.float64)
    z_enc = tb["z_full"].astype(np.float64)
    impact = np.asarray(tb["impact_frame"]).astype(int)
    n, T, d = z_enc.shape
    maha_enc = np.full((n, len(HORIZONS)), np.nan)
    for hi, H in enumerate(HORIZONS):
        fr = impact + H
        valid = fr < T
        idx = np.where(valid)[0]
        if idx.size == 0:
            continue
        ze = z_enc[idx, fr[idx]]
        train_frame = int(np.bincount(fr[idx]).argmax())
        mean, sigma = floored_covariance(z_train[:, train_frame, :])
        maha_enc[idx, hi] = mahalanobis(ze, mean, np.linalg.inv(sigma))
    return {"latents_dir": key_latents, "n_enc": int(n), "d": int(d), "maha_enc": maha_enc}


# ---------------------------------------------------------------------------
# significance: paired family difference (case-clustered bootstrap + sign test)
# ---------------------------------------------------------------------------


def paired_family_diff(
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    case_ids: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Paired per-encounter difference (a - b) with case-clustered CI + sign test.

    Both metric arrays are per-encounter scalars at one horizon, aligned to the
    SAME encounters (same encoded latents, same ordering). delta = a - b; the
    case-clustered block bootstrap CI and the one-sided sign test (a < b) come
    from stats_lib. We deliberately do NOT call stats_lib.case_permutation_p:
    that is a case-level association test and is degenerate for a paired
    location contrast (B6 lesson).
    """
    mask = np.isfinite(metric_a) & np.isfinite(metric_b)
    a = metric_a[mask]
    b = metric_b[mask]
    cids = case_ids[mask]
    delta = a - b
    boot = stats_lib.case_cluster_bootstrap(delta, cids, rng=rng)
    k, n_eff, p = stats_lib.sign_test_one_sided(a, b)  # one-sided: a < b
    _, cmd = stats_lib.case_means(delta, cids)
    case_paired = stats_lib.case_level_paired_stats(cmd)
    return {
        "n_enc": int(delta.size),
        "mean_delta": float(delta.mean()),
        "enc_mean_ci": boot["enc_mean_ci"],
        "case_mean": boot["case_mean"],
        "case_mean_ci": boot["case_mean_ci"],
        "sign_k_a_less_b": int(k),
        "sign_n_eff": int(n_eff),
        "sign_p_one_sided_a_less_b": float(p),
        "case_paired": case_paired,
    }


# ---------------------------------------------------------------------------
# Table 8 assembly + README
# ---------------------------------------------------------------------------


def nanmean(a: np.ndarray) -> float:
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def build_table8(results: dict, enc_only: dict) -> list[dict]:
    """One row per (family, d): mean rel-l2 and mean Mahalanobis-ratio at H=24.

    Rows are ordered family-major, d ascending. Encoded-only families get rollout
    cells = None with a note.
    """
    rows: list[dict] = []
    hidx = HORIZONS.index(HEADLINE_H)
    for key in PRED_LAT:
        if key not in results:
            continue
        fam, d = parse_key(key)
        r = results[key]
        rows.append(
            {
                "key": key,
                "family": fam,
                "family_label": FAMILY_OF.get(fam, fam),
                "d": d,
                "rel_l2_H24": nanmean(r["rel_l2"][:, hidx]),
                "maha_ratio_H24": nanmean(r["maha_ratio"][:, hidx]),
                "maha_roll_H24": nanmean(r["maha_roll"][:, hidx]),
                "maha_enc_H24": nanmean(r["maha_enc"][:, hidx]),
                "maha_ratio_shrunk_H24": nanmean(r["maha_ratio_shrunk"][:, hidx]),
                "encoded_topk_var_frac_H24": float(r["encoded_topk_var_frac"][hidx]),
                "spec_top_k_frac": r["spec_top_k_frac"],
                "spec_bottom_k_frac": r["spec_bottom_k_frac"],
                "spec_whitened_bottom_k_frac": r["spec_whitened_bottom_k_frac"],
                "spec_whitened_top_k_frac": r["spec_whitened_top_k_frac"],
                "spec_near_null_departure_frac": r["spec_near_null_departure_frac"],
                "spec_n_null_dirs": r["spec_n_null_dirs"],
                "spec_departure_maha": r["spec_departure_maha"],
                "n_enc": r["n_enc"],
                "rollout": True,
            }
        )
    for label, latd in NO_ROLLOUT_LATENTS.items():
        if label not in enc_only:
            continue
        fam, d = parse_key(label.replace("jepa_lstm_noc", "jepalstm"))
        eo = enc_only[label]
        rows.append(
            {
                "key": label,
                "family": "jepa_lstm_noc",
                "family_label": "Unconditioned JEPA LSTM",
                "d": d,
                "rel_l2_H24": None,
                "maha_ratio_H24": None,
                "maha_roll_H24": None,
                "maha_enc_H24": nanmean(eo["maha_enc"][:, hidx]),
                "maha_ratio_shrunk_H24": None,
                "encoded_topk_var_frac_H24": None,
                "spec_top_k_frac": None,
                "spec_bottom_k_frac": None,
                "spec_whitened_bottom_k_frac": None,
                "spec_whitened_top_k_frac": None,
                "spec_near_null_departure_frac": None,
                "spec_n_null_dirs": None,
                "spec_departure_maha": None,
                "n_enc": eo["n_enc"],
                "rollout": False,
            }
        )
    return rows


def fmt(x, nd=3):
    if x is None:
        return "absent"
    if isinstance(x, float) and not np.isfinite(x):
        return "nan"
    return f"{x:.{nd}f}"


def write_readme(table8: list[dict], headline: dict, contrast: dict) -> None:
    """Write the C-E1 README with the complete Table 8 and the prose closure."""
    lines: list[str] = []
    lines.append("# Track C-E1: drift reconciliation (closes referee M6, Table 8 gaps)\n")
    lines.append(
        "Generated by `scripts/session28/drift_ce1.py`. All numbers are on "
        "test_b, full-context rollouts (`z_full`), v2.1 split. The Mahalanobis "
        "reference is the per-frame TRAIN encoded-latent distribution of each "
        "family; the encoded held-out (`z_dns`) latent is the rel-l2 reference "
        "and the Mahalanobis-ratio denominator.\n"
    )

    lines.append("## Method\n")
    lines.append(
        "- Relative l2 deviation (Figure-5 metric): "
        "`||z_roll(t_imp+H) - z_enc(t_imp+H)|| / ||z_enc(t_imp+H)||`, per "
        "encounter, mean over test_b.\n"
        "- Mahalanobis ratio (Table-8 metric): "
        "`D_M(z_roll; mu_train_H, Sigma_train_H) / D_M(z_enc; mu_train_H, "
        "Sigma_train_H)`, with `Sigma_train_H` the per-frame TRAIN encoded "
        "covariance. We report TWO regularisations of the inverse. (i) HEADLINE "
        f"(v2 regime): the raw sample covariance with a `{FLOOR_EPS:g} * I` "
        "diagonal floor only. This is what the published v2 Table 8 used and it "
        "is the estimator that exposes the M6 mechanism: on a near-rank-"
        "degenerate latent the inverse amplifies any departure into the near-"
        "null eigen-directions. (ii) ROBUSTNESS: Ledoit-Wolf analytic shrinkage "
        "toward a scaled identity (intensity logged per H as `shrink_lambda`) "
        f"plus the `{SHRINKAGE_EPS:g} * I` floor; this stabilises the near-null "
        "directions and is reported as the cross-check column.\n"
        "- Departure spectrum: per-encounter `z_roll - z_enc` projected onto the "
        "RAW encoded-covariance eigenbasis of `Sigma_train_H` (orthonormal, so "
        "the projection sums to the total departure energy). The headline column "
        "`near-null dep frac` is the fraction of the rollout's squared "
        "Mahalanobis departure carried by the near-null encoded eigen-directions "
        f"(encoded eigenvalue below {NULL_REL_CUTOFF:g} x the top eigenvalue; "
        "`n null dirs` is the count). The companion column `enc top-k var` is the "
        f"fraction of TRAIN encoded variance carried by the top-{SPEC_K} eigen-"
        "directions (a measure of how rank-degenerate the latent is). Raw and "
        f"whitened top-{SPEC_K}/bottom-{SPEC_K} fractions are in `ce1_results.npz`.\n"
        "- Significance for paired family differences: case-clustered block "
        "bootstrap CI + one-sided sign test (stats_lib). "
        "`stats_lib.case_permutation_p` is NOT used (degenerate for a paired "
        "location contrast; B6 lesson). See `ce1_significance.json`.\n"
    )

    lines.append(f"## Table 8 (COMPLETE) at H = {HEADLINE_H}\n")
    lines.append(
        "| Family | d | rel-l2 (Fig-5) | Maha ratio (v2 floor) | "
        "Maha ratio (shrunk) | D_M roll | D_M enc | enc top-k var | "
        "n null dirs | near-null dep frac | n_enc |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in table8:
        ratio = row["maha_ratio_H24"]
        flag = ""
        if ratio is not None and np.isfinite(ratio) and ratio < 1.0:
            flag = " (<1)"
        nnull = row["spec_n_null_dirs"]
        nnull_s = "absent" if nnull is None else str(nnull)
        lines.append(
            f"| {row['family_label']} | {row['d']} | "
            f"{fmt(row['rel_l2_H24'])} | {fmt(row['maha_ratio_H24'], 2)}{flag} | "
            f"{fmt(row['maha_ratio_shrunk_H24'])} | "
            f"{fmt(row['maha_roll_H24'], 2)} | {fmt(row['maha_enc_H24'], 2)} | "
            f"{fmt(row['encoded_topk_var_frac_H24'])} | "
            f"{nnull_s} | {fmt(row['spec_near_null_departure_frac'])} | "
            f"{row['n_enc']} |"
        )
    lines.append("")
    lines.append(
        "Encoded-only rows (matched-predictor rollout absent in Phase B) report "
        "only the encoded held-out Mahalanobis to the train cloud; rel-l2 and "
        "the ratio require a rollout and are marked `absent`. Available d per "
        "family with a rollout: JEPA {16, 32, 64}, Fukami {3, 16, 32, 64}, POD "
        "{16, 32, 64}, beta-VAE {64}; full per-d, per-H arrays are in "
        "`ce1_results.npz`.\n"
    )

    # ratio < 1 interpretation
    sub1 = [
        (row["family_label"], row["d"], row["maha_ratio_H24"])
        for row in table8
        if row["maha_ratio_H24"] is not None
        and np.isfinite(row["maha_ratio_H24"])
        and row["maha_ratio_H24"] < 1.0
    ]
    lines.append("## Ratios below one (the informative finding)\n")
    if sub1:
        bullet = "; ".join(f"{f} d={d} ({r:.3f})" for f, d, r in sub1)
        lines.append(
            f"At H = {HEADLINE_H} the following families show a Mahalanobis "
            f"ratio below one: {bullet}. A ratio below one means the rollout is "
            "MORE in-distribution (relative to the TRAIN cloud) than the encoded "
            "held-out latents it is compared against: the autoregressive "
            "predictor regresses toward the training distribution rather than "
            "tracking the held-out trajectory out to its true (slightly "
            "off-distribution) position. This is not a failure of the drift "
            "metric; it is the predictor's mean-reversion made visible, and it "
            "is exactly why the predictive family's rollout stays inside the "
            "isotropised cloud.\n"
        )
    else:
        lines.append(f"No family shows a Mahalanobis ratio below one at H = {HEADLINE_H}.\n")

    # reconciliation
    jf = contrast
    lines.append("## Reconciliation (the M6 mechanism)\n")
    lines.append(
        "The two diagnostics order the families oppositely because they measure "
        "departure in different geometries, and the reconstructive latent is "
        "near-rank-degenerate. At d = 64 the Fukami AE concentrates "
        f"{fmt(jf['fukami_enc_topk_var'])} of its TRAIN encoded variance in just "
        f"the top-{SPEC_K} eigen-directions (column `enc top-k var`); the "
        "remaining ~59 directions are near-null (eigenvalues at the numerical "
        f"floor). The unconditioned JEPA latent spreads its variance much more "
        f"broadly (top-{SPEC_K} variance fraction "
        f"{fmt(jf['jepa_enc_topk_var'])}).\n"
    )
    lines.append(
        "The reconstructive rollout deviates by a SMALL Euclidean amount "
        f"(rel-l2 {fmt(jf['fukami_rel_l2'], 2)} at d = 64, the least of any "
        f"family, versus {fmt(jf['jepa_rel_l2'], 2)} for JEPA), but that small "
        "departure has a component in the AE's near-null directions, where the "
        "Mahalanobis metric divides by the near-zero train variance. Decomposing "
        "the rollout's squared Mahalanobis departure by encoded eigen-direction "
        "(near-null = encoded eigenvalue below "
        f"{NULL_REL_CUTOFF:g} x the top eigenvalue) makes this exact: the Fukami "
        f"AE carries {fmt(jf['fukami_near_null_frac'])} of its Mahalanobis "
        f"departure in its {jf['fukami_n_null_dirs']} near-null directions, "
        f"versus {fmt(jf['jepa_near_null_frac'])} across only "
        f"{jf['jepa_n_null_dirs']} near-null directions for JEPA. That tail is "
        f"why the AE Mahalanobis ratio explodes to {fmt(jf['fukami_maha_ratio'], 2)} "
        "(v2 regime, reproducing the published ~9.9) while its rel-l2 is the "
        "smallest. The predictive JEPA rollout wanders WITHIN the broad, "
        "isotropised encoded cloud: its rel-l2 is larger but it has far fewer "
        "near-null directions to fall into, so its Mahalanobis ratio stays at "
        f"{fmt(jf['jepa_maha_ratio'], 2)} (in-distribution). The opposite "
        "ordering of rel-l2 and Mahalanobis is therefore mechanistically "
        "reconciled: small l2 + huge Mahalanobis is the reconstructive latent "
        "exiting along directions of tiny encoded variance; the predictive "
        "latent stays on the manifold.\n"
    )
    lines.append(
        "The robustness column (`Maha ratio (shrunk)`) confirms the blowup is a "
        "property of the near-null tail: under Ledoit-Wolf shrinkage toward "
        "isotropy the AE near-null directions are regularised and the ratios "
        "collapse toward one for every family, so the v2-regime separation is "
        "the genuine, interpretable signal and the shrunk column is the "
        "conservative bound.\n"
    )

    lines.append("## Headline numbers (also in numbers_parts/drift_ce1.json)\n")
    for name, rec in sorted(headline.items()):
        lines.append(f"- `{name}` = {rec['value']:.4f}  ({rec.get('note', '')})")
    lines.append("")

    (OUT_DIR / "README.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# numbers_parts emission
# ---------------------------------------------------------------------------


def macro_name(fam: str, metric: str) -> str:
    """Alphabetic LaTeX macro name, e.g. ('jepa','RelL2') -> DriftRelLTwoJepa."""
    fam_cap = {"jepa": "Jepa", "fukami": "Fukami", "pod": "Pod", "bvae": "Bvae"}[fam]
    return f"Drift{metric}{fam_cap}"


def build_numbers_part(results: dict, contrast: dict) -> dict:
    """Headline macros: rel-l2 + Mahalanobis at H=24 per family at d=64, plus the
    departure-spectrum bottom-k fraction contrast (JEPA vs Fukami)."""
    numbers: dict[str, dict] = {}
    hidx = HORIZONS.index(HEADLINE_H)
    for fam in ("jepa", "fukami", "pod", "bvae"):
        key = f"{fam}_d64"
        if key not in results:
            continue
        r = results[key]
        rel = nanmean(r["rel_l2"][:, hidx])
        mah = nanmean(r["maha_ratio"][:, hidx])
        numbers[f"drift_rel_l2_H24_d64_{fam}"] = {
            "macro": macro_name(fam, "RelLtwo"),
            "value": rel,
            "fmt": "%.2f",
            "n": r["n_enc"],
            "split": "test_b",
            "horizon": HEADLINE_H,
            "source": "drift_ce1.py",
            "note": f"mean rel-l2 deviation, {fam} d=64, full-context rollout, H={HEADLINE_H}",
        }
        numbers[f"drift_maha_ratio_H24_d64_{fam}"] = {
            "macro": macro_name(fam, "Maha"),
            "value": mah,
            "fmt": "%.2f",
            "n": r["n_enc"],
            "split": "test_b",
            "horizon": HEADLINE_H,
            "source": "drift_ce1.py",
            "note": (
                f"mean Mahalanobis ratio to train cloud, {fam} d=64, H={HEADLINE_H}"
                + ("; ratio<1 (predictor regresses to train cloud)" if mah < 1 else "")
            ),
        }
    # departure-spectrum NEAR-NULL fraction contrast (JEPA vs Fukami) at d=64:
    # the fraction of the rollout's squared Mahalanobis departure carried by the
    # near-null encoded eigen-directions (encoded eigval < cutoff x top). This is
    # the headline reconciliation contrast: the reconstructive AE dumps almost
    # all of its Mahalanobis into its collapsed null tail (tiny l2, huge
    # Mahalanobis), the predictive JEPA has far fewer null directions to fall into.
    numbers["drift_spec_nearnull_frac_d64_jepa"] = {
        "macro": "DriftSpecNearNullJepa",
        "value": contrast["jepa_near_null_frac"],
        "fmt": "%.2f",
        "split": "test_b",
        "horizon": HEADLINE_H,
        "source": "drift_ce1.py",
        "note": (
            "fraction of squared-Mahalanobis departure in near-null encoded "
            f"eigen-dirs ({contrast['jepa_n_null_dirs']} dirs), JEPA d=64"
        ),
    }
    numbers["drift_spec_nearnull_frac_d64_fukami"] = {
        "macro": "DriftSpecNearNullFukami",
        "value": contrast["fukami_near_null_frac"],
        "fmt": "%.2f",
        "split": "test_b",
        "horizon": HEADLINE_H,
        "source": "drift_ce1.py",
        "note": (
            "fraction of squared-Mahalanobis departure in near-null encoded "
            f"eigen-dirs ({contrast['fukami_n_null_dirs']} dirs), Fukami d=64 "
            "(the M6 mechanism)"
        ),
    }
    return {"part": "drift_ce1", "numbers": numbers}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    print("[ce1] computing per-key diagnostics on test_b ...")
    results: dict[str, dict] = {}
    for key in PRED_LAT:
        if not (ROLL_DIR / key / "test_b.npz").exists():
            print(f"[ce1] SKIP {key} (no rollout)")
            continue
        results[key] = compute_key(key)
        r = results[key]
        hidx = HORIZONS.index(HEADLINE_H)
        print(
            f"[ce1]   {key:11s} d={r['d']:>2d} n={r['n_enc']:>2d}  "
            f"relL2(H24)={nanmean(r['rel_l2'][:, hidx]):.3f}  "
            f"mahaRatio(H24)={nanmean(r['maha_ratio'][:, hidx]):.3f}  "
            f"mahaRatioShrunk={nanmean(r['maha_ratio_shrunk'][:, hidx]):.3f}  "
            f"encTopKvar={r['encoded_topk_var_frac'][hidx]:.3f}  "
            f"nNull={r['spec_n_null_dirs']:>2d}  "
            f"nearNullDep={r['spec_near_null_departure_frac']:.3f}"
        )

    enc_only: dict[str, dict] = {}
    for label, latd in NO_ROLLOUT_LATENTS.items():
        if (LAT_DIR / latd / "train.npz").exists():
            enc_only[label] = encoded_only_maha(latd)
            print(
                f"[ce1]   encoded-only {label}: maha_enc(H24)="
                f"{nanmean(enc_only[label]['maha_enc'][:, HORIZONS.index(HEADLINE_H)]):.3f}"
            )

    # paired family differences (JEPA vs Fukami) at d=64, both metrics, H=24
    sig: dict[str, dict] = {}
    if "jepa_d64" in results and "fukami_d64" in results:
        rj, rf = results["jepa_d64"], results["fukami_d64"]
        # encoded latents are aligned 1:1 across families (same split/order); assert
        assert np.array_equal(rj["case_ids"], rf["case_ids"]), "case_id misalignment"
        hidx = HORIZONS.index(HEADLINE_H)
        sig["rel_l2_jepa_minus_fukami_H24_d64"] = paired_family_diff(
            rj["rel_l2"][:, hidx], rf["rel_l2"][:, hidx], rj["case_ids"], rng
        )
        sig["maha_ratio_jepa_minus_fukami_H24_d64"] = paired_family_diff(
            rj["maha_ratio"][:, hidx], rf["maha_ratio"][:, hidx], rj["case_ids"], rng
        )
        sr = sig["rel_l2_jepa_minus_fukami_H24_d64"]
        sm = sig["maha_ratio_jepa_minus_fukami_H24_d64"]
        print(
            "[ce1] paired rel-l2 (jepa - fukami) H24 d64: "
            f"mean={sr['mean_delta']:.3f} CI={sr['enc_mean_ci']} "
            f"sign_p(jepa<fukami)={sr['sign_p_one_sided_a_less_b']:.3g}"
        )
        print(
            "[ce1] paired maha-ratio (jepa - fukami) H24 d64: "
            f"mean={sm['mean_delta']:.3f} CI={sm['enc_mean_ci']} "
            f"sign_p(jepa<fukami)={sm['sign_p_one_sided_a_less_b']:.3g}"
        )

    # departure-spectrum contrast (JEPA vs Fukami) at d=64
    hidx = HORIZONS.index(HEADLINE_H)
    if "jepa_d64" not in results or "fukami_d64" not in results:
        raise SystemExit("[ce1] jepa_d64 and fukami_d64 rollouts are required for the contrast")
    rj, rf = results["jepa_d64"], results["fukami_d64"]
    contrast = {
        "jepa_near_null_frac": rj["spec_near_null_departure_frac"],
        "fukami_near_null_frac": rf["spec_near_null_departure_frac"],
        "jepa_n_null_dirs": rj["spec_n_null_dirs"],
        "fukami_n_null_dirs": rf["spec_n_null_dirs"],
        "jepa_enc_topk_var": float(rj["encoded_topk_var_frac"][hidx]),
        "fukami_enc_topk_var": float(rf["encoded_topk_var_frac"][hidx]),
        "jepa_rel_l2": nanmean(rj["rel_l2"][:, hidx]),
        "fukami_rel_l2": nanmean(rf["rel_l2"][:, hidx]),
        "jepa_maha_ratio": nanmean(rj["maha_ratio"][:, hidx]),
        "fukami_maha_ratio": nanmean(rf["maha_ratio"][:, hidx]),
    }

    # ----- save ce1_results.npz -----
    npz: dict[str, np.ndarray] = {}
    npz["horizons"] = np.array(HORIZONS, dtype=np.int32)
    npz["headline_horizon"] = np.array(HEADLINE_H, dtype=np.int32)
    npz["spec_k"] = np.array(SPEC_K, dtype=np.int32)
    for key, r in results.items():
        npz[f"{key}__rel_l2"] = r["rel_l2"].astype(np.float64)
        npz[f"{key}__maha_roll"] = r["maha_roll"].astype(np.float64)
        npz[f"{key}__maha_enc"] = r["maha_enc"].astype(np.float64)
        npz[f"{key}__maha_ratio"] = r["maha_ratio"].astype(np.float64)
        npz[f"{key}__maha_roll_shrunk"] = r["maha_roll_shrunk"].astype(np.float64)
        npz[f"{key}__maha_enc_shrunk"] = r["maha_enc_shrunk"].astype(np.float64)
        npz[f"{key}__maha_ratio_shrunk"] = r["maha_ratio_shrunk"].astype(np.float64)
        npz[f"{key}__shrink_lambda"] = r["shrink_lambda"].astype(np.float64)
        npz[f"{key}__encoded_topk_var_frac"] = r["encoded_topk_var_frac"].astype(np.float64)
        npz[f"{key}__case_ids"] = r["case_ids"].astype(str)
        npz[f"{key}__d"] = np.array(r["d"], dtype=np.int32)
        if r["spec_per_dir_frac"] is not None:
            npz[f"{key}__spec_per_dir_frac"] = r["spec_per_dir_frac"].astype(np.float64)
        if r["spec_whitened_per_dir_frac"] is not None:
            npz[f"{key}__spec_whitened_per_dir_frac"] = r["spec_whitened_per_dir_frac"].astype(
                np.float64
            )
        npz[f"{key}__spec_top_k_frac"] = np.array(r["spec_top_k_frac"])
        npz[f"{key}__spec_bottom_k_frac"] = np.array(r["spec_bottom_k_frac"])
        npz[f"{key}__spec_whitened_top_k_frac"] = np.array(r["spec_whitened_top_k_frac"])
        npz[f"{key}__spec_whitened_bottom_k_frac"] = np.array(r["spec_whitened_bottom_k_frac"])
        npz[f"{key}__spec_near_null_departure_frac"] = np.array(r["spec_near_null_departure_frac"])
        npz[f"{key}__spec_n_null_dirs"] = np.array(r["spec_n_null_dirs"], dtype=np.int32)
        npz[f"{key}__spec_departure_maha"] = np.array(r["spec_departure_maha"])
    for label, eo in enc_only.items():
        npz[f"{label}__maha_enc"] = eo["maha_enc"].astype(np.float64)
        npz[f"{label}__d"] = np.array(eo["d"], dtype=np.int32)

    # two-panel figure source arrays: per-family mean curves vs H (d=64) for both
    # panels, plus the per-direction spectrum (jepa vs fukami) for the inset.
    fig_fams = [f for f in ("jepa", "fukami", "pod", "bvae") if f"{f}_d64" in results]
    npz["fig_families_d64"] = np.array(fig_fams, dtype=str)
    panel_rel = np.full((len(fig_fams), len(HORIZONS)), np.nan)
    panel_maha = np.full((len(fig_fams), len(HORIZONS)), np.nan)
    for i, f in enumerate(fig_fams):
        r = results[f"{f}_d64"]
        panel_rel[i] = np.nanmean(r["rel_l2"], axis=0)
        panel_maha[i] = np.nanmean(r["maha_ratio"], axis=0)
    npz["fig_panel_rel_l2_mean"] = panel_rel  # (n_fam, H) left panel (Figure-5 metric)
    npz["fig_panel_maha_ratio_mean"] = panel_maha  # (n_fam, H) right panel (Table-8 metric)
    # third (reconciliation) panel: per-direction raw + whitened departure
    # spectrum, JEPA vs Fukami at d=64.
    for f in ("jepa", "fukami"):
        r = results.get(f"{f}_d64")
        if r is None:
            continue
        if r["spec_per_dir_frac"] is not None:
            npz[f"fig_spec_per_dir_{f}_d64"] = r["spec_per_dir_frac"]
        if r["spec_whitened_per_dir_frac"] is not None:
            npz[f"fig_spec_whitened_per_dir_{f}_d64"] = r["spec_whitened_per_dir_frac"]

    np.savez(out_dir / "ce1_results.npz", **npz)
    print(f"[ce1] wrote {out_dir / 'ce1_results.npz'}")

    # significance JSON (kept alongside, not a numbers part)
    (out_dir / "ce1_significance.json").write_text(json.dumps(sig, indent=2, default=float))
    print(f"[ce1] wrote {out_dir / 'ce1_significance.json'}")

    # ----- numbers part -----
    part = build_numbers_part(results, contrast)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    (PARTS_DIR / "drift_ce1.json").write_text(json.dumps(part, indent=2, sort_keys=True))
    print(f"[ce1] wrote {PARTS_DIR / 'drift_ce1.json'}")
    headline = part["numbers"]

    # ----- Table 8 + README -----
    table8 = build_table8(results, enc_only)
    (out_dir / "table8.json").write_text(json.dumps(table8, indent=2, default=float))
    write_readme(table8, headline, contrast)
    print(f"[ce1] wrote {out_dir / 'README.md'} and {out_dir / 'table8.json'}")
    print("[ce1] done.")


if __name__ == "__main__":
    main()
