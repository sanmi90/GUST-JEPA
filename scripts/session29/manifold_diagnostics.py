"""SESSION29 Track H: metric-independent manifold-departure diagnostics (v2.1).

CONTEXT (referee mechanism-fragility, M6 follow-up). The v2.1 paper's drift
mechanism -- the reconstructive rollout (Fukami observable-augmented AE) leaves
the training manifold while the predictive rollout (unconditioned JEPA
transformer) stays on it -- currently leans on a Mahalanobis ratio
(scripts/session28/drift_ce1.py) that is estimator-fragile: the blow-up lives in
the near-null tail of a near-rank-degenerate covariance and a Ledoit-Wolf shrunk
inverse collapses the ratios toward one. A referee can dismiss the MAGNITUDE.

This module corroborates the mechanism with TWO additional diagnostics that use
NO covariance inverse and therefore have no shrinkage knob:

1. kNN distance to the training manifold: for a query latent z_q, the mean
   Euclidean distance to its k nearest neighbours in the per-frame TRAIN encoded
   point cloud, normalised by the cloud scale (RMS distance of TRAIN points to
   their own centroid). A point off the data manifold is farther from its
   neighbours than a point on it. Purely Euclidean; rotation-invariant.

2. Local-PCA reconstruction residual: fit a low-rank (n_components) PCA on the k
   nearest TRAIN neighbours of z_q, then measure the norm of z_q's component
   ORTHOGONAL to that local tangent subspace, normalised by the cloud scale. A
   point on a locally low-dimensional manifold reconstructs well (small residual);
   a point that has left the manifold has energy in the orthogonal complement.
   Also purely Euclidean.

Both are computed for the d = 64 families jepa / fukami / pod, on the
full-context ROLLOUT latent at frame impact+H (rollout key z_full) against the
per-frame TRAIN encoded cloud at the same modal absolute frame, for H in
{4, 8, 16, 24, 32}. The loading convention (PRED_LAT map, z_full rollout vs z_dns
reference, train cloud at z_train[:, impact+H, :]) is verbatim from drift_ce1.py.

VERDICT. STRONG if the reconstructive (fukami) rollout departs the training
manifold MORE than the predictive (jepa) under BOTH kNN-distance AND local-PCA
residual at the headline horizon -- this corroborates the Mahalanobis ordering
metric-independently and the mechanism is publishable with magnitude language.
WEAK otherwise -- retract the magnitude claim and keep only the departure
direction.

Significance for the paired family difference (fukami - jepa, per encounter)
reuses the case-clustered block bootstrap + one-sided sign test from
scripts/session28/stats_lib.py (the paper's uncertainty convention); we report
fukami > jepa (the reconstructive latent departs more).

Outputs (absolute):
    outputs/session29/manifold_diagnostics.json  (+ _provenance via cm)
    outputs/session29/manifold_diagnostics.md

CPU only; run under nice. Usage:
    python scripts/session29/manifold_diagnostics.py
    python scripts/session29/manifold_diagnostics.py --dry-run
"""

from __future__ import annotations

import argparse

import numpy as np

import _s29_common as cm

REPO = cm.REPO
LAT_DIR = REPO / "outputs" / "session28" / "latents"
ROLL_DIR = REPO / "outputs" / "session28" / "rollouts"
OUT_JSON = REPO / "outputs" / "session29" / "manifold_diagnostics.json"
OUT_MD = REPO / "outputs" / "session29" / "manifold_diagnostics.md"

stats_lib = cm.stats_lib

HORIZONS = (4, 8, 16, 24, 32)
HEADLINE_H = 24
DEFAULT_K = 10
DEFAULT_KPCA = 20  # local-PCA needs enough neighbours to fit n_components
DEFAULT_NCOMP = 5

# Rollout-key -> encoded-latent directory, verbatim from drift_ce1.py PRED_LAT
# for the three d=64 families used here. jepa binds to the UNCONDITIONED
# transformer latents (jepa_tf_noc); fukami/pod bind to their s42/canonical dirs.
PRED_LAT = {
    "jepa_d64": "jepa_tf_noc_d64_s42",
    "fukami_d64": "fukami_d64_s42",
    "pod_d64": "pod_d64",
}

FAMILY_LABEL = {
    "jepa_d64": "Unconditioned JEPA transformer (predictive)",
    "fukami_d64": "Fukami observable-augmented AE (reconstructive)",
    "pod_d64": "POD (linear reconstructive)",
}

# Which family is predictive vs reconstructive for the verdict contrast.
PREDICTIVE_KEY = "jepa_d64"
RECONSTRUCTIVE_KEY = "fukami_d64"


# ---------------------------------------------------------------------------
# metric-independent manifold diagnostics (no covariance inverse)
# ---------------------------------------------------------------------------


def cloud_scale(cloud: np.ndarray) -> float:
    """RMS Euclidean distance of TRAIN points to their own centroid.

    A single, estimator-free length scale that makes the kNN distance and the
    local-PCA residual comparable across families with different latent norms.
    Homogeneous of degree 1 in the data (scaling the cloud by a scales this by a).

    Args:
        cloud: (m, d) training-manifold point cloud.

    Returns:
        Positive scalar scale (>= tiny floor).
    """
    cloud = np.asarray(cloud, dtype=np.float64)
    c = cloud - cloud.mean(axis=0, keepdims=True)
    rms = float(np.sqrt(np.mean(np.sum(c**2, axis=1))))
    return max(rms, 1e-30)


def _knn_indices(query: np.ndarray, cloud: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and Euclidean distances of the k nearest cloud points per query row.

    Brute-force O(n_q * m * d); the clouds here are <= a few hundred points so
    this is fast on CPU and dependency-free (no sklearn neighbour index needed).

    Returns:
        idx (n_q, k) neighbour indices, dist (n_q, k) Euclidean distances
        (ascending per row).
    """
    query = np.asarray(query, dtype=np.float64)
    cloud = np.asarray(cloud, dtype=np.float64)
    m = cloud.shape[0]
    k = min(k, m)
    # (n_q, m) squared distances
    d2 = (
        np.sum(query**2, axis=1, keepdims=True)
        - 2.0 * (query @ cloud.T)
        + np.sum(cloud**2, axis=1)[None, :]
    )
    d2 = np.clip(d2, 0.0, None)
    part = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
    # sort the k partitioned indices by true distance
    rows = np.arange(query.shape[0])[:, None]
    order = np.argsort(d2[rows, part], axis=1)
    idx = part[rows, order]
    dist = np.sqrt(d2[rows, idx])
    return idx, dist


def knn_distance(query: np.ndarray, cloud: np.ndarray, k: int, scale: float) -> np.ndarray:
    """Mean Euclidean distance to the k nearest cloud points, in cloud-scale units.

    Args:
        query: (n_q, d) query latents.
        cloud: (m, d) training-manifold point cloud.
        k: number of nearest neighbours.
        scale: cloud-scale normaliser (cloud_scale(cloud)).

    Returns:
        (n_q,) normalised mean kNN distance per query.
    """
    _, dist = _knn_indices(query, cloud, k)
    return dist.mean(axis=1) / scale


def local_pca_residual(
    query: np.ndarray,
    cloud: np.ndarray,
    k: int,
    n_components: int,
    scale: float,
) -> np.ndarray:
    """Orthogonal reconstruction residual of each query against a LOCAL PCA fit.

    For each query: take its k nearest cloud neighbours, fit a PCA (centred on
    the neighbour mean) keeping n_components principal directions, then measure
    the norm of the query's component orthogonal to that local tangent subspace.
    A query that lies on the locally low-dimensional manifold reconstructs well
    (small residual); a query that has departed has energy in the orthogonal
    complement. Normalised by the cloud scale so families are comparable.

    Args:
        query: (n_q, d) query latents.
        cloud: (m, d) training-manifold point cloud.
        k: neighbour count for the local fit (must exceed n_components).
        n_components: local tangent-subspace dimension.
        scale: cloud-scale normaliser.

    Returns:
        (n_q,) normalised orthogonal residual per query.
    """
    query = np.asarray(query, dtype=np.float64)
    cloud = np.asarray(cloud, dtype=np.float64)
    idx, _ = _knn_indices(query, cloud, k)
    n_q, d = query.shape
    out = np.empty(n_q)
    for i in range(n_q):
        nbrs = cloud[idx[i]]  # (k, d)
        mu = nbrs.mean(axis=0)
        xc = nbrs - mu
        # right singular vectors are the principal directions
        _, _, vt = np.linalg.svd(xc, full_matrices=False)
        nc = min(n_components, vt.shape[0])
        basis = vt[:nc]  # (nc, d) orthonormal tangent basis
        qc = query[i] - mu
        proj = basis.T @ (basis @ qc)  # reconstruction in the tangent subspace
        resid = qc - proj
        out[i] = float(np.linalg.norm(resid)) / scale
    return out


# ---------------------------------------------------------------------------
# per-family computation
# ---------------------------------------------------------------------------


def compute_key(key: str, k: int, kpca: int, ncomp: int) -> dict:
    """Both manifold diagnostics for one rollout key on test_b, per horizon.

    The query is the full-context ROLLOUT latent z_full at frame impact+H; the
    reference manifold is the per-frame TRAIN encoded cloud at the same modal
    absolute frame (verbatim drift_ce1 convention). We ALSO compute the encoded
    held-out (z_dns) diagnostics at the same frame so the rollout values can be
    read against an on-manifold control if desired.
    """
    latd = PRED_LAT[key]
    train = np.load(LAT_DIR / latd / "train.npz", allow_pickle=True)
    roll = np.load(ROLL_DIR / key / "test_b.npz", allow_pickle=True)

    z_roll = roll["z_full"].astype(np.float64)  # (n, T, d)
    z_enc = roll["z_dns"].astype(np.float64)  # (n, T, d) encoded held-out ref
    z_train = train["z_full"].astype(np.float64)  # (m, T, d)
    case_ids = np.asarray(roll["case_ids"]).astype(str)
    impact = np.asarray(roll["impact_frame"]).astype(int)
    n, T, d = z_roll.shape
    m = z_train.shape[0]

    knn_roll = np.full((n, len(HORIZONS)), np.nan)
    knn_enc = np.full((n, len(HORIZONS)), np.nan)
    pca_roll = np.full((n, len(HORIZONS)), np.nan)
    pca_enc = np.full((n, len(HORIZONS)), np.nan)
    scale_h = np.full(len(HORIZONS), np.nan)

    for hi, H in enumerate(HORIZONS):
        fr = impact + H  # (n,)
        valid = fr < T
        if not valid.any():
            continue
        idx = np.where(valid)[0]
        zr = z_roll[idx, fr[idx]]  # (nv, d)
        ze = z_enc[idx, fr[idx]]  # (nv, d)
        train_frame = int(np.bincount(fr[idx]).argmax())
        cloud = z_train[:, train_frame, :]  # (m, d) training-manifold cloud
        scale = cloud_scale(cloud)
        scale_h[hi] = scale

        knn_roll[idx, hi] = knn_distance(zr, cloud, k, scale)
        knn_enc[idx, hi] = knn_distance(ze, cloud, k, scale)
        pca_roll[idx, hi] = local_pca_residual(zr, cloud, kpca, ncomp, scale)
        pca_enc[idx, hi] = local_pca_residual(ze, cloud, kpca, ncomp, scale)

    return {
        "key": key,
        "latents_dir": latd,
        "n_enc": int(n),
        "n_train": int(m),
        "d": int(d),
        "case_ids": case_ids,
        "knn_roll": knn_roll,
        "knn_enc": knn_enc,
        "pca_roll": pca_roll,
        "pca_enc": pca_enc,
        "scale": scale_h,
    }


def median_curve(a: np.ndarray) -> list[float]:
    """Per-horizon median over encounters (ignoring NaN), as a JSON-safe list."""
    out = []
    for hi in range(a.shape[1]):
        col = a[:, hi]
        out.append(float(np.nanmedian(col)) if np.isfinite(col).any() else float("nan"))
    return out


def paired_recon_minus_pred(
    recon: np.ndarray,
    pred: np.ndarray,
    case_ids: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Paired per-encounter (reconstructive - predictive) departure at one horizon.

    Positive delta means the reconstructive rollout departs the manifold MORE.
    Case-clustered block bootstrap CI + one-sided sign test (pred < recon), both
    from stats_lib, matching drift_ce1's paired-difference convention. We do NOT
    use stats_lib.case_permutation_p (degenerate for a paired location contrast).
    """
    mask = np.isfinite(recon) & np.isfinite(pred)
    a = recon[mask]
    b = pred[mask]
    cids = case_ids[mask]
    delta = a - b  # recon - pred; > 0 = reconstructive departs more
    boot = stats_lib.case_cluster_bootstrap(delta, cids, rng=rng)
    # one-sided sign test that pred < recon (b < a) => predictive departs LESS
    k, n_eff, p = stats_lib.sign_test_one_sided(b, a)
    _, cmd = stats_lib.case_means(delta, cids)
    case_paired = stats_lib.case_level_paired_stats(cmd)
    return {
        "n_enc": int(delta.size),
        "median_recon": float(np.median(a)),
        "median_pred": float(np.median(b)),
        "median_delta_recon_minus_pred": float(np.median(delta)),
        "mean_delta_recon_minus_pred": float(delta.mean()),
        "ratio_recon_over_pred": float(np.median(a) / max(np.median(b), 1e-30)),
        "enc_mean_ci": boot["enc_mean_ci"],
        "case_mean": boot["case_mean"],
        "case_mean_ci": boot["case_mean_ci"],
        "sign_k_pred_less_recon": int(k),
        "sign_n_eff": int(n_eff),
        "sign_p_one_sided_pred_less_recon": float(p),
        "case_paired": case_paired,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keys", nargs="+", default=list(PRED_LAT))
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="kNN neighbour count")
    ap.add_argument("--kpca", type=int, default=DEFAULT_KPCA, help="neighbour count for local PCA")
    ap.add_argument("--n-components", type=int, default=DEFAULT_NCOMP, help="local tangent dim")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", default=str(OUT_JSON))
    ap.add_argument("--out-md", default=str(OUT_MD))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.kpca <= args.n_components:
        raise SystemExit(f"--kpca ({args.kpca}) must exceed --n-components ({args.n_components})")

    # fail loud if any required input is missing
    inputs = []
    for key in args.keys:
        if key not in PRED_LAT:
            raise SystemExit(f"unknown key {key!r}; known: {list(PRED_LAT)}")
        latd = PRED_LAT[key]
        tnpz = LAT_DIR / latd / "train.npz"
        rnpz = ROLL_DIR / key / "test_b.npz"
        for p in (tnpz, rnpz):
            if not p.exists():
                raise SystemExit(f"missing required input: {p}")
        inputs += [tnpz, rnpz]

    if args.dry_run:
        print(
            f"[dry-run] keys={args.keys} k={args.k} kpca={args.kpca} "
            f"n_components={args.n_components} horizons={HORIZONS}"
        )
        for key in args.keys:
            latd = PRED_LAT[key]
            roll = np.load(ROLL_DIR / key / "test_b.npz", allow_pickle=True)
            train = np.load(LAT_DIR / latd / "train.npz", allow_pickle=True)
            print(
                f"  {key:11s} -> {latd:22s} roll z_full{roll['z_full'].shape} "
                f"train z_full{train['z_full'].shape} "
                f"impact={int(np.asarray(roll['impact_frame'])[0])}"
            )
        print(f"  would write {args.out_json}")
        print(f"  would write {args.out_md}")
        return

    rng = np.random.default_rng(args.seed)

    print("[manifold] computing metric-independent departure diagnostics on test_b ...")
    results: dict[str, dict] = {}
    for key in args.keys:
        r = compute_key(key, args.k, args.kpca, args.n_components)
        results[key] = r
        hidx = HORIZONS.index(HEADLINE_H)
        print(
            f"[manifold]   {key:11s} d={r['d']:>2d} n={r['n_enc']:>2d} "
            f"m_train={r['n_train']:>3d}  "
            f"knn(H{HEADLINE_H}) roll={np.nanmedian(r['knn_roll'][:, hidx]):.3f} "
            f"enc={np.nanmedian(r['knn_enc'][:, hidx]):.3f}  "
            f"pca(H{HEADLINE_H}) roll={np.nanmedian(r['pca_roll'][:, hidx]):.3f} "
            f"enc={np.nanmedian(r['pca_enc'][:, hidx]):.3f}"
        )

    # per-family per-horizon median curves (rollout query, the headline series)
    per_family: dict[str, dict] = {}
    for key, r in results.items():
        per_family[key] = {
            "family_label": FAMILY_LABEL.get(key, key),
            "d": r["d"],
            "n_enc": r["n_enc"],
            "n_train": r["n_train"],
            "knn_distance_rollout_median": median_curve(r["knn_roll"]),
            "knn_distance_encoded_median": median_curve(r["knn_enc"]),
            "local_pca_residual_rollout_median": median_curve(r["pca_roll"]),
            "local_pca_residual_encoded_median": median_curve(r["pca_enc"]),
            "cloud_scale_per_horizon": [float(x) for x in r["scale"]],
        }

    # reconstructive-vs-predictive separation, per horizon, both diagnostics.
    have_contrast = PREDICTIVE_KEY in results and RECONSTRUCTIVE_KEY in results
    separation: dict[str, dict] = {}
    if have_contrast:
        rp = results[PREDICTIVE_KEY]
        rr = results[RECONSTRUCTIVE_KEY]
        assert np.array_equal(
            rp["case_ids"], rr["case_ids"]
        ), "case_id misalignment between predictive and reconstructive families"
        for diag, roll_key in (("knn_distance", "knn_roll"), ("local_pca_residual", "pca_roll")):
            per_h = {}
            for hi, H in enumerate(HORIZONS):
                per_h[str(H)] = paired_recon_minus_pred(
                    rr[roll_key][:, hi], rp[roll_key][:, hi], rr["case_ids"], rng
                )
            separation[diag] = per_h

    # ---- VERDICT ----
    verdict = None
    if have_contrast:
        hk = str(HEADLINE_H)
        knn_h = separation["knn_distance"][hk]
        pca_h = separation["local_pca_residual"][hk]
        # corroborates if reconstructive departs MORE (median delta > 0) at the
        # headline horizon for the diagnostic.
        knn_corrob = knn_h["median_delta_recon_minus_pred"] > 0
        pca_corrob = pca_h["median_delta_recon_minus_pred"] > 0
        both = knn_corrob and pca_corrob
        branch = "STRONG" if both else "WEAK"
        verdict = {
            "headline_horizon": HEADLINE_H,
            "predictive_family": FAMILY_LABEL[PREDICTIVE_KEY],
            "reconstructive_family": FAMILY_LABEL[RECONSTRUCTIVE_KEY],
            "knn_corroborates_mahalanobis": bool(knn_corrob),
            "local_pca_corroborates_mahalanobis": bool(pca_corrob),
            "both_corroborate": bool(both),
            "branch": branch,
            "knn_median_recon": knn_h["median_recon"],
            "knn_median_pred": knn_h["median_pred"],
            "knn_ratio_recon_over_pred": knn_h["ratio_recon_over_pred"],
            "knn_sign_p_pred_less_recon": knn_h["sign_p_one_sided_pred_less_recon"],
            "pca_median_recon": pca_h["median_recon"],
            "pca_median_pred": pca_h["median_pred"],
            "pca_ratio_recon_over_pred": pca_h["ratio_recon_over_pred"],
            "pca_sign_p_pred_less_recon": pca_h["sign_p_one_sided_pred_less_recon"],
            "note": (
                "STRONG = the reconstructive (fukami) rollout departs the training "
                "manifold MORE than the predictive (jepa) under BOTH kNN-distance "
                "AND local-PCA residual at the headline horizon, corroborating the "
                "Mahalanobis ordering with NO covariance inverse (no shrinkage knob) "
                "=> the mechanism is metric-independent and magnitude language is "
                "defensible. WEAK = only Mahalanobis showed it => retract magnitude "
                "language, claim only the departure direction. CIs are wide at 10 "
                "test_b cases; the robust signal is the SIGN consistency (sign test "
                "pred < recon) and the per-case Wilcoxon, not the ratio magnitude."
            ),
        }

    payload = {
        "_provenance": cm.provenance(inputs, seed=args.seed),
        "config": {
            "horizons": list(HORIZONS),
            "headline_horizon": HEADLINE_H,
            "k_knn": args.k,
            "k_local_pca": args.kpca,
            "n_components_local_pca": args.n_components,
            "split": "test_b",
            "query": "full-context rollout z_full at frame impact+H",
            "reference_cloud": "per-frame TRAIN encoded latents at modal frame impact+H",
            "normalisation": "RMS distance of TRAIN cloud to its centroid (metric-independent)",
            "pred_lat_map": PRED_LAT,
        },
        "per_family": per_family,
        "reconstructive_vs_predictive": separation,
        "verdict": verdict,
    }
    from pathlib import Path as _P

    out_json = _P(args.out_json)
    cm.write_artifact(out_json, payload)

    # ---- markdown report ----
    lines: list[str] = []
    lines.append("# Track H: metric-independent manifold-departure diagnostics (v2.1)\n")
    lines.append(
        "Generated by `scripts/session29/manifold_diagnostics.py`. test_b, "
        "full-context rollouts (`z_full`), d = 64. Query = rollout latent at frame "
        "impact+H; reference manifold = per-frame TRAIN encoded cloud at the modal "
        "frame impact+H (verbatim `drift_ce1.py` convention). Both diagnostics use "
        "NO covariance inverse, so there is no shrinkage knob to dismiss.\n"
    )
    lines.append("## Method\n")
    lines.append(
        f"- kNN distance: mean Euclidean distance to the k = {args.k} nearest TRAIN "
        "cloud points, normalised by the cloud scale (RMS distance of TRAIN points "
        "to their centroid).\n"
        f"- local-PCA residual: fit a rank-{args.n_components} PCA on the query's "
        f"k = {args.kpca} nearest TRAIN neighbours, report the norm of the query's "
        "component orthogonal to that local tangent subspace, same normalisation.\n"
        "- Both rotation-invariant and free of any covariance regularisation.\n"
        "- Significance: paired per-encounter (reconstructive - predictive) with the "
        "case-clustered block bootstrap + one-sided sign test from `stats_lib` "
        "(`case_permutation_p` deliberately NOT used; B6 lesson).\n"
    )

    for diag, lab in (
        (
            "knn_distance_rollout_median",
            "kNN distance to train manifold (median, cloud-scale units)",
        ),
        (
            "local_pca_residual_rollout_median",
            "local-PCA orthogonal residual (median, cloud-scale units)",
        ),
    ):
        lines.append(f"## {lab}\n")
        lines.append("| family | " + " | ".join(f"H={h}" for h in HORIZONS) + " |")
        lines.append("|---|" + "|".join(["---"] * len(HORIZONS)) + "|")
        for key in args.keys:
            pf = per_family[key]
            vals = " | ".join(f"{v:.3f}" for v in pf[diag])
            lines.append(f"| {pf['family_label']} | {vals} |")
        lines.append("")

    if separation:
        lines.append("## Reconstructive vs predictive separation (fukami - jepa), per horizon\n")
        for diag in ("knn_distance", "local_pca_residual"):
            lines.append(f"### {diag}\n")
            lines.append(
                "| H | median recon | median pred | ratio recon/pred | "
                "median delta | sign p (pred<recon) | Wilcoxon p |"
            )
            lines.append("|---|---|---|---|---|---|---|")
            for H in HORIZONS:
                s = separation[diag][str(H)]
                wp = s["case_paired"]["wilcoxon_p_one_sided"]
                wp_s = f"{wp:.3g}" if isinstance(wp, float) else str(wp)
                lines.append(
                    f"| {H} | {s['median_recon']:.3f} | {s['median_pred']:.3f} | "
                    f"{s['ratio_recon_over_pred']:.2f} | "
                    f"{s['median_delta_recon_minus_pred']:+.3f} | "
                    f"{s['sign_p_one_sided_pred_less_recon']:.3g} | {wp_s} |"
                )
            lines.append("")

    if verdict:
        lines.append("## Verdict\n")
        lines.append(
            f"**{verdict['branch']}.** At H = {HEADLINE_H}: kNN corroborates "
            f"Mahalanobis = {verdict['knn_corroborates_mahalanobis']} "
            f"(recon/pred ratio {verdict['knn_ratio_recon_over_pred']:.2f}, "
            f"sign p {verdict['knn_sign_p_pred_less_recon']:.3g}); "
            f"local-PCA corroborates = {verdict['local_pca_corroborates_mahalanobis']} "
            f"(recon/pred ratio {verdict['pca_ratio_recon_over_pred']:.2f}, "
            f"sign p {verdict['pca_sign_p_pred_less_recon']:.3g}).\n"
        )
        lines.append(verdict["note"] + "\n")

    out_md = _P(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))

    print(f"\nwrote {out_json}\nwrote {out_md}")
    if verdict:
        print(
            f"VERDICT: {verdict['branch']} | kNN corroborates="
            f"{verdict['knn_corroborates_mahalanobis']} "
            f"local-PCA corroborates={verdict['local_pca_corroborates_mahalanobis']}"
        )


if __name__ == "__main__":
    main()
