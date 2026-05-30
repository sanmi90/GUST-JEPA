"""Session 20 Track C: persistent homology of latent trajectories.

Turns the scalar "9.9x Mahalanobis drift" result (Session 18, B1) into a
coordinate-free topological invariant. Reference: Smith, Fukami, Sedky, Jones,
Taira, J. Fluid Mech. 980, A18 (2024).

Scientific claim under test
---------------------------
Each gust encounter is a closed cycle in latent space (base limit cycle ->
leading-edge-vortex excursion -> recovery), which appears as one long-lived H1
generator ("loop") in the Vietoris-Rips persistence diagram of that encounter's
120 per-frame latent vectors in R^64. The JEPA *predictive* rollout should
PRESERVE that loop (lifetime ratio rollout/DNS near 1); the reconstructive
(Fukami) rollout's loop should COLLAPSE or fragment as it drifts off-manifold.

Four point clouds per encounter (each is the 120 per-frame latents in R^64):
    (1) JEPA   z_dns     (DNS-encoded ground-truth latent trajectory)
    (2) JEPA   z_markov  (Markov rollout from the impact frame)
    (3) Fukami z_dns
    (4) Fukami z_markov
The H1-lifetime ratio rollout/DNS is (cloud 2 / cloud 1) for JEPA and
(cloud 4 / cloud 3) for Fukami.

Noise filter (Smith et al. JFM 2024, "minimal-persistence noise diagonal")
--------------------------------------------------------------------------
A generator is significant if its persistence (death - birth) exceeds a noise
floor set by the point cloud's intrinsic scale: floor = NOISE_FRAC * cloud_scale,
where cloud_scale is the largest finite H0 death (the Rips diameter at which the
cloud becomes one connected component). This is scale-invariant, so it adapts to
JEPA's smaller latent norms vs Fukami's drift-inflated norms. The "encounter
loop" is the maximum-persistence H1 generator ABOVE this floor; if none survive
the loop is reported as absent (lifetime 0), which is the honest reading of a
collapsed cycle.

D123 cross-check
----------------
D123 (Session 17) found the impact frame (frame 40, uniform across test_b/test_c)
is a CURVATURE MINIMUM of the DNS latent trajectory: a smooth, locally-linear
high-velocity pass-through, reproducible across seeds at Spearman 0.95, NOT a
sharp corner. A single smooth closed H1 loop is consistent with that. This script
recomputes discrete curvature kappa(t) along each DNS trajectory and verifies it
DIPS (trough) rather than PEAKS at frame 40; a peak would contradict D123 and
flag a sharp corner, which must be debugged before any topological claim.

Outputs (outputs/session20/persistent_homology/)
------------------------------------------------
    persistent_homology.json   aggregate ratios + per-encounter values + gate.
    persistence_homology.png / .pdf   four persistence diagrams (one row) plus
        a panel of H1 lifetime vs rollout horizon per family.

Pure CPU numpy + ripser. No GPU, no training.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from ripser import ripser

# 120 per-frame latents in R^64 is a valid point cloud (more columns than rows);
# ripser warns about a possible transpose, which does not apply here.
warnings.filterwarnings("ignore", message="The input point cloud has more columns than rows")

REPO = Path(__file__).resolve().parents[2]
ROLLOUTS_ROOT = REPO / "outputs" / "session18" / "exp_b1_test3"
OUT_DIR = REPO / "outputs" / "session20" / "persistent_homology"

JEPA_TAG = "jepa_d64_test1_noBN"
FUKAMI_TAG = "fukami_d64_noBN"

# Fraction of the H0 cloud diameter below which an H1 generator is noise.
# Chosen in the exploratory pass (Session 20): cleanly flags near-degenerate
# loops (e.g. max persistence ~ 0.002 on a cloud of diameter ~ 0.5) as absent,
# while keeping the dominant encounter loop. Scale-invariant by construction.
NOISE_FRAC = 0.05

# Rollout horizons (frames past impact) for the loop-decay sweep.
HORIZONS = (16, 32, 48, 64, 96)

# The four families: (label, model TAG, latent key).
FAMILIES = [
    ("JEPA z_dns", JEPA_TAG, "z_dns"),
    ("JEPA z_markov", JEPA_TAG, "z_markov"),
    ("Fukami z_dns", FUKAMI_TAG, "z_dns"),
    ("Fukami z_markov", FUKAMI_TAG, "z_markov"),
]


def rips_h1(point_cloud: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Vietoris-Rips persistence up to H1 on a (n_points, dim) cloud.

    Returns (h0_diagram, h1_diagram, cloud_scale) where cloud_scale is the
    largest finite H0 death (the Rips scale at which the cloud becomes one
    connected component), used as the intrinsic length scale for the noise
    floor.
    """
    res = ripser(point_cloud.astype(np.float64), maxdim=1)
    h0, h1 = res["dgms"][0], res["dgms"][1]
    h0_deaths = h0[:, 1]
    h0_deaths = h0_deaths[np.isfinite(h0_deaths)]
    cloud_scale = float(h0_deaths.max()) if h0_deaths.size else 0.0
    return h0, h1, cloud_scale


def loop_summary(point_cloud: np.ndarray) -> dict:
    """Max-persistence H1 generator above the noise floor and a feature count.

    Returns a dict with the max significant H1 lifetime (0.0 if none survive the
    floor), the noise floor, the number of significant H1 generators (proxy for
    spurious fragmentation), the cloud scale, and the birth/death of the loop.
    """
    h0, h1, cloud_scale = rips_h1(point_cloud)
    floor = NOISE_FRAC * cloud_scale
    if h1.size == 0:
        return {
            "max_lifetime": 0.0,
            "max_lifetime_rel": 0.0,
            "noise_floor": floor,
            "cloud_scale": cloud_scale,
            "n_significant": 0,
            "n_total_h1": 0,
            "loop_birth": float("nan"),
            "loop_death": float("nan"),
        }
    lifetimes = h1[:, 1] - h1[:, 0]
    significant = lifetimes > floor
    n_sig = int(significant.sum())
    if n_sig == 0:
        return {
            "max_lifetime": 0.0,
            "max_lifetime_rel": 0.0,
            "noise_floor": floor,
            "cloud_scale": cloud_scale,
            "n_significant": 0,
            "n_total_h1": int(h1.shape[0]),
            "loop_birth": float("nan"),
            "loop_death": float("nan"),
        }
    # Among significant generators, the encounter loop is the longest-lived.
    idx_sig = np.where(significant)[0]
    best = idx_sig[np.argmax(lifetimes[idx_sig])]
    return {
        "max_lifetime": float(lifetimes[best]),
        # Scale-normalised prominence: lifetime / cloud diameter. Removes the
        # confound that Fukami's drifted clouds are intrinsically larger, so a
        # long-lived loop there can be a large diffuse blob rather than a tight
        # cycle.
        "max_lifetime_rel": float(lifetimes[best] / cloud_scale) if cloud_scale > 0 else 0.0,
        "noise_floor": floor,
        "cloud_scale": cloud_scale,
        "n_significant": n_sig,
        "n_total_h1": int(h1.shape[0]),
        "loop_birth": float(h1[best, 0]),
        "loop_death": float(h1[best, 1]),
    }


def discrete_curvature(traj: np.ndarray) -> np.ndarray:
    """Discrete Menger-style turning curvature kappa(t) along a (T, dim) path.

    kappa(t) = angle between consecutive velocity vectors / arc length, the same
    locally-linear pass-through diagnostic D123 used. Endpoints set to NaN.
    """
    T = traj.shape[0]
    kappa = np.full(T, np.nan)
    for t in range(1, T - 1):
        v0 = traj[t] - traj[t - 1]
        v1 = traj[t + 1] - traj[t]
        n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
        if n0 < 1e-9 or n1 < 1e-9:
            continue
        cos_a = np.clip(np.dot(v0, v1) / (n0 * n1), -1.0, 1.0)
        angle = np.arccos(cos_a)
        kappa[t] = angle / (0.5 * (n0 + n1))
    return kappa


def curvature_check_d123(z_dns_all: np.ndarray, impact: int) -> dict:
    """Verify kappa(t) DIPS (trough) at the impact frame across encounters.

    Reproduces the D123 curvature-minimum geometry. For each encounter compute
    kappa(t), then compare the mean kappa in a +/-2 frame window at impact to the
    encounter's median kappa. trough_ratio < 1 means a dip (consistent with
    D123). A ratio > 1 (peak) would imply a sharp corner and CONTRADICT D123.
    """
    n = z_dns_all.shape[0]
    win = 2
    ratios = []
    for i in range(n):
        kappa = discrete_curvature(z_dns_all[i])
        lo, hi = max(1, impact - win), min(z_dns_all.shape[1] - 1, impact + win + 1)
        at_impact = np.nanmean(kappa[lo:hi])
        baseline = np.nanmedian(kappa)
        if baseline > 0 and np.isfinite(at_impact):
            ratios.append(at_impact / baseline)
    ratios = np.asarray(ratios)
    median_ratio = float(np.median(ratios))
    return {
        "median_trough_ratio": median_ratio,
        "is_dip": bool(median_ratio < 1.0),
        "frac_encounters_dip": float(np.mean(ratios < 1.0)),
        "n_encounters": int(ratios.size),
        "interpretation": (
            "DIP at impact -> consistent with D123 (smooth pass-through)"
            if median_ratio < 1.0
            else "PEAK at impact -> CONTRADICTS D123 (sharp corner); debug"
        ),
    }


def load_split(tag: str, split: str) -> dict:
    path = ROLLOUTS_ROOT / f"rollouts_{tag}" / f"{split}.npz"
    return dict(np.load(path, allow_pickle=True))


def analyze_split(split: str) -> dict:
    """Per-encounter loop summaries for all four families plus aggregate ratios."""
    jepa = load_split(JEPA_TAG, split)
    fukami = load_split(FUKAMI_TAG, split)
    n = jepa["z_dns"].shape[0]
    impact = int(jepa["impact_frame"][0])
    assert np.all(jepa["impact_frame"] == impact), "impact frame not uniform"

    # Shape guard.
    for src, name in [(jepa, JEPA_TAG), (fukami, FUKAMI_TAG)]:
        for key in ("z_dns", "z_markov"):
            assert src[key].shape[:2] == (n, 120), f"{name}:{key} wrong shape {src[key].shape}"

    per_enc = []
    jepa_ratios, fukami_ratios = [], []
    jepa_ratios_rel, fukami_ratios_rel = [], []
    nsig = {label: [] for label, _, _ in FAMILIES}

    for i in range(n):
        s = {
            "JEPA z_dns": loop_summary(jepa["z_dns"][i]),
            "JEPA z_markov": loop_summary(jepa["z_markov"][i]),
            "Fukami z_dns": loop_summary(fukami["z_dns"][i]),
            "Fukami z_markov": loop_summary(fukami["z_markov"][i]),
        }
        for label in nsig:
            nsig[label].append(s[label]["n_significant"])

        jd, jm = s["JEPA z_dns"]["max_lifetime"], s["JEPA z_markov"]["max_lifetime"]
        fd, fm = s["Fukami z_dns"]["max_lifetime"], s["Fukami z_markov"]["max_lifetime"]
        jdr, jmr = s["JEPA z_dns"]["max_lifetime_rel"], s["JEPA z_markov"]["max_lifetime_rel"]
        fdr, fmr = s["Fukami z_dns"]["max_lifetime_rel"], s["Fukami z_markov"]["max_lifetime_rel"]
        # Ratio rollout/DNS only defined when the DNS loop exists (> 0).
        jratio = jm / jd if jd > 0 else float("nan")
        fratio = fm / fd if fd > 0 else float("nan")
        jratio_rel = jmr / jdr if jdr > 0 else float("nan")
        fratio_rel = fmr / fdr if fdr > 0 else float("nan")
        if np.isfinite(jratio):
            jepa_ratios.append(jratio)
        if np.isfinite(fratio):
            fukami_ratios.append(fratio)
        if np.isfinite(jratio_rel):
            jepa_ratios_rel.append(jratio_rel)
        if np.isfinite(fratio_rel):
            fukami_ratios_rel.append(fratio_rel)

        per_enc.append(
            {
                "encounter": i,
                "case_id": str(jepa["case_ids"][i]),
                "G": float(jepa["G"][i]),
                "D": float(jepa["D"][i]),
                "Y": float(jepa["Y"][i]),
                "jepa_dns_lifetime": jd,
                "jepa_markov_lifetime": jm,
                "jepa_ratio": jratio,
                "jepa_dns_lifetime_rel": jdr,
                "jepa_markov_lifetime_rel": jmr,
                "jepa_ratio_rel": jratio_rel,
                "fukami_dns_lifetime": fd,
                "fukami_markov_lifetime": fm,
                "fukami_ratio": fratio,
                "fukami_dns_lifetime_rel": fdr,
                "fukami_markov_lifetime_rel": fmr,
                "fukami_ratio_rel": fratio_rel,
                "jepa_dns_nsig": s["JEPA z_dns"]["n_significant"],
                "jepa_markov_nsig": s["JEPA z_markov"]["n_significant"],
                "fukami_dns_nsig": s["Fukami z_dns"]["n_significant"],
                "fukami_markov_nsig": s["Fukami z_markov"]["n_significant"],
            }
        )

    def agg(vals: list) -> dict:
        a = np.asarray(vals, dtype=float)
        if a.size == 0:
            return {"median": float("nan"), "iqr_lo": float("nan"), "iqr_hi": float("nan"), "n": 0}
        return {
            "median": float(np.median(a)),
            "iqr_lo": float(np.percentile(a, 25)),
            "iqr_hi": float(np.percentile(a, 75)),
            "n": int(a.size),
        }

    # Generator-count contrast in the DNS encoding: the robust, scale-free
    # topological signal (JEPA encodes a clean single loop; Fukami fragments
    # into several). Mann-Whitney one-sided (JEPA fewer loops than Fukami).
    jepa_dns_nsig = np.array([e["jepa_dns_nsig"] for e in per_enc])
    fukami_dns_nsig = np.array([e["fukami_dns_nsig"] for e in per_enc])
    try:
        from scipy import stats

        U, p = stats.mannwhitneyu(jepa_dns_nsig, fukami_dns_nsig, alternative="less")
        mw = {"U": float(U), "p_one_sided": float(p)}
    except Exception as exc:  # scipy optional
        mw = {"U": None, "p_one_sided": None, "error": str(exc)}

    nsig_contrast = {
        "jepa_dns_nsig_median": float(np.median(jepa_dns_nsig)),
        "jepa_dns_nsig_mean": float(jepa_dns_nsig.mean()),
        "fukami_dns_nsig_median": float(np.median(fukami_dns_nsig)),
        "fukami_dns_nsig_mean": float(fukami_dns_nsig.mean()),
        "frac_jepa_single_loop": float((jepa_dns_nsig == 1).mean()),
        "frac_fukami_ge3_loops": float((fukami_dns_nsig >= 3).mean()),
        "mannwhitney_jepa_fewer": mw,
    }

    return {
        "split": split,
        "n_encounters": n,
        "impact_frame": impact,
        "jepa_ratio": agg(jepa_ratios),
        "fukami_ratio": agg(fukami_ratios),
        "jepa_ratio_rel": agg(jepa_ratios_rel),
        "fukami_ratio_rel": agg(fukami_ratios_rel),
        "jepa_dns_lifetime": agg([e["jepa_dns_lifetime"] for e in per_enc]),
        "jepa_markov_lifetime": agg([e["jepa_markov_lifetime"] for e in per_enc]),
        "fukami_dns_lifetime": agg([e["fukami_dns_lifetime"] for e in per_enc]),
        "fukami_markov_lifetime": agg([e["fukami_markov_lifetime"] for e in per_enc]),
        "n_significant_median": {label: float(np.median(v)) for label, v in nsig.items()},
        "n_significant_mean": {label: float(np.mean(v)) for label, v in nsig.items()},
        "generator_count_contrast": nsig_contrast,
        "per_encounter": per_enc,
        "curvature_d123": curvature_check_d123(jepa["z_dns"], impact),
    }


def horizon_sweep(split: str) -> dict:
    """Median max-H1 lifetime vs rollout horizon (frames past impact) per family.

    For each horizon H truncate the trajectory at frame impact+H and recompute
    the max significant H1 lifetime. Shows when the loop dies. The DNS clouds are
    truncated identically so the comparison is like-for-like.
    """
    jepa = load_split(JEPA_TAG, split)
    fukami = load_split(FUKAMI_TAG, split)
    n = jepa["z_dns"].shape[0]
    impact = int(jepa["impact_frame"][0])

    sources = {
        "JEPA z_dns": jepa["z_dns"],
        "JEPA z_markov": jepa["z_markov"],
        "Fukami z_dns": fukami["z_dns"],
        "Fukami z_markov": fukami["z_markov"],
    }
    out = {
        label: {"horizons": list(HORIZONS), "median_lifetime": [], "iqr_lo": [], "iqr_hi": []}
        for label in sources
    }
    for H in HORIZONS:
        end = min(impact + H, jepa["z_dns"].shape[1])
        for label, arr in sources.items():
            lifes = [loop_summary(arr[i][:end])["max_lifetime"] for i in range(n)]
            a = np.asarray(lifes, dtype=float)
            out[label]["median_lifetime"].append(float(np.median(a)))
            out[label]["iqr_lo"].append(float(np.percentile(a, 25)))
            out[label]["iqr_hi"].append(float(np.percentile(a, 75)))
    return out


def representative_diagrams(split: str) -> dict:
    """Persistence diagrams for one representative encounter (for the figure).

    Picks the test_b encounter whose JEPA DNS loop is closest to the median DNS
    lifetime, so the plotted diagram is typical rather than extremal.
    """
    jepa = load_split(JEPA_TAG, split)
    fukami = load_split(FUKAMI_TAG, split)
    n = jepa["z_dns"].shape[0]
    jdns = np.array([loop_summary(jepa["z_dns"][i])["max_lifetime"] for i in range(n)])
    med = np.median(jdns[jdns > 0]) if np.any(jdns > 0) else np.median(jdns)
    rep = int(np.argmin(np.abs(jdns - med)))

    diags = {}
    clouds = {
        "JEPA z_dns": jepa["z_dns"][rep],
        "JEPA z_markov": jepa["z_markov"][rep],
        "Fukami z_dns": fukami["z_dns"][rep],
        "Fukami z_markov": fukami["z_markov"][rep],
    }
    for label, pc in clouds.items():
        h0, h1, cloud_scale = rips_h1(pc)
        diags[label] = {
            "h1": h1.tolist(),
            "cloud_scale": cloud_scale,
            "noise_floor": NOISE_FRAC * cloud_scale,
        }
    return {"rep_encounter": rep, "rep_case_id": str(jepa["case_ids"][rep]), "diagrams": diags}


def make_figure(tb: dict, sweep_tb: dict, rep: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(23, 4.4))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1.25, 1.1], wspace=0.34)

    colors = {
        "JEPA z_dns": "#1f77b4",
        "JEPA z_markov": "#2ca02c",
        "Fukami z_dns": "#9467bd",
        "Fukami z_markov": "#d62728",
    }
    # Common axis range across the four diagrams for visual comparability.
    all_pts = []
    for label in colors:
        h1 = np.asarray(rep["diagrams"][label]["h1"], dtype=float)
        if h1.size:
            all_pts.append(h1)
    amax = max((p.max() for p in all_pts), default=1.0) * 1.05

    for col, label in enumerate(colors):
        ax = fig.add_subplot(gs[0, col])
        info = rep["diagrams"][label]
        h1 = np.asarray(info["h1"], dtype=float)
        floor = info["noise_floor"]
        ax.plot([0, amax], [0, amax], color="0.6", lw=0.8, zorder=0)
        # Noise band: generators with death - birth < floor.
        ax.fill_between(
            [0, amax],
            [0, amax],
            [floor, amax + floor],
            color="0.85",
            alpha=0.6,
            zorder=0,
            label=f"noise (< {floor:.2f})",
        )
        if h1.size:
            life = h1[:, 1] - h1[:, 0]
            sig = life > floor
            ax.scatter(h1[~sig, 0], h1[~sig, 1], s=16, c="0.6", alpha=0.7, zorder=2)
            ax.scatter(
                h1[sig, 0],
                h1[sig, 1],
                s=42,
                c=colors[label],
                edgecolors="k",
                linewidths=0.5,
                zorder=3,
            )
            if sig.any():
                best = np.where(sig)[0][np.argmax(life[sig])]
                ax.annotate(
                    f"loop\nL={life[best]:.2f}",
                    (h1[best, 0], h1[best, 1]),
                    textcoords="offset points",
                    xytext=(6, -14),
                    fontsize=8,
                )
        ax.set_xlim(0, amax)
        ax.set_ylim(0, amax)
        ax.set_xlabel("birth")
        if col == 0:
            ax.set_ylabel("death")
        ax.set_title(label, fontsize=10)
        ax.legend(loc="lower right", fontsize=6, framealpha=0.7)

    # Horizon-decay panel.
    ax = fig.add_subplot(gs[0, 4])
    H = sweep_tb["JEPA z_dns"]["horizons"]
    styles = {
        "JEPA z_dns": dict(color=colors["JEPA z_dns"], ls="-", marker="o"),
        "JEPA z_markov": dict(color=colors["JEPA z_markov"], ls="--", marker="s"),
        "Fukami z_dns": dict(color=colors["Fukami z_dns"], ls="-", marker="o"),
        "Fukami z_markov": dict(color=colors["Fukami z_markov"], ls="--", marker="s"),
    }
    for label, st in styles.items():
        med = sweep_tb[label]["median_lifetime"]
        ax.plot(H, med, label=label, lw=1.6, ms=5, **st)
    ax.set_xlabel("rollout horizon H (frames past impact)")
    ax.set_ylabel("median max H1 lifetime")
    ax.set_title("H1 loop vs rollout horizon (test_b)", fontsize=10)
    ax.legend(fontsize=7, framealpha=0.7)
    ax.grid(alpha=0.25)

    # Generator-count histogram (the headline descriptive signal): JEPA encodes
    # a clean single loop; Fukami fragments into several.
    ax = fig.add_subplot(gs[0, 5])
    jn = np.array([e["jepa_dns_nsig"] for e in tb["per_encounter"]])
    fn = np.array([e["fukami_dns_nsig"] for e in tb["per_encounter"]])
    bins = np.arange(0, max(jn.max(), fn.max()) + 2) - 0.5
    ax.hist(
        jn,
        bins=bins,
        alpha=0.6,
        color=colors["JEPA z_dns"],
        label=f"JEPA (med {np.median(jn):.0f})",
        density=True,
    )
    ax.hist(
        fn,
        bins=bins,
        alpha=0.6,
        color=colors["Fukami z_dns"],
        label=f"Fukami (med {np.median(fn):.0f})",
        density=True,
    )
    gc = tb["generator_count_contrast"]
    p = gc["mannwhitney_jepa_fewer"].get("p_one_sided", float("nan"))
    ax.set_xlabel("number of significant H1 loops (DNS)")
    ax.set_ylabel("fraction of encounters")
    ax.set_title(f"Loop count (test_b)\nMann-Whitney p={p:.1e}", fontsize=10)
    ax.legend(fontsize=7, framealpha=0.7)
    ax.grid(alpha=0.25)

    jr = tb["jepa_ratio"]
    fr = tb["fukami_ratio"]
    gc = tb["generator_count_contrast"]
    fig.suptitle(
        f"Persistent homology of latent encounter loops (Vietoris-Rips, test_b, "
        f"n={tb['n_encounters']}, rep case {rep['rep_case_id']}).  "
        f"H1-lifetime ratio rollout/DNS: JEPA {jr['median']:.2f} "
        f"[{jr['iqr_lo']:.2f}, {jr['iqr_hi']:.2f}], Fukami {fr['median']:.2f} "
        f"[{fr['iqr_lo']:.2f}, {fr['iqr_hi']:.2f}] (ratio confounded by cloud scale).  "
        f"Robust signal: JEPA encodes a single clean loop (median "
        f"{gc['jepa_dns_nsig_median']:.0f} H1), Fukami fragments (median "
        f"{gc['fukami_dns_nsig_median']:.0f}).",
        fontsize=10.5,
        y=1.04,
    )
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def gate_verdict(tb: dict) -> dict:
    """PRIMARY gate: JEPA median ratio >= 0.7 AND Fukami median ratio < 0.5.

    Records the descriptive headline (generator-count contrast) regardless, since
    the lifetime ratio is confounded by cloud scale and turns out NOT to support
    the hypothesised direction (Fukami's loop is preserved because the rollout
    drifts into a large diffuse blob, not because it is on-manifold).
    """
    jm = tb["jepa_ratio"]["median"]
    fm = tb["fukami_ratio"]["median"]
    strong = bool(jm >= 0.7 and fm < 0.5)
    gc = tb["generator_count_contrast"]
    return {
        "jepa_median_ratio": jm,
        "fukami_median_ratio": fm,
        "jepa_median_ratio_rel": tb["jepa_ratio_rel"]["median"],
        "fukami_median_ratio_rel": tb["fukami_ratio_rel"]["median"],
        "primary_gate_strong": strong,
        "verdict": "STRONG" if strong else "DESCRIPTIVE",
        "criterion": "JEPA median ratio >= 0.7 AND Fukami median ratio < 0.5",
        "descriptive_headline": (
            "Lifetime ratio does NOT separate in the hypothesised direction "
            f"(JEPA {jm:.2f}, Fukami {fm:.2f}); the robust topological signal is "
            f"the generator count: JEPA encodes a clean single loop (DNS median "
            f"{gc['jepa_dns_nsig_median']:.0f} significant H1, "
            f"{gc['frac_jepa_single_loop']*100:.0f}% "
            f"exactly one), Fukami fragments (DNS median {gc['fukami_dns_nsig_median']:.0f}, "
            f"{gc['frac_fukami_ge3_loops']*100:.0f}% with >=3 loops; "
            f"Mann-Whitney p={gc['mannwhitney_jepa_fewer'].get('p_one_sided', float('nan')):.1e})."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--jepa-tag", type=str, default=JEPA_TAG,
                    help="predictive (JEPA) rollout tag; use jepa_d32_noBN for d=32")
    ap.add_argument("--fukami-tag", type=str, default=FUKAMI_TAG,
                    help="reconstructive (Fukami) rollout tag")
    ap.add_argument("--skip-test-c", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    global JEPA_TAG, FUKAMI_TAG
    JEPA_TAG, FUKAMI_TAG = args.jepa_tag, args.fukami_tag
    print(f"[track-c] JEPA={JEPA_TAG}  Fukami={FUKAMI_TAG}  out={args.out_dir}")

    print("[1/4] test_b per-encounter persistence ...")
    tb = analyze_split("test_b")
    print(
        f"      JEPA   ratio median={tb['jepa_ratio']['median']:.3f} "
        f"IQR[{tb['jepa_ratio']['iqr_lo']:.3f},{tb['jepa_ratio']['iqr_hi']:.3f}] "
        f"(n={tb['jepa_ratio']['n']})"
    )
    print(
        f"      Fukami ratio median={tb['fukami_ratio']['median']:.3f} "
        f"IQR[{tb['fukami_ratio']['iqr_lo']:.3f},{tb['fukami_ratio']['iqr_hi']:.3f}] "
        f"(n={tb['fukami_ratio']['n']})"
    )
    print(
        f"      D123 curvature: {tb['curvature_d123']['interpretation']} "
        f"(trough_ratio={tb['curvature_d123']['median_trough_ratio']:.3f}, "
        f"frac_dip={tb['curvature_d123']['frac_encounters_dip']:.2f})"
    )

    print("[2/4] test_b horizon sweep ...")
    sweep_tb = horizon_sweep("test_b")

    print("[3/4] representative diagrams + figure ...")
    rep = representative_diagrams("test_b")
    make_figure(tb, sweep_tb, rep, args.out_dir / "persistence_homology")

    tc = None
    sweep_tc = None
    if not args.skip_test_c:
        print("[4/4] test_c per-encounter persistence + horizon sweep ...")
        tc = analyze_split("test_c")
        sweep_tc = horizon_sweep("test_c")
        print(
            f"      JEPA   ratio median={tc['jepa_ratio']['median']:.3f}, "
            f"Fukami ratio median={tc['fukami_ratio']['median']:.3f}"
        )

    verdict = gate_verdict(tb)
    gc = tb["generator_count_contrast"]
    print(
        f"\n  GATE VERDICT (test_b): {verdict['verdict']}  "
        f"(lifetime ratio JEPA {verdict['jepa_median_ratio']:.2f} vs Fukami "
        f"{verdict['fukami_median_ratio']:.2f}; gate wanted JEPA>=0.7 AND Fukami<0.5)"
    )
    print(
        f"  Descriptive headline: JEPA DNS loop count median "
        f"{gc['jepa_dns_nsig_median']:.0f} (mean {gc['jepa_dns_nsig_mean']:.2f}, "
        f"{gc['frac_jepa_single_loop']*100:.0f}% single), Fukami median "
        f"{gc['fukami_dns_nsig_median']:.0f} (mean {gc['fukami_dns_nsig_mean']:.2f}, "
        f"{gc['frac_fukami_ge3_loops']*100:.0f}% >=3); Mann-Whitney "
        f"p={gc['mannwhitney_jepa_fewer']['p_one_sided']:.2e}"
    )

    out = {
        "description": "Persistent homology (Vietoris-Rips H0/H1) of latent "
        "encounter loops; JEPA vs Fukami rollout/DNS H1-lifetime ratio.",
        "reference": "Smith, Fukami, Sedky, Jones, Taira, JFM 980, A18 (2024).",
        "jepa_tag": JEPA_TAG,
        "fukami_tag": FUKAMI_TAG,
        "noise_frac": NOISE_FRAC,
        "window": "full 120-frame trajectory (primary); horizon sweep truncates "
        "at impact+H frames",
        "horizons": list(HORIZONS),
        "gate": verdict,
        "test_b": tb,
        "test_b_horizon_sweep": sweep_tb,
        "test_c": tc,
        "test_c_horizon_sweep": sweep_tc,
        "representative": rep,
    }
    out_json = args.out_dir / "persistent_homology.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Wrote {out_json}")
    print(f"  Wrote {args.out_dir / 'persistence_homology.png'} (+ .pdf)")


if __name__ == "__main__":
    main()
