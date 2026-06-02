#!/usr/bin/env python3
"""Session 23: continuous error-map trend plots of the predictive (JEPA) advantage.

WHY THIS EXISTS
---------------
The manuscript deferred a per-stratum table of where the predictive (JEPA) wake
closure beats the reconstructive autoencoder (Fukami AE). A coarse stratification
of n=42 held-out encounters into G/D/Y bins is underpowered and arbitrary in its
bin edges. This script replaces it with CONTINUOUS error-map trends: the paired
per-encounter improvement Delta_e = e_AE - e_JEPA plotted against each gust axis
(G, D, Y) and against the baseline shedding phase phi at impact, each with a
LOWESS trend and a bootstrap band over encounters.

It is post-processing only; no retraining, CPU-only.

DATA PROVENANCE (real per-encounter values only; no approximation)
------------------------------------------------------------------
1. Per-encounter wake-enstrophy absolute errors e_JEPA, e_AE are read from the
   VERIFIED Session 21 paired-closure machinery
   (scripts/session21/session21_paired_closure_stats.load_per_encounter_abs_error),
   which is itself the Session 20 ridge-probe closure pipeline
   (scripts/session20/exp_closure_r2). For each test_b encounter it returns
   |probe(z) - DNS| of the wake enstrophy at frame impact+H (H=16), in a canonical
   sorted (case_id, encounter) order so the two families are index-aligned and the
   improvement is genuinely PAIRED per encounter.
     mode = "repr"     : probe on the simulation-encoded latent  z_dns   (Table 2)
     mode = "forecast" : probe on the Markov rollout from impact  z_markov (Table 3)
   Primary panels use the representational mode; the forecast mode is also produced.

2. (G, D, Y, case_id, encounter_index, impact_frame) come from the DNS-encoded
   latents outputs/session14/latents/S12_E_d64/test_b.npz. They are reordered into
   the SAME canonical sorted (case_id, encounter) order as the error arrays.

3. phi = baseline shedding phase at impact. Computed from the Session 20 Track E
   limit-cycle machinery (HANDOFF D136): the four no-gust baseline episodes are
   concatenated and PCA'd to define the latent limit cycle, and the Hilbert
   analytic-signal phase of the leading PC (PC1) is the protophase (the protophase
   D136 validates as monotone over a shedding period). Per test_b encounter, PC1 of
   the encounter's own DNS-encoded trajectory over the UNDISTURBED PRE-IMPACT window
   [0, impact] is Hilbert-transformed and the phase is read at the impact frame,
   wrapped to [0, 2*pi). This reads the shedding phase the gust hits from the clean
   pre-impact flow, BEFORE the gust displaces the latent.

   HONEST CAVEAT (verified, not papered over): the impact frame is fixed at 40 for
   every encounter and gusts are released every 120 frames while the shedding period
   is ~56 frames, so 120 ~= 2.14 periods and consecutive encounters alternate between
   two near-opposite shedding states. The resulting phi distribution is BIMODAL /
   near-discrete (clusters near ~1.2 and ~4.4 rad), not well spread on [0, 2*pi). The
   phi panel and its LOWESS trend are therefore only weakly informative; this is
   flagged in the JSON (modes[*].summary.phi_well_spread = false). The (PC1, PC2)
   plane-angle protophase of the gust-displaced latent at impact was tried first and
   rejected: it collapses to a single ~0.3 rad cluster because the gust has already
   moved the latent off the baseline orbit by impact, so it is not a baseline phase.
   No phase is fabricated; phi is computed entirely from the real limit-cycle PCA.

OUTPUTS
-------
- outputs/session23/error_maps/error_maps.json : per-encounter table + provenance
- outputs/session23/error_maps/error_maps.png  : 4 panels (Delta_e vs G, D, Y, phi)
  for the representational mode, plus a forecast-mode companion figure
  error_maps_forecast.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import hilbert  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from statsmodels.nonparametric.smoothers_lowess import lowess  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session21"))
sys.path.insert(0, str(REPO / "scripts" / "session20"))

# Canonical per-encounter paired closure errors (the verified D139 machinery).
from session21_paired_closure_stats import load_per_encounter_abs_error  # noqa: E402

# Baseline limit-cycle machinery (D136 phase / amplitude).
from exp_phase_amplitude import estimate_period, load_baseline_episodes  # noqa: E402

# Shared paper figure style.
from figstyle import FAMILY_COLOR, figure_size, use_style  # noqa: E402

OUT_DIR = REPO / "outputs" / "session23" / "error_maps"
DNS_LATENTS = REPO / "outputs" / "session14" / "latents" / "S12_E_d64"
HEADLINE_OBS = "wake_enstrophy"
LOWESS_FRAC = 0.6
N_BOOT = 2000
BAND_LO, BAND_HI = 5.0, 95.0  # ~90% bootstrap band
RNG_SEED = 0


# ---------------------------------------------------------------------------
# 1. Per-encounter (G, D, Y, case_id) in the loader's canonical sorted order.
# ---------------------------------------------------------------------------
def canonical_metadata() -> dict:
    """test_b (G, D, Y, case_id, encounter) reordered to match the error arrays.

    The error loader keys each encounter by (str(case_id), int(encounter)) and
    returns the values in ``sorted(keys)`` order. We reproduce that exact order
    here from the DNS latents so the metadata is index-aligned with the errors.
    """
    tb = np.load(DNS_LATENTS / "test_b.npz", allow_pickle=True)
    cid = tb["case_id"].astype(str)
    ei = tb["encounter_index"].astype(int)
    keys = [(str(c), int(e)) for c, e in zip(cid, ei)]
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    return {
        "order": np.asarray(order, dtype=int),
        "case_id": cid[order],
        "encounter_index": ei[order],
        "G": tb["G"][order].astype(float),
        "D": tb["D"][order].astype(float),
        "Y": tb["Y"][order].astype(float),
        "impact_frame": tb["impact_frame"][order].astype(int),
        "z_full": tb["z_full"][order].astype(np.float64),
        "sorted_keys": [keys[i] for i in order],
    }


# ---------------------------------------------------------------------------
# 2. Baseline shedding phase at impact (D136 protophase).
# ---------------------------------------------------------------------------
def baseline_phase_machinery() -> dict:
    """Fit the baseline limit-cycle PCA and the protophase reference.

    Returns the fitted PCA, the orbit centroid in the (PC1, PC2) plane, and the
    period estimate for reporting.
    """
    zb, baseline_impact = load_baseline_episodes()  # (n_ep, 120, 64)
    continuous = zb.reshape(-1, zb.shape[-1])
    pca = PCA(n_components=6).fit(continuous)
    scores = pca.transform(continuous)
    period = estimate_period(scores[:, 0])
    centroid12 = np.array([scores[:, 0].mean(), scores[:, 1].mean()])
    # Hilbert protophase of PC1, kept only for validation reporting.
    pc1 = scores[:, 0]
    hphi = np.unwrap(np.angle(hilbert(pc1 - pc1.mean())))
    pframes = period["period_frames"]
    hilbert_advance = float(hphi[int(pframes)] - hphi[0]) if np.isfinite(pframes) else float("nan")
    return {
        "pca": pca,
        "centroid12": centroid12,
        "baseline_impact": int(baseline_impact),
        "period_frames": float(pframes),
        "period_tc": float(period["period_tc"]),
        "pc12_var_fraction": float(pca.explained_variance_ratio_[:2].sum()),
        "hilbert_advance_over_period_rad": hilbert_advance,
        "n_baseline_episodes": int(zb.shape[0]),
    }


def impact_phase(z_full: np.ndarray, impact: np.ndarray, mach: dict) -> tuple[np.ndarray, dict]:
    """Baseline shedding protophase in [0, 2*pi) at each encounter's impact.

    Project the encounter's own DNS-encoded trajectory onto the baseline limit
    cycle's PC1, Hilbert-transform the UNDISTURBED pre-impact window [0, impact],
    and read the analytic-signal phase at the impact frame. This is the D136
    protophase evaluated on the clean pre-impact shedding (before the gust displaces
    the latent). Returns phi and a spread diagnostic.
    """
    pca = mach["pca"]
    n = z_full.shape[0]
    phi = np.empty(n, dtype=float)
    for i in range(n):
        f0 = int(impact[i])
        s1 = pca.transform(z_full[i])[:, 0]  # PC1 of this encounter's trajectory
        seg = s1[: f0 + 1]
        seg = seg - seg.mean()
        phi[i] = float(np.angle(hilbert(seg))[-1]) % (2.0 * np.pi)
    # Spread diagnostic: a near-uniform phi fills all bins evenly; a bimodal /
    # near-discrete phi concentrates in a few bins and leaves interior gaps. Flag
    # "well spread" only if ALL 8 bins are occupied AND no bin holds more than 1/3
    # of the mass (an interior empty bin or a dominant bin signals clustering).
    counts, _ = np.histogram(phi, bins=8, range=(0.0, 2.0 * np.pi))
    occupied = int((counts > 0).sum())
    max_frac = float(counts.max()) / float(counts.sum())
    well_spread = bool(occupied == 8 and max_frac <= 1.0 / 3.0)
    diag = {
        "phi_8bin_counts": counts.tolist(),
        "phi_bins_occupied": occupied,
        "phi_max_bin_fraction": max_frac,
        "phi_well_spread": well_spread,
        "phi_definition": "Hilbert phase of baseline-PC1 on pre-impact window [0,impact]",
    }
    return phi, diag


# ---------------------------------------------------------------------------
# 3. LOWESS trend + bootstrap band over encounters.
# ---------------------------------------------------------------------------
def lowess_with_band(
    x: np.ndarray,
    y: np.ndarray,
    frac: float = LOWESS_FRAC,
    n_boot: int = N_BOOT,
    seed: int = RNG_SEED,
    n_grid: int = 200,
) -> dict:
    """LOWESS curve on a fixed grid plus a bootstrap band (resample encounters).

    The band resamples encounters with replacement, refits LOWESS, and evaluates
    each refit on a common x-grid via interpolation; the band is the [BAND_LO,
    BAND_HI] percentile envelope of the refits at each grid point.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    n = x.size
    grid = np.linspace(x.min(), x.max(), n_grid)

    def fit_on_grid(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        sm = lowess(yy, xx, frac=frac, return_sorted=True)
        # interpolate the (possibly de-duplicated) lowess output onto the grid
        return np.interp(grid, sm[:, 0], sm[:, 1])

    point = fit_on_grid(x, y)

    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, n_grid), dtype=float)
    import warnings

    with warnings.catch_warnings():
        # Degenerate resamples (few distinct x) make LOWESS emit a benign
        # divide-by-zero; we handle those rows via the ptp guard / nan-percentile.
        warnings.simplefilter("ignore", RuntimeWarning)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            xb, yb = x[idx], y[idx]
            if np.ptp(xb) < 1e-9:
                boots[b] = point
            else:
                boots[b] = fit_on_grid(xb, yb)
    band_lo = np.nanpercentile(boots, BAND_LO, axis=0)
    band_hi = np.nanpercentile(boots, BAND_HI, axis=0)

    # Reported trend direction is the SPEARMAN-rho sign: a rank-based, scale-free,
    # leverage-robust read of the monotone direction the reader infers from the
    # scatter. The OLS slope and the LOWESS end-minus-start are kept as auxiliary
    # diagnostics (they can disagree on a weak/non-monotone axis, e.g. when a few
    # high-leverage |G| outliers tilt the OLS line against the rank trend).
    from scipy.stats import spearmanr

    slope = float(np.polyfit(x, y, 1)[0])
    rho, pval = spearmanr(x, y)
    return {
        "grid": grid,
        "curve": point,
        "band_lo": band_lo,
        "band_hi": band_hi,
        "ols_slope": slope,
        "trend_sign": "increasing" if rho > 0 else "decreasing",
        "lowess_end_minus_start": float(point[-1] - point[0]),
        "spearman_rho": float(rho),
        "spearman_p": float(pval),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# 4. Figure.
# ---------------------------------------------------------------------------
def make_figure(table: dict, mode: str, trends: dict, out_path: Path, mach: dict) -> None:
    use_style()
    axes_keys = [("G", r"gust strength $G$"), ("D", r"gust scale $D$"),
                 ("Y", r"offset $Y/c$"), ("phi", r"baseline shedding phase $\varphi$ at impact")]
    phi_well_spread = table["phi_diagnostic"]["phi_well_spread"]
    fig, axes = plt.subplots(2, 2, figsize=figure_size(1.0, aspect=0.85))
    axes = axes.ravel()
    color = FAMILY_COLOR["jepa"]
    dyn = np.asarray(table["Delta_e"], dtype=float)
    for ax, (key, label) in zip(axes, axes_keys):
        x = np.asarray(table[key], dtype=float)
        tr = trends[key]
        ax.axhline(0.0, color="0.6", lw=0.8, ls="--", zorder=1)
        ax.scatter(x, dyn, s=16, c=color, alpha=0.7, edgecolors="white",
                   linewidths=0.3, zorder=3)
        ax.fill_between(tr["grid"], tr["band_lo"], tr["band_hi"], color=color,
                        alpha=0.18, lw=0, zorder=2)
        ax.plot(tr["grid"], tr["curve"], color=color, lw=1.4, zorder=4)
        ax.set_xlabel(label)
        if key == "phi":
            ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
            ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
            if not phi_well_spread:
                ax.text(0.5, 0.97, "bimodal $\\varphi$: trend weakly informative",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=6, color="0.35")
        ax.set_ylabel(r"$\Delta e = e_{\mathrm{AE}} - e_{\mathrm{JEPA}}$")
        ax.set_title(f"{tr['trend_sign']}, " r"$\rho=$" f"{tr['spearman_rho']:+.2f}",
                     fontsize=7.5)
    modename = "representational" if mode == "repr" else "forecast"
    fig.suptitle(
        rf"Predictive advantage on wake enstrophy ({modename}, $H=16$): "
        rf"$\Delta e>0$ favours JEPA  ($n={dyn.size}$)",
        fontsize=8.5, y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def build_table(mode: str, meta: dict, mach: dict) -> dict:
    e_jepa = load_per_encounter_abs_error(HEADLINE_OBS, mode, "jepa_d64")
    e_ae = load_per_encounter_abs_error(HEADLINE_OBS, mode, "fukami_d64")
    if e_jepa.shape != e_ae.shape:
        raise RuntimeError(f"shape mismatch {e_jepa.shape} vs {e_ae.shape}")
    n = e_jepa.size
    if n != len(meta["sorted_keys"]):
        raise RuntimeError(
            f"error array n={n} != metadata n={len(meta['sorted_keys'])}; "
            "canonical ordering not aligned"
        )
    phi, phi_diag = impact_phase(meta["z_full"], meta["impact_frame"], mach)
    delta = e_ae - e_jepa
    return {
        "mode": mode,
        "n": int(n),
        "case_id": meta["case_id"].tolist(),
        "encounter_index": meta["encounter_index"].astype(int).tolist(),
        "G": meta["G"].tolist(),
        "D": meta["D"].tolist(),
        "Y": meta["Y"].tolist(),
        "phi": phi.tolist(),
        "e_JEPA": e_jepa.tolist(),
        "e_AE": e_ae.tolist(),
        "Delta_e": delta.tolist(),
        "phi_diagnostic": phi_diag,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = canonical_metadata()
    mach = baseline_phase_machinery()

    payload: dict = {
        "headline_observable": HEADLINE_OBS,
        "horizon_frames": 16,
        "lowess_frac": LOWESS_FRAC,
        "bootstrap": {"n_boot": N_BOOT, "band_pct": [BAND_LO, BAND_HI], "seed": RNG_SEED},
        "phase_provenance": {
            "source": "scripts/session20/exp_phase_amplitude.py baseline limit cycle",
            "protophase": "(PC1,PC2) plane angle about orbit centroid, [0,2pi)",
            "n_baseline_episodes": mach["n_baseline_episodes"],
            "baseline_impact_frame": mach["baseline_impact"],
            "period_frames": mach["period_frames"],
            "period_tc": mach["period_tc"],
            "pc12_var_fraction": mach["pc12_var_fraction"],
            "hilbert_advance_over_period_rad": mach["hilbert_advance_over_period_rad"],
        },
        "error_provenance": (
            "scripts/session21/session21_paired_closure_stats.load_per_encounter_abs_error "
            "-> scripts/session20/exp_closure_r2 ridge probe; |probe-DNS| at impact+16; "
            "repr=z_dns, forecast=z_markov; canonical sorted (case_id, encounter) order"
        ),
        "modes": {},
    }

    figures = {}
    for mode in ("repr", "forecast"):
        table = build_table(mode, meta, mach)
        trends = {}
        for key in ("G", "D", "Y", "phi"):
            tr = lowess_with_band(np.asarray(table[key]), np.asarray(table["Delta_e"]))
            trends[key] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in tr.items()
            }
        delta = np.asarray(table["Delta_e"])
        ej = np.asarray(table["e_JEPA"])
        ea = np.asarray(table["e_AE"])
        table_summary = {
            "n": table["n"],
            "delta_mean": float(delta.mean()),
            "delta_median": float(np.median(delta)),
            "jepa_wins": int((ej < ea).sum()),
            "e_JEPA_mean": float(ej.mean()),
            "e_AE_mean": float(ea.mean()),
            "trend_direction": {k: trends[k]["trend_sign"] for k in ("G", "D", "Y", "phi")},
            "spearman_rho": {k: trends[k]["spearman_rho"] for k in ("G", "D", "Y", "phi")},
            "spearman_p": {k: trends[k]["spearman_p"] for k in ("G", "D", "Y", "phi")},
            "phi_well_spread": table["phi_diagnostic"]["phi_well_spread"],
            "phi_8bin_counts": table["phi_diagnostic"]["phi_8bin_counts"],
        }
        payload["modes"][mode] = {
            "table": table,
            "trends": trends,
            "summary": table_summary,
        }
        suffix = "" if mode == "repr" else "_forecast"
        fig_path = OUT_DIR / f"error_maps{suffix}.png"
        make_figure(table, mode, {k: trends[k] for k in ("G", "D", "Y", "phi")},
                    fig_path, mach)
        figures[mode] = str(fig_path)

    payload["figures"] = figures
    payload["ranges"] = {
        ax: {"min": float(np.min(meta[ax])), "max": float(np.max(meta[ax]))}
        for ax in ("G", "D", "Y")
    }

    json_path = OUT_DIR / "error_maps.json"
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    # ---- console report ----
    print("=" * 76)
    print("SESSION 23 -- ERROR-MAP TRENDS (predictive JEPA advantage, wake enstrophy)")
    print("=" * 76)
    print(f"phi: baseline limit-cycle protophase, period ~ {mach['period_frames']:.0f} fr "
          f"= {mach['period_tc']:.2f} tc, PC1+PC2 var {mach['pc12_var_fraction']*100:.0f}%")
    for mode in ("repr", "forecast"):
        s = payload["modes"][mode]["summary"]
        modename = "representational" if mode == "repr" else "forecast"
        print(f"\n[{modename}] n={s['n']}  e_JEPA={s['e_JEPA_mean']:.3g}  "
              f"e_AE={s['e_AE_mean']:.3g}  Delta_e mean={s['delta_mean']:+.3g}  "
              f"median={s['delta_median']:+.3g}  JEPA wins {s['jepa_wins']}/{s['n']}")
        for k in ("G", "D", "Y", "phi"):
            print(f"    {k:>3}: {s['trend_direction'][k]:>10}  "
                  f"Spearman rho={s['spearman_rho'][k]:+.3f} (p={s['spearman_p'][k]:.2g})")
    print(f"\nJSON: {json_path}")
    for mode, fp in figures.items():
        print(f"FIG ({mode}): {fp}")
    print("=" * 76)


if __name__ == "__main__":
    main()
