"""Session 23: leading-edge-vortex (LEV) tracking across decoded vorticity fields.

READ-ONLY analysis (CPU numpy/scipy; reuses already-decoded fields; no GPU, no
training). Revision plan Part V.1 (Odaka/Fukami-style large-scale structures).

Goal
----
Turn the paper's scalar "wake-enstrophy advantage" (D129/D131; the predictive
JEPA latent is the only one whose held-out wake-enstrophy closure beats the
predict-the-mean floor) and the scale-decomposition result (D137; the predictive
decode retains the lift-bearing large-scale LEV/shear that the reconstructive AE
smooths) into a MEASURED statement about the leading-edge vortex itself. We track
the dominant suction-side LEV across the decoded mid-plane omega_z fields of each
encoder family and ask, per family, how far its LEV centroid and circulation drift
from the DNS LEV at the impact frame and at horizon H=16.

Method
------
Fields (outputs/session20/decoded/{split}.npz) are pipeline-normalised 3-sigma
omega_z, shape (n_enc, 7, 192, 96). Array axes are (encounter, frame-offset, x, y):
axis 2 (192) is streamwise x in (-1.5, 4.5); axis 3 (96) is cross-stream y in
(-1.5, 1.5). Isotropic grid dx = dy = 1/32 chord (32 px/chord). Frame offsets are
[-8, 0, 8, 16, 24, 32, 40] relative to impact, so index 1 = impact, index 3 = H+16.

1. Large-scale field (EXACTLY matching scripts/session20/exp_scale_decomposition.py,
   Motoori & Goto / Odaka et al. JFM 1031 R3 2026): u_L = gaussian_filter(field,
   sigma = 0.05 * 32 = 1.6 px, mode="nearest") over the two spatial axes.
2. LEV search region: the leading-edge / wing region x/c < 1, i.e. streamwise pixel
   index < 80 (x = 0 -> index 48, x = 1 -> index 80), with the frozen
   airfoil-adjacent mask (140 inside-solid + 1-cell-neighbour cells) removed so the
   body itself does not enter the lobe.
3. The suction-side LEV at alpha = 14 deg has NEGATIVE omega_z (clockwise, with the
   convention omega_z = du/dy - dv/dx). We threshold the large-scale field below a
   single ABSOLUTE iso-level OMEGA_THR (identical across families; vortex-ID iso-
   contour), label connected components (scipy.ndimage.label), and take the strongest
   component (largest integrated |omega_z|). Centroid (x_LEV, y_LEV) is the
   |omega|-weighted center_of_mass (scipy.ndimage.center_of_mass) in physical coords;
   peak |omega_z| is the lobe max; signed circulation is
   Gamma_LEV = sum_{lobe} omega_z * dx * dy.

   An absolute iso-level (not a per-field fraction of that field's own peak) is
   required for a FAIR cross-family comparison: the reconstructive (Fukami) decode
   collapses the large-scale LEV amplitude to ~11% of DNS (D137), so a
   fraction-of-own-peak threshold would inflate a diffuse smudge into a spurious
   "LEV". With a fixed iso-level a family whose LEV has been smoothed below the level
   is recorded as "LEV not detected" (centroid NaN, Gamma 0), which is the physically
   correct statement; we report the per-family detection rate alongside the metrics.

4. Per family and stage (impact, H=16): centroid distance to DNS, |Gamma_LEV -
   Gamma_LEV_DNS|, retained-circulation fraction Gamma_LEV / Gamma_LEV_DNS, peak-
   amplitude ratio. Centroid / circulation-error statistics are pooled over the
   encounters where BOTH DNS and that family detect an LEV (so the distance is
   defined); the detection rate is reported separately.
5. Wake-enstrophy <-> LEV link: per-encounter large-scale wake enstrophy is
   recomputed with the IDENTICAL wake mask and definition as
   exp_scale_decomposition.py (0.5 * sum_{wake px} u_L^2, wake = streamwise index
   >= 48 with the airfoil mask removed). Per-encounter wake-enstrophy error is
   |Omega_w(family) - Omega_w(DNS)|; per-encounter LEV-circulation error is
   |Gamma_LEV(family) - Gamma_LEV(DNS)|. We report the Spearman rank correlation
   between the two, pooled over families and over (impact, H=16), on the encounters
   where the LEV is detected.

Gate
----
At H=16 on test_b: is the predictive (JEPA d=64) decode's LEV-centroid distance AND
circulation error SMALLER than the reconstructive (Fukami d=64) decode's, and is the
per-encounter Spearman > 0? If not, that contradicts D129/D131 (wake-enstrophy gap)
and D137 (scale decomposition); the script prints the contradiction rather than
writing around it.

Output
------
outputs/session23/lev_tracking/lev.json   (all numbers + config + gate)
outputs/session23/lev_tracking/lev.csv     (per-family per-stage per-split table)
outputs/session23/lev_tracking/lev_tracking.png  (3 panels)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import center_of_mass, gaussian_filter, label
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DECODED_ROOT = REPO / "outputs" / "session20" / "decoded"
MASK_PATH = REPO / "outputs" / "data_pipeline" / "v1" / "airfoil_adjacent_mask.npy"
OUT_DIR = REPO / "outputs" / "session23" / "lev_tracking"

# ---- grid conventions (CLAUDE.md; identical to exp_scale_decomposition.py) ----
PX_PER_CHORD = 32
X_EXT = (-1.5, 4.5)
Y_EXT = (-1.5, 1.5)
H_PX, W_PX = 192, 96
DX = (X_EXT[1] - X_EXT[0]) / H_PX  # = 1/32
DY = (Y_EXT[1] - Y_EXT[0]) / W_PX  # = 1/32

SIGMA_OVER_C = 0.05
SIGMA_PX = SIGMA_OVER_C * PX_PER_CHORD  # = 1.6 px (matches scale decomposition EXACTLY)

# leading-edge / wing region for the LEV search: x/c < 1 -> streamwise index < 80.
LE_X_INDEX = int(round((0.0 - X_EXT[0]) / DX))  # = 48 (x = 0)
TE_X_INDEX = int(round((1.0 - X_EXT[0]) / DX))  # = 80 (x = 1)

# Absolute large-scale-omega iso-level for LEV identification (normalised units).
# DNS LE-region negative-omega peaks span min/median/max ~ 0.44/1.49/5.16 (impact)
# and 0.67/1.27/5.19 (H=16); a level of 0.5 detects the LEV in ~all DNS/JEPA/POD
# encounters and correctly flags the collapsed-amplitude Fukami fields.
OMEGA_THR = 0.5

SPLITS = ("test_a", "test_b", "test_c")
# (npz key, canonical family label, figstyle family tag)
METHODS = [
    ("jepa_norm", "jepa_d64", "jepa"),
    ("fukami_norm", "fukami", "fukami"),
    ("pod_norm", "pod", "pod"),
]
# The three families the figure / gate compare. jepa_d32 is loaded too for the table.
METHODS_ALL = METHODS + [("jepa_d32_norm", "jepa_d32", "jepa")]

OFFSETS = [-8, 0, 8, 16, 24, 32, 40]
IMPACT_IDX = 1
PLUS16_IDX = 3  # offset +16
STAGES = (("impact", IMPACT_IDX), ("H16", PLUS16_IDX))


def load_masks() -> tuple[np.ndarray, np.ndarray]:
    """Return (airfoil_mask, wake_mask), both boolean (192, 96)."""
    airfoil = np.load(MASK_PATH).astype(bool)  # 140 inside-solid + adjacent cells
    wake = np.zeros((H_PX, W_PX), dtype=bool)
    wake[LE_X_INDEX:, :] = True  # downstream half (streamwise index >= 48)
    wake &= ~airfoil
    return airfoil, wake


def large_scale(field: np.ndarray) -> np.ndarray:
    """Gaussian large-scale field u_L on the last two (spatial) axes.

    field: (..., 192, 96). Matches exp_scale_decomposition.scale_decompose exactly.
    """
    sig = [0.0] * (field.ndim - 2) + [SIGMA_PX, SIGMA_PX]
    return gaussian_filter(field, sigma=sig, mode="nearest")


def wake_enstrophy_per_enc(field: np.ndarray, wake: np.ndarray) -> np.ndarray:
    """Large-scale wake enstrophy 0.5 * sum_{wake px} u_L^2 per (enc, offset).

    field: (n, 7, 192, 96) raw (normalised) -> filtered here. Returns (n, 7).
    Identical definition to exp_scale_decomposition.wake_enstrophy.
    """
    fL = large_scale(field)
    masked = fL * wake[None, None, :, :]
    return 0.5 * (masked**2).sum(axis=(2, 3))


def lev_from_field(
    field2d_L: np.ndarray, airfoil: np.ndarray, thr: float = OMEGA_THR
) -> dict | None:
    """Extract the dominant suction-side (negative) LEV from a large-scale field.

    field2d_L: (192, 96) large-scale normalised omega_z (already filtered).
    Returns dict(x, y, peak, Gamma, size) in physical coords, or None if no
    large-scale negative lobe exceeds the absolute iso-level in the x/c < 1 region.
    """
    reg = field2d_L.copy()
    reg[TE_X_INDEX:, :] = 0.0  # restrict to leading-edge / wing region x/c < 1
    reg[airfoil] = 0.0  # remove the body and its 1-cell neighbourhood

    binary = reg < -thr  # suction-side LEV is negative omega_z
    if not binary.any():
        return None
    lab, n = label(binary)
    # strongest component = largest integrated |omega_z|
    strengths = [float((-reg[lab == i]).sum()) for i in range(1, n + 1)]
    k = int(np.argmax(strengths)) + 1
    sel = lab == k

    weights = np.where(sel, -reg, 0.0)  # positive |omega| weights on the lobe
    ci, cj = center_of_mass(weights)
    x_lev = X_EXT[0] + (ci + 0.5) * DX
    y_lev = Y_EXT[0] + (cj + 0.5) * DY
    peak = float(-reg[sel].min())
    gamma = float(reg[sel].sum() * DX * DY)  # signed circulation (negative)
    return {"x": x_lev, "y": y_lev, "peak": peak, "Gamma": gamma, "size": int(sel.sum())}


def boot_ci_mean(vals: np.ndarray, rng, n: int = 2000, ci: float = 0.95):
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 3:
        return float("nan"), float("nan")
    stats = [vals[rng.integers(0, vals.size, vals.size)].mean() for _ in range(n)]
    a = (1 - ci) / 2
    return float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a))


def process_split(split: str, airfoil: np.ndarray, wake: np.ndarray, rng) -> dict:
    blob = np.load(DECODED_ROOT / f"{split}.npz", allow_pickle=True)
    target = blob["target_norm"].astype(np.float32)
    n_enc = target.shape[0]

    # --- per-encounter LEV features for DNS and every method, at every stage ---
    # lev[label][stage] -> list of (dict|None), length n_enc.
    lev = {"dns": {}}
    tgt_L = large_scale(target)  # (n, 7, 192, 96)
    for stage, idx in STAGES:
        lev["dns"][stage] = [lev_from_field(tgt_L[i, idx], airfoil) for i in range(n_enc)]

    method_L = {}
    for key, label_, _fam in METHODS_ALL:
        fL = large_scale(blob[key].astype(np.float32))
        method_L[label_] = fL
        lev[label_] = {}
        for stage, idx in STAGES:
            lev[label_][stage] = [lev_from_field(fL[i, idx], airfoil) for i in range(n_enc)]

    # --- per-encounter large-scale wake enstrophy (identical to scale decomp) ---
    ens = {"dns": wake_enstrophy_per_enc(target, wake)}
    for key, label_, _fam in METHODS_ALL:
        ens[label_] = wake_enstrophy_per_enc(blob[key].astype(np.float32), wake)

    out: dict = {"n_enc": int(n_enc), "metrics": {}, "_lev": lev, "_ens": ens}

    # --- aggregate metrics per method and stage ---
    for key, label_, _fam in METHODS_ALL:
        out["metrics"][label_] = {}
        for stage, idx in STAGES:
            dns_lev = lev["dns"][stage]
            fam_lev = lev[label_][stage]

            both = [
                i for i in range(n_enc) if dns_lev[i] is not None and fam_lev[i] is not None
            ]
            det_dns = sum(d is not None for d in dns_lev)
            det_fam = sum(d is not None for d in fam_lev)

            cdist, gerr, gfrac, peakratio = [], [], [], []
            for i in both:
                dd, ff = dns_lev[i], fam_lev[i]
                cdist.append(np.hypot(ff["x"] - dd["x"], ff["y"] - dd["y"]))
                gerr.append(abs(ff["Gamma"] - dd["Gamma"]))
                if abs(dd["Gamma"]) > 1e-12:
                    gfrac.append(ff["Gamma"] / dd["Gamma"])
                if abs(dd["peak"]) > 1e-12:
                    peakratio.append(ff["peak"] / dd["peak"])

            cdist = np.array(cdist)
            gerr = np.array(gerr)
            clo, chi = boot_ci_mean(cdist, rng)
            glo, ghi = boot_ci_mean(gerr, rng)
            out["metrics"][label_][stage] = {
                "n_both": len(both),
                "detection_rate_dns": det_dns / n_enc,
                "detection_rate_family": det_fam / n_enc,
                "centroid_dist_mean": float(np.nanmean(cdist)) if cdist.size else float("nan"),
                "centroid_dist_median": (
                    float(np.nanmedian(cdist)) if cdist.size else float("nan")
                ),
                "centroid_dist_ci": [clo, chi],
                "gamma_abs_err_mean": float(np.nanmean(gerr)) if gerr.size else float("nan"),
                "gamma_abs_err_median": (
                    float(np.nanmedian(gerr)) if gerr.size else float("nan")
                ),
                "gamma_abs_err_ci": [glo, ghi],
                "retained_circ_frac_mean": (
                    float(np.nanmean(gfrac)) if gfrac else float("nan")
                ),
                "peak_ratio_mean": float(np.nanmean(peakratio)) if peakratio else float("nan"),
                "mean_x_lev_family": (
                    float(np.nanmean([fam_lev[i]["x"] for i in both])) if both else float("nan")
                ),
                "mean_y_lev_family": (
                    float(np.nanmean([fam_lev[i]["y"] for i in both])) if both else float("nan")
                ),
                "mean_gamma_family": (
                    float(np.nanmean([fam_lev[i]["Gamma"] for i in both]))
                    if both
                    else float("nan")
                ),
                "mean_x_lev_dns": (
                    float(np.nanmean([dns_lev[i]["x"] for i in both])) if both else float("nan")
                ),
                "mean_y_lev_dns": (
                    float(np.nanmean([dns_lev[i]["y"] for i in both])) if both else float("nan")
                ),
                "mean_gamma_dns": (
                    float(np.nanmean([dns_lev[i]["Gamma"] for i in both]))
                    if both
                    else float("nan")
                ),
            }
    return out


def compute_spearman(results: dict, split: str = "test_b") -> dict:
    """Per-encounter Spearman: wake-enstrophy error vs LEV-circulation error.

    Pooled over the three compared families (jepa_d64, fukami, pod) and over
    (impact, H=16), on encounters where the family LEV is detected. Returns the
    pooled rho plus the per-family/per-stage points (for the scatter panel).
    """
    res = results[split]
    lev, ens = res["_lev"], res["_ens"]
    n_enc = res["n_enc"]
    dns_ens = ens["dns"]

    pooled_x, pooled_y = [], []  # x = wake-ens error, y = LEV-circ error
    points = {}  # (family_tag) -> {"wake_err": [...], "lev_err": [...], "stage": [...]}
    for key, label_, fam in METHODS:  # the three compared families only
        points[label_] = {"wake_err": [], "lev_err": [], "stage": []}
        for stage, idx in STAGES:
            for i in range(n_enc):
                dd = lev["dns"][stage][i]
                ff = lev[label_][stage][i]
                if dd is None or ff is None:
                    continue
                we = abs(float(ens[label_][i, idx]) - float(dns_ens[i, idx]))
                le = abs(ff["Gamma"] - dd["Gamma"])
                pooled_x.append(we)
                pooled_y.append(le)
                points[label_]["wake_err"].append(we)
                points[label_]["lev_err"].append(le)
                points[label_]["stage"].append(stage)
    rho, pval = (float("nan"), float("nan"))
    if len(pooled_x) >= 3:
        rho, pval = spearmanr(pooled_x, pooled_y)
    return {
        "rho": float(rho),
        "p_value": float(pval),
        "n_points": len(pooled_x),
        "_points": points,
        "_pooled": (np.array(pooled_x), np.array(pooled_y)),
    }


def make_figure(results: dict, spear: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(REPO / "scripts" / "session21"))
    import figstyle as fs  # noqa: E402

    fs.use_style()
    FAM = {"jepa_d64": "jepa", "fukami": "fukami", "pod": "pod"}
    stages = ["impact", "H16"]
    stage_x = [0, 16]  # horizon axis (frames from impact)

    fig, axes = plt.subplots(1, 3, figsize=(fs.TEXTWIDTH_IN, fs.TEXTWIDTH_IN * 0.34))

    # Panel 1: x_LEV centroid error vs horizon (test_b).
    ax = axes[0]
    rb = results["test_b"]["metrics"]
    for label_ in ("jepa_d64", "fukami", "pod"):
        fam = FAM[label_]
        ys = [rb[label_][s]["centroid_dist_mean"] for s in stages]
        ax.plot(
            stage_x,
            ys,
            marker=fs.FAMILY_MARKER[fam],
            color=fs.FAMILY_COLOR[fam],
            label=fs.FAMILY_LABEL[fam],
            lw=1.1,
            ms=4.0,
        )
    ax.set_xlabel("frames from impact")
    ax.set_ylabel("LEV centroid error (chords)")
    ax.set_xticks(stage_x)
    ax.set_title("(a) LEV position error", fontsize=8.5)
    ax.legend(fontsize=6.0, loc="upper left")

    # Panel 2: |Gamma_LEV - Gamma_DNS| vs horizon (test_b).
    ax = axes[1]
    for label_ in ("jepa_d64", "fukami", "pod"):
        fam = FAM[label_]
        ys = [rb[label_][s]["gamma_abs_err_mean"] for s in stages]
        ax.plot(
            stage_x,
            ys,
            marker=fs.FAMILY_MARKER[fam],
            color=fs.FAMILY_COLOR[fam],
            label=fs.FAMILY_LABEL[fam],
            lw=1.1,
            ms=4.0,
        )
    ax.set_xlabel("frames from impact")
    ax.set_ylabel(r"$|\Gamma_{\rm LEV}-\Gamma_{\rm LEV}^{\rm DNS}|$")
    ax.set_xticks(stage_x)
    ax.set_title("(b) LEV circulation error", fontsize=8.5)

    # Panel 3: scatter wake-enstrophy error vs LEV-circulation error (test_b).
    ax = axes[2]
    pts = spear["_points"]
    for label_ in ("jepa_d64", "fukami", "pod"):
        fam = FAM[label_]
        ax.scatter(
            pts[label_]["wake_err"],
            pts[label_]["lev_err"],
            s=10,
            color=fs.FAMILY_COLOR[fam],
            marker=fs.FAMILY_MARKER[fam],
            alpha=0.7,
            edgecolors="none",
            label=fs.FAMILY_LABEL[fam],
        )
    ax.set_xlabel(r"wake-enstrophy error $|\Omega_w-\Omega_w^{\rm DNS}|$")
    ax.set_ylabel(r"LEV-circ. error $|\Delta\Gamma_{\rm LEV}|$")
    ax.set_title(
        f"(c) Spearman $\\rho={spear['rho']:.2f}$ (n={spear['n_points']})", fontsize=8.5
    )
    ax.legend(fontsize=6.0, loc="lower right")

    fig.tight_layout(pad=0.4)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"lev_tracking.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[lev] wrote figure {OUT_DIR / 'lev_tracking.png'}")


def write_csv(results: dict) -> None:
    rows = []
    for split in SPLITS:
        m = results[split]["metrics"]
        for label_ in [x[1] for x in METHODS_ALL]:
            for stage, _ in STAGES:
                t = m[label_][stage]
                rows.append(
                    {
                        "split": split,
                        "family": label_,
                        "stage": stage,
                        "n_enc": results[split]["n_enc"],
                        "n_both_detected": t["n_both"],
                        "detection_rate_family": round(t["detection_rate_family"], 4),
                        "detection_rate_dns": round(t["detection_rate_dns"], 4),
                        "centroid_dist_mean": _r(t["centroid_dist_mean"]),
                        "centroid_dist_median": _r(t["centroid_dist_median"]),
                        "centroid_dist_ci_lo": _r(t["centroid_dist_ci"][0]),
                        "centroid_dist_ci_hi": _r(t["centroid_dist_ci"][1]),
                        "gamma_abs_err_mean": _r(t["gamma_abs_err_mean"]),
                        "gamma_abs_err_median": _r(t["gamma_abs_err_median"]),
                        "gamma_abs_err_ci_lo": _r(t["gamma_abs_err_ci"][0]),
                        "gamma_abs_err_ci_hi": _r(t["gamma_abs_err_ci"][1]),
                        "retained_circ_frac_mean": _r(t["retained_circ_frac_mean"]),
                        "peak_ratio_mean": _r(t["peak_ratio_mean"]),
                        "mean_x_lev_family": _r(t["mean_x_lev_family"]),
                        "mean_y_lev_family": _r(t["mean_y_lev_family"]),
                        "mean_gamma_family": _r(t["mean_gamma_family"]),
                        "mean_x_lev_dns": _r(t["mean_x_lev_dns"]),
                        "mean_y_lev_dns": _r(t["mean_y_lev_dns"]),
                        "mean_gamma_dns": _r(t["mean_gamma_dns"]),
                    }
                )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "lev.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[lev] wrote {OUT_DIR / 'lev.csv'} ({len(rows)} rows)")


def _r(x, nd=5):
    return None if (x is None or (isinstance(x, float) and not np.isfinite(x))) else round(x, nd)


def evaluate_gate(results: dict, spear: dict) -> dict:
    """The explicit gate on test_b at H=16: predictive < reconstructive on LEV
    centroid AND circulation error, and Spearman > 0."""
    m = results["test_b"]["metrics"]
    j = m["jepa_d64"]["H16"]
    fk = m["fukami"]["H16"]
    cen_ok = j["centroid_dist_mean"] < fk["centroid_dist_mean"]
    cir_ok = j["gamma_abs_err_mean"] < fk["gamma_abs_err_mean"]
    spear_ok = spear["rho"] > 0
    passed = bool(cen_ok and cir_ok and spear_ok)

    diagnosis = ""
    if not passed:
        bits = []
        if not cen_ok:
            bits.append(
                f"JEPA centroid error {j['centroid_dist_mean']:.3f} >= Fukami "
                f"{fk['centroid_dist_mean']:.3f}"
            )
        if not cir_ok:
            bits.append(
                f"JEPA circ error {j['gamma_abs_err_mean']:.4f} >= Fukami "
                f"{fk['gamma_abs_err_mean']:.4f}"
            )
        if not spear_ok:
            bits.append(f"Spearman rho = {spear['rho']:.3f} <= 0")
        diagnosis = (
            "CONTRADICTS D129/D131 (wake-enstrophy gap) and D137 (scale decomp). "
            + "; ".join(bits)
            + ". Candidate causes to check: frame indexing (impact=offset idx 1, "
            "H16=idx 3 per OFFSETS), omega sign (suction-side LEV is NEGATIVE), "
            "large-scale filter (sigma=1.6 px / mode=nearest, matching scale decomp), "
            "absolute iso-level OMEGA_THR. NOTE: Fukami detection rate at H=16 is "
            f"{fk['detection_rate_family']:.2f} on only {fk['n_both']} commonly-"
            "detected encounters, so its centroid/circ stats come from its few "
            "non-collapsed fields and may not be representative."
        )
    return {
        "passed": passed,
        "centroid_smaller_than_fukami": bool(cen_ok),
        "circulation_smaller_than_fukami": bool(cir_ok),
        "spearman_positive": bool(spear_ok),
        "jepa_centroid_dist_H16": _r(j["centroid_dist_mean"]),
        "fukami_centroid_dist_H16": _r(fk["centroid_dist_mean"]),
        "jepa_gamma_abs_err_H16": _r(j["gamma_abs_err_mean"]),
        "fukami_gamma_abs_err_H16": _r(fk["gamma_abs_err_mean"]),
        "jepa_detection_rate_H16": _r(j["detection_rate_family"]),
        "fukami_detection_rate_H16": _r(fk["detection_rate_family"]),
        "spearman_rho": _r(spear["rho"]),
        "spearman_p": _r(spear["p_value"]),
        "diagnosis": diagnosis,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    airfoil, wake = load_masks()
    print(
        f"[lev] sigma = {SIGMA_PX:.2f} px (sigma/c = {SIGMA_OVER_C}); abs iso-level "
        f"|omega_L| > {OMEGA_THR}; LE region x/c<1 = streamwise idx < {TE_X_INDEX}; "
        f"wake pixels = {int(wake.sum())}"
    )

    results = {}
    for split in SPLITS:
        results[split] = process_split(split, airfoil, wake, rng)
        print(f"\n[{split}] n={results[split]['n_enc']}  LEV tracking vs DNS")
        for label_ in [x[1] for x in METHODS_ALL]:
            for stage, _ in STAGES:
                t = results[split]["metrics"][label_][stage]
                print(
                    f"  {label_:9s} @{stage:6s} det={t['detection_rate_family']:.2f} "
                    f"(n_both={t['n_both']:2d}) "
                    f"cdist={t['centroid_dist_mean']:.3f} "
                    f"|dGamma|={t['gamma_abs_err_mean']:.4f} "
                    f"retainedfrac={t['retained_circ_frac_mean']:.2f} "
                    f"peakratio={t['peak_ratio_mean']:.2f}"
                )

    spear = compute_spearman(results, split="test_b")
    print(
        f"\n[lev] test_b per-encounter Spearman (wake-ens err vs LEV-circ err, "
        f"pooled families x stages): rho = {spear['rho']:.3f} "
        f"(p = {spear['p_value']:.2e}, n = {spear['n_points']})"
    )

    gate = evaluate_gate(results, spear)
    print("\n[lev] ===== GATE (test_b, H=16) =====")
    print(
        f"  predictive centroid < reconstructive : {gate['centroid_smaller_than_fukami']} "
        f"(JEPA {gate['jepa_centroid_dist_H16']} vs Fukami {gate['fukami_centroid_dist_H16']})"
    )
    print(
        f"  predictive circ-err < reconstructive : {gate['circulation_smaller_than_fukami']} "
        f"(JEPA {gate['jepa_gamma_abs_err_H16']} vs Fukami {gate['fukami_gamma_abs_err_H16']})"
    )
    print(f"  Spearman > 0                         : {gate['spearman_positive']}")
    print(f"  GATE PASSED                          : {gate['passed']}")
    if gate["diagnosis"]:
        print(f"  DIAGNOSIS: {gate['diagnosis']}")

    make_figure(results, spear)
    write_csv(results)

    serial = {
        "config": {
            "sigma_px": SIGMA_PX,
            "sigma_over_c": SIGMA_OVER_C,
            "px_per_chord": PX_PER_CHORD,
            "omega_iso_level": OMEGA_THR,
            "le_region": f"streamwise index < {TE_X_INDEX} (x/c < 1)",
            "le_x_index": LE_X_INDEX,
            "te_x_index": TE_X_INDEX,
            "lev_sign": "negative omega_z (suction-side, clockwise; du/dy - dv/dx)",
            "centroid": "|omega|-weighted center_of_mass of strongest connected component",
            "circulation": "Gamma_LEV = sum_{lobe} omega_z * dx * dy (signed, normalised)",
            "wake_enstrophy": "0.5 * sum_{wake px} u_L^2 (identical to exp_scale_decomposition)",
            "offsets": OFFSETS,
            "impact_index": IMPACT_IDX,
            "h16_index": PLUS16_IDX,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "decoded_source": str(DECODED_ROOT),
            "note": "Absolute iso-level (not fraction-of-own-peak) for fair cross-family "
            "comparison; collapsed-amplitude fields record LEV-not-detected, reported "
            "via detection_rate_family.",
        },
        "splits": {
            s: {
                "n_enc": results[s]["n_enc"],
                "metrics": results[s]["metrics"],
            }
            for s in SPLITS
        },
        "spearman_test_b": {
            "rho": spear["rho"],
            "p_value": spear["p_value"],
            "n_points": spear["n_points"],
            "definition": "per-encounter |Omega_w - Omega_w_DNS| vs |Gamma_LEV - Gamma_LEV_DNS|, "
            "pooled over jepa_d64/fukami/pod and (impact, H=16), LEV-detected encounters",
        },
        "gate": gate,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "lev.json", "w") as f:
        json.dump(serial, f, indent=2)
    print(f"[lev] wrote {OUT_DIR / 'lev.json'}")


if __name__ == "__main__":
    main()
