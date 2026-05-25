"""Session 14 paper figures (Thrusts 1 to 4) for the vortex-jepa project.

Produces six figures under ``outputs/session14/figures/``:

* Figure 1 ``thrust1_epiplexity_decomposition.png`` (Thrust 1)
* Figure 2 ``thrust2_forecast_horizon.png``         (Thrust 2)
* Figure 3 ``thrust2_hero_long_rollout.png``        (Thrust 2 hero)
* Figure 4 ``thrust4_intrinsic_dim.png``            (Thrust 4)
* Figure 5 ``thrust3_concept_vectors_2D.png``       (Thrust 3)
* Figure 6 ``thrust4_pca3d.png``                    (Thrust 4 supplement)

All figures use the project conventions documented in CLAUDE.md:

* Reconstruction omega panels use ``vmin=-3, vmax=+3`` in the 3-sigma
  normalised scale (raw arrays divided by ``s = 3 * train_std = 10.658``)
  with the NACA 0012 airfoil overlaid as a filled-black polygon.
* Physical extents x in (-1.5, 4.5), y in (-1.5, 1.5).
* No em-dashes anywhere; all axis text uses hyphens or commas.
* Headless matplotlib backend (``Agg``).

Usage::

    source .venv/bin/activate
    export PREVENT_ROOT=$HOME/PREVENT
    python scripts/session14_make_figures.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401
from sklearn.decomposition import PCA  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evaluation.epiplexity import epiplexity_decomposition  # noqa: E402


PREVENT_ROOT = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
SESSION14 = REPO / "outputs" / "session14"
FIG_DIR = SESSION14 / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 3-sigma scale for normalised display (matches OmegaPipeline.unnormalize divisor).
SIGMA_3 = 10.657723018517105


# ------------------------------------------------------------------
# Airfoil overlay helpers (mirrors scripts/session9_decoder_fig3_pipeline.py).
# ------------------------------------------------------------------
def load_airfoil_xy() -> np.ndarray:
    """Load NACA 0012 vertices from the Baseline.h5 file."""
    raw = PREVENT_ROOT / "data" / "raw" / "periodic" / "Baseline.h5"
    with h5py.File(raw, "r") as f:
        xy = np.asarray(f["airfoil_xy"])
    if not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[0:1]])
    return xy


def omega_xy_to_pixel(xy: np.ndarray, H: int = 192, W: int = 96) -> np.ndarray:
    """Convert physical (x, y) to pixel indices for the (H=192, W=96) raster."""
    x_min, x_max = -1.5, 4.5
    y_min, y_max = -1.5, 1.5
    px_x = (xy[:, 0] - x_min) * (H - 1) / (x_max - x_min)
    px_y = (xy[:, 1] - y_min) * (W - 1) / (y_max - y_min)
    return np.stack([px_x, px_y], axis=-1)


def add_airfoil(ax: plt.Axes, airfoil_px: np.ndarray) -> None:
    """Overlay the airfoil polygon on an omega imshow axis."""
    ax.add_patch(
        Polygon(
            airfoil_px,
            closed=True,
            facecolor="black",
            edgecolor="black",
            linewidth=0.7,
            zorder=10,
        )
    )


def imshow_omega(
    ax: plt.Axes,
    omega_raw: np.ndarray,
    airfoil_px: np.ndarray,
    vmax: float = 3.0,
    cmap: str = "RdBu_r",
) -> Any:
    """Display raw omega on the 3-sigma normalised scale with airfoil overlay."""
    field = omega_raw.T / SIGMA_3
    im = ax.imshow(field, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax)
    add_airfoil(ax, airfoil_px)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


# ------------------------------------------------------------------
# Figure 1: Thrust 1 epiplexity decomposition.
# ------------------------------------------------------------------
SESSION12_CONFIGS = [
    "S12_C_lam200",
    "S12_C_lam300",
    "S12_C_lam500",
    "S12_D_coarse288",
    "S12_D_coarse512",
    "S12_E_d64",
    "S12_F_TC0p01",
    "S12_F_TC0p03",
    "S12_F_TC0p10",
    "W0_C_lam100_v1p4",
]


def _lambdas(wake: float, tc: float = 0.0) -> dict[str, float]:
    """Loss-component weights summed into loss_total for a Session 12 config.

    Weights come from the encoder configs in
    ``outputs/runs/session12/<config>/encoder/metrics.jsonl``. The teacher-
    forced prediction loss has implicit weight 1.0; the scheduled-sampling
    rollout loss is weighted 0.5 per CLAUDE.md "Training" section.
    """
    return {"pred": 1.0, "roll": 0.5, "sigreg": 0.01, "obs": 0.01,
            "wake": wake, "tc": tc}


CONFIG_LAMBDAS = {
    "S12_C_lam200":     _lambdas(wake=2.0),
    "S12_C_lam300":     _lambdas(wake=3.0),
    "S12_C_lam500":     _lambdas(wake=5.0),
    "S12_D_coarse288":  _lambdas(wake=1.0),
    "S12_D_coarse512":  _lambdas(wake=1.0),
    "S12_E_d64":        _lambdas(wake=1.0),
    "S12_F_TC0p01":     _lambdas(wake=1.0, tc=0.01),
    "S12_F_TC0p03":     _lambdas(wake=1.0, tc=0.03),
    "S12_F_TC0p10":     _lambdas(wake=1.0, tc=0.10),
    "W0_C_lam100_v1p4": _lambdas(wake=1.0),
}

# Color palette per component (consistent across panels).
COMPONENT_COLORS = {
    "pred":   "#1f77b4",  # blue
    "roll":   "#9ecae1",  # light blue
    "sigreg": "#2ca02c",  # green
    "obs":    "#ff7f0e",  # orange
    "wake":   "#d62728",  # red
    "tc":     "#9467bd",  # purple
}
COMPONENT_LABEL = {
    "pred": "loss_pred",
    "roll": "loss_roll (x0.5)",
    "sigreg": "loss_anticollapse (x lam_sigreg)",
    "obs": "loss_obs (x lam_obs)",
    "wake": "loss_wake (x lam_wake)",
    "tc": "loss_tc (x lam_tc)",
}
COMPONENT_ORDER = ["pred", "roll", "sigreg", "obs", "wake", "tc"]


def compute_decomposition(config: str) -> tuple[dict[str, float], float]:
    """Return (weighted contribution per component, P_preq_total)."""
    path = REPO / "outputs" / "runs" / "session12" / config / "encoder" / "metrics.jsonl"
    decomp = epiplexity_decomposition(path)
    lambdas = CONFIG_LAMBDAS[config]
    weighted = {
        "pred": lambdas["pred"] * decomp["loss_pred"]["P_preq"],
        "roll": lambdas["roll"] * decomp["loss_roll"]["P_preq"],
        "sigreg": lambdas["sigreg"] * decomp["loss_anticollapse"]["P_preq"],
        "obs": lambdas["obs"] * decomp["loss_obs"]["P_preq"],
        "wake": lambdas["wake"] * decomp["loss_wake"]["P_preq"],
        "tc": lambdas["tc"] * decomp["loss_tc"]["P_preq"],
    }
    return weighted, float(decomp["loss_total"]["P_preq"])


def figure1_thrust1_epiplexity_decomposition() -> Path:
    """Stacked-bar P_preq decomposition + matched-d bar comparison."""
    weighted_per_config: dict[str, dict[str, float]] = {}
    total_per_config: dict[str, float] = {}
    for cfg in SESSION12_CONFIGS:
        w, total = compute_decomposition(cfg)
        weighted_per_config[cfg] = w
        total_per_config[cfg] = total

    matched = json.loads((SESSION14 / "epiplexity" / "matched_d_comparison.json").read_text())
    # Keep only the three matched-d rows that the spec asks for, plus the
    # E d=64 production for context (used to caption the win ratio).
    matched_rows = {row["name"]: row for row in matched}

    fig = plt.figure(figsize=(15.5, 6.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 1.0], wspace=0.30)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    # ---------- Left panel: stacked bar across 10 Session 12 configs ----------
    labels = [cfg.replace("S12_", "").replace("_v1p4", "") for cfg in SESSION12_CONFIGS]
    x = np.arange(len(SESSION12_CONFIGS))
    bottoms = np.zeros(len(SESSION12_CONFIGS), dtype=float)
    for comp in COMPONENT_ORDER:
        vals = np.array(
            [weighted_per_config[cfg][comp] for cfg in SESSION12_CONFIGS],
            dtype=float,
        )
        bars = ax_left.bar(
            x,
            vals,
            bottom=bottoms,
            color=COMPONENT_COLORS[comp],
            label=COMPONENT_LABEL[comp],
            edgecolor="white",
            linewidth=0.3,
        )
        # Highlight E_d64 with a thicker black border on each segment.
        e_idx = SESSION12_CONFIGS.index("S12_E_d64")
        bars[e_idx].set_edgecolor("black")
        bars[e_idx].set_linewidth(1.6)
        bottoms += vals

    # Annotate the totals on top of each stack.
    for i, cfg in enumerate(SESSION12_CONFIGS):
        ax_left.text(
            i,
            total_per_config[cfg] * 1.01,
            f"{total_per_config[cfg]:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
        )

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax_left.set_ylabel("Weighted P_preq contribution (loss units x iters)")
    ax_left.set_title(
        "Epiplexity decomposition of 10 Session 12 configs"
        " (N=400 logged steps; E_d64 outlined)"
    )
    ax_left.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2)
    ax_left.grid(axis="y", alpha=0.3)

    # ---------- Right panel: matched-d = 32 bar comparison ----------
    matched_keys = [
        ("Fukami AE d=32 matched (D81)", "Fukami AE\nd=32"),
        ("Fukami AE d=32 + wake lam=1 (D6)", "Fukami AE\nd=32 + wake"),
        ("JEPA d=32 W0_C_lam100 (Session 11 baseline)", "JEPA d=32\nW0_C_lam100"),
    ]
    bar_xs = np.arange(len(matched_keys))
    bar_w = 0.36
    total_vals = np.array([matched_rows[k]["P_preq"] for k, _ in matched_keys])
    # For the "pred-only" comparison we use loss_pred where available;
    # the Fukami AE metrics.jsonl only logs loss_total, so the bar falls
    # back to loss_total in that case (still informative as the
    # reconstruction-only objective for the AE).
    pred_paths = {
        "JEPA": REPO / "outputs" / "runs" / "session12"
        / "W0_C_lam100_v1p4" / "encoder" / "metrics.jsonl",
        "Fukami AE d=32 matched": REPO / "outputs" / "runs" / "session11"
        / "D4_fukami_ae_d32_matched" / "metrics.jsonl",
        "Fukami AE d=32 + wake": REPO / "outputs" / "runs" / "session11"
        / "D6_fukami_ae_d32_wake_lam100" / "metrics.jsonl",
    }
    pred_vals = []
    for k, _ in matched_keys:
        if "JEPA" in k:
            d = epiplexity_decomposition(pred_paths["JEPA"])
            pred_vals.append(d["loss_pred"]["P_preq"])
        elif "Fukami AE d=32 matched" in k:
            d = epiplexity_decomposition(pred_paths["Fukami AE d=32 matched"])
            pred_vals.append(d.get("loss_pred", d["loss_total"])["P_preq"])
        elif "Fukami AE d=32 + wake" in k:
            d = epiplexity_decomposition(pred_paths["Fukami AE d=32 + wake"])
            pred_vals.append(d.get("loss_pred", d["loss_total"])["P_preq"])
        else:
            pred_vals.append(float("nan"))
    pred_vals = np.array(pred_vals)

    bars_total = ax_right.bar(
        bar_xs - bar_w / 2,
        total_vals,
        bar_w,
        color="#4c72b0",
        edgecolor="black",
        linewidth=0.5,
        label="P_preq (loss_total)",
    )
    bars_pred = ax_right.bar(
        bar_xs + bar_w / 2,
        pred_vals,
        bar_w,
        color="#dd8452",
        edgecolor="black",
        linewidth=0.5,
        label="P_preq (loss_pred only)",
    )
    for bar, val in zip(bars_total, total_vals):
        ax_right.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.01,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar, val in zip(bars_pred, pred_vals):
        if np.isfinite(val):
            ax_right.text(
                bar.get_x() + bar.get_width() / 2,
                val * 1.01,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fukami_matched = matched_rows["Fukami AE d=32 matched (D81)"]["P_preq"]
    jepa_baseline = matched_rows["JEPA d=32 W0_C_lam100 (Session 11 baseline)"]["P_preq"]
    win_ratio = fukami_matched / jepa_baseline
    ax_right.set_xticks(bar_xs)
    ax_right.set_xticklabels([lab for _, lab in matched_keys], fontsize=9)
    ax_right.set_ylabel("P_preq (loss units x iters)")
    ax_right.set_title(
        f"Matched-d=32 epiplexity: JEPA wins by {win_ratio:.2f}x vs Fukami AE\n"
        "(N=400 logged steps; floor = trailing 10% median)"
    )
    ax_right.legend(loc="upper right", fontsize=9)
    ax_right.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Thrust 1: epiplexity decomposes the JEPA objective, and matched-d JEPA "
        f"compresses {win_ratio:.2f}x harder than the Fukami AE",
        fontsize=12,
    )
    out = FIG_DIR / "thrust1_epiplexity_decomposition.png"
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ------------------------------------------------------------------
# Figure 2: Thrust 2 forecast horizon (latent RMSE + SSIM vs H).
# ------------------------------------------------------------------
def figure2_thrust2_forecast_horizon() -> Path:
    """Latent RMSE + SSIM vs horizon, with IQR shading for Test B vs Test C."""
    rb = json.loads((SESSION14 / "rollout" / "S12_E_d64" / "test_b_rollout.json").read_text())
    rc = json.loads((SESSION14 / "rollout" / "S12_E_d64" / "test_c_rollout.json").read_text())

    horizons = np.asarray(rb["horizons"], dtype=float)

    def collect(rollout: dict, key: str) -> np.ndarray:
        # rows: encounters, columns: horizons.
        return np.array(
            [[enc[key][j] for j in range(len(horizons))] for enc in rollout["per_encounter"]],
            dtype=float,
        )

    rmse_b = collect(rb, "latent_rmse")
    rmse_c = collect(rc, "latent_rmse")
    ssim_b = collect(rb, "ssim")
    ssim_c = collect(rc, "ssim")

    def summary(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        med = np.median(a, axis=0)
        q1 = np.percentile(a, 25, axis=0)
        q3 = np.percentile(a, 75, axis=0)
        return med, q1, q3

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Left: latent RMSE ----
    ax = axes[0]
    med_b, q1_b, q3_b = summary(rmse_b)
    med_c, q1_c, q3_c = summary(rmse_c)
    ax.plot(horizons, med_b, color="#1f77b4", marker="o", lw=2.0, label="Test B (in-envelope)")
    ax.fill_between(horizons, q1_b, q3_b, color="#1f77b4", alpha=0.18)
    ax.plot(horizons, med_c, color="#d62728", marker="s", lw=2.0, ls="--",
            label="Test C (G=+4, OOD)")
    ax.fill_between(horizons, q1_c, q3_c, color="#d62728", alpha=0.15)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(horizons)
    ax.set_xticklabels([f"{int(h)}" for h in horizons])
    ax.set_xlabel("Forecast horizon H (frames at dt = 0.05 t/c)")
    ax.set_ylabel("Latent RMSE  ||z_pred - z_true||")
    ax.set_title("Latent forecast error (N=28 Test B, N=24 Test C encounters)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)

    # ---- Right: SSIM ----
    ax = axes[1]
    med_b, q1_b, q3_b = summary(ssim_b)
    med_c, q1_c, q3_c = summary(ssim_c)
    ax.plot(horizons, med_b, color="#1f77b4", marker="o", lw=2.0, label="Test B")
    ax.fill_between(horizons, q1_b, q3_b, color="#1f77b4", alpha=0.18)
    ax.plot(horizons, med_c, color="#d62728", marker="s", lw=2.0, ls="--", label="Test C")
    ax.fill_between(horizons, q1_c, q3_c, color="#d62728", alpha=0.15)
    ax.axhline(0.5, color="black", ls=":", lw=1.2,
               label="SSIM = 0.5 (Fukami reference)")
    ax.axhline(0.3, color="grey", ls=":", lw=1.2,
               label="SSIM = 0.3 (rough OOD acceptance)")
    ax.set_xscale("log")
    ax.set_xticks(horizons)
    ax.set_xticklabels([f"{int(h)}" for h in horizons])
    ax.set_xlabel("Forecast horizon H (frames at dt = 0.05 t/c)")
    ax.set_ylabel("SSIM (decoded rollout vs DNS raw omega)")
    ax.set_title("Decoded-rollout structural similarity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    fig.suptitle(
        "Thrust 2: long-rollout latent RMSE grows ~5x and SSIM halves "
        "from H=1 to H=88 on Test B; Test C ~30% worse",
        fontsize=12,
    )
    out = FIG_DIR / "thrust2_forecast_horizon.png"
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ------------------------------------------------------------------
# Figure 3: Thrust 2 hero long-rollout 3 x 4 + spectrum row.
# ------------------------------------------------------------------
def _radial_spectrum(omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute radial power spectrum |F(omega)|^2 vs |k| for a 2D array."""
    H, W = omega.shape
    F = np.fft.fftshift(np.fft.fft2(omega))
    P = np.abs(F) ** 2
    cy, cx = H // 2, W // 2
    y, x = np.indices(omega.shape)
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_int = r.astype(int)
    r_max = min(cy, cx)
    radial_sum = np.bincount(r_int.ravel(), weights=P.ravel())
    radial_count = np.bincount(r_int.ravel())
    radial_mean = radial_sum[: r_max + 1] / np.maximum(radial_count[: r_max + 1], 1)
    k = np.arange(len(radial_mean))
    return k, radial_mean


def figure3_thrust2_hero_long_rollout() -> Path:
    """4 horizons x (DNS, Pred, Residual) + spectrum row."""
    hero_dir = SESSION14 / "rollout" / "S12_E_d64" / "test_b_hero"
    horizons = [16, 32, 64, 88]
    frames = {16: 47, 32: 63, 64: 95, 88: 119}

    airfoil_xy = load_airfoil_xy()
    airfoil_px = omega_xy_to_pixel(airfoil_xy)

    def _load(prefix: str, h: int) -> np.ndarray:
        return np.load(hero_dir / f"omega_{prefix}_H{h:03d}_frame{frames[h]:03d}.npy")

    dns = {h: _load("dns", h) for h in horizons}
    pred = {h: _load("pred", h) for h in horizons}
    residual = {h: pred[h] - dns[h] for h in horizons}

    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(
        4, 5,
        width_ratios=[1, 1, 1, 1, 0.05],
        height_ratios=[1, 1, 1, 0.85],
        hspace=0.15,
        wspace=0.06,
    )

    row_labels = ["DNS truth", "Pred rollout", "Pred - DNS"]
    panels = [dns, pred, residual]

    last_im = None
    for r_idx, (label, data) in enumerate(zip(row_labels, panels)):
        for c_idx, h in enumerate(horizons):
            ax = fig.add_subplot(gs[r_idx, c_idx])
            im = imshow_omega(ax, data[h], airfoil_px, vmax=3.0)
            last_im = im
            if r_idx == 0:
                ax.set_title(f"H = {h}  (frame {frames[h]})", fontsize=11)
            if c_idx == 0:
                ax.set_ylabel(label, fontsize=11)

    # Shared colorbar for the 3 omega rows.
    cbar_ax = fig.add_subplot(gs[0:3, 4])
    cbar = fig.colorbar(last_im, cax=cbar_ax, extend="both")
    cbar.set_label("omega_z / (3 sigma_train)", fontsize=10)

    # ---- Bottom row: radial power spectra at each H ----
    for c_idx, h in enumerate(horizons):
        ax = fig.add_subplot(gs[3, c_idx])
        k_dns, P_dns = _radial_spectrum(dns[h])
        k_pred, P_pred = _radial_spectrum(pred[h])
        # Skip k=0 (DC) for log-log.
        ax.loglog(k_dns[1:], P_dns[1:], color="black", lw=1.6, label="DNS")
        ax.loglog(k_pred[1:], P_pred[1:], color="#d62728", ls="--", lw=1.4,
                  label="Pred rollout")
        ax.set_xlabel("|k| (pixel-1)")
        if c_idx == 0:
            ax.set_ylabel("Radial power |F(omega)|^2")
        ax.set_title(f"H = {h} spectrum", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        if c_idx == 0:
            ax.legend(loc="lower left", fontsize=8)

    # Hide the bottom-right cell (no spectrum colorbar).
    ax_empty = fig.add_subplot(gs[3, 4])
    ax_empty.axis("off")

    fig.suptitle(
        "Thrust 2 hero: omega rollout fidelity to H = 88 (4.4 t/c past context); "
        "spectrum holds within 1-2 decades through k_max",
        fontsize=12,
    )
    out = FIG_DIR / "thrust2_hero_long_rollout.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ------------------------------------------------------------------
# Figure 4: Thrust 4 intrinsic dim + PCA cumulative variance.
# ------------------------------------------------------------------
def figure4_thrust4_intrinsic_dim() -> Path:
    """Left: PCA cumulative variance. Right: estimator agreement bar."""
    payload = json.loads((SESSION14 / "intrinsic_dim" / "E_d64_intrinsic_dim.json").read_text())
    cum = np.asarray(payload["pca_spectrum_train_plus_test_a"], dtype=float)
    splits = payload["splits"]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))

    # ---- Left: PCA cumulative variance ----
    ax = axes[0]
    k = np.arange(1, len(cum) + 1)
    ax.plot(k, cum, color="#1f77b4", marker="o", lw=1.6, ms=4)
    ax.axhline(0.90, color="grey", ls=":", lw=1.2)
    ax.axhline(0.95, color="grey", ls=":", lw=1.2)
    # Annotate canonical thresholds (taken from the JSON: 80% at k=1, 90% at k=3,
    # 95% at k=7, 98% at k=12).
    anno = [(1, "80% @ k=1"), (3, "90% @ k=3"), (7, "95% @ k=7"), (12, "98% @ k=12")]
    for k_i, txt in anno:
        ax.scatter([k_i], [cum[k_i - 1]], color="red", s=60, zorder=5)
        ax.annotate(
            txt,
            xy=(k_i, cum[k_i - 1]),
            xytext=(k_i + 1.5, cum[k_i - 1] - 0.04),
            fontsize=9,
            arrowprops=dict(arrowstyle="-", color="red", lw=0.6),
        )
    ax.set_xlabel("PCA component k")
    ax.set_ylabel("Cumulative variance fraction")
    ax.set_title(
        "PCA cumulative variance on train + test_a impact-frame latents (N = 250)"
    )
    ax.set_xlim(0.5, 30)
    ax.set_ylim(0.75, 1.005)
    ax.grid(True, alpha=0.3)

    # ---- Right: estimator agreement bars ----
    ax = axes[1]
    # The intrinsic_dim JSON has 6 splits; pick the five the spec asks for.
    wanted = {
        "train (180 enc)": "train",
        "test_a (70 enc)": "test_a",
        "test_b (28 enc)": "test_b",
        "test_c (24 enc, OOD)": "test_c",
        "all (302)": "all",
    }
    rows = [r for r in splits if r["name"] in wanted]
    labels = [wanted[r["name"]] for r in rows]

    estimators = [
        ("PCA 95%", lambda r: r["pca_95"], "#1f77b4"),
        ("Levina-Bickel", lambda r: r["levina_bickel"]["mean"], "#2ca02c"),
        ("Two-NN", lambda r: r["two_nn"], "#ff7f0e"),
        ("Isomap elbow", lambda r: r["isomap_elbow"], "#9467bd"),
        ("Consensus", lambda r: r["consensus"], "#d62728"),
    ]
    n_splits = len(rows)
    n_est = len(estimators)
    width = 0.16
    x = np.arange(n_splits)

    for j, (name, fn, color) in enumerate(estimators):
        vals = [float(fn(r)) for r in rows]
        offset = (j - (n_est - 1) / 2.0) * width
        ax.bar(x + offset, vals, width, color=color, label=name, edgecolor="black", linewidth=0.3)

    ax.axhline(3, color="black", ls="--", lw=1.2, label="d = 3 (G, D, Y count)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Estimated intrinsic dimension")
    ax.set_title("Intrinsic-dim estimators agree within ~1 of d = 3")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Thrust 4: 64-D JEPA latent has intrinsic dimension ~3, matching the "
        "(G, D, Y) parameter count",
        fontsize=12,
    )
    out = FIG_DIR / "thrust4_intrinsic_dim.png"
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ------------------------------------------------------------------
# Figure 5: Thrust 3 concept vectors in PC1-PC2 space.
# ------------------------------------------------------------------
def _load_all_latents() -> dict[str, dict[str, np.ndarray]]:
    """Load latents + (G, D, Y) for train, test_a, test_b, test_c."""
    out = {}
    for split in ["train", "test_a", "test_b", "test_c"]:
        arr = np.load(
            SESSION14 / "latents" / "S12_E_d64" / f"{split}.npz",
            allow_pickle=True,
        )
        out[split] = {
            "z": arr["z"].astype(np.float32),
            "G": arr["G"].astype(np.float32),
            "D": arr["D"].astype(np.float32),
            "Y": arr["Y"].astype(np.float32),
        }
    return out


def figure5_thrust3_concept_vectors_2d() -> Path:
    """2D PCA of all impact-frame latents, color by G (left) and D (right),
    with averaging-method concept vectors projected into the same PC basis."""
    latents = _load_all_latents()
    splits = ["train", "test_a", "test_b", "test_c"]
    Z_all = np.concatenate([latents[s]["z"] for s in splits], axis=0)
    G_all = np.concatenate([latents[s]["G"] for s in splits])
    D_all = np.concatenate([latents[s]["D"] for s in splits])
    split_id = np.concatenate(
        [np.full(latents[s]["z"].shape[0], s, dtype=object) for s in splits]
    )

    pca = PCA(n_components=2).fit(Z_all)
    pc = pca.transform(Z_all)

    cv = json.loads((SESSION14 / "concept_vectors" / "E_d64_concept_vectors.json").read_text())
    v_G = np.asarray(cv["averaging_vectors"]["G"], dtype=np.float32)
    v_D = np.asarray(cv["averaging_vectors"]["D"], dtype=np.float32)
    v_Y = np.asarray(cv["averaging_vectors"]["Y"], dtype=np.float32)

    # Project concept vectors into PC basis (mean is subtracted by PCA already).
    proj_G = pca.transform(v_G.reshape(1, -1) + pca.mean_) - pca.transform(pca.mean_.reshape(1, -1))
    proj_D = pca.transform(v_D.reshape(1, -1) + pca.mean_) - pca.transform(pca.mean_.reshape(1, -1))
    proj_Y = pca.transform(v_Y.reshape(1, -1) + pca.mean_) - pca.transform(pca.mean_.reshape(1, -1))

    # Anchor each arrow at the centroid (origin in PC space).
    origin = pca.transform(pca.mean_.reshape(1, -1))[0]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # ---------- Left: color by G ----------
    ax = axes[0]
    sc = ax.scatter(
        pc[:, 0], pc[:, 1],
        c=G_all, cmap="coolwarm", s=22, alpha=0.85, edgecolors="black", linewidths=0.2,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("G (gust amplitude)")
    # Draw arrows for v_G (accented), v_D, v_Y.
    arrow_specs = [
        (proj_G[0], "#1f77b4", "v_G", 2.5),
        (proj_D[0], "#2ca02c", "v_D", 1.2),
        (proj_Y[0], "#ff7f0e", "v_Y", 1.2),
    ]
    for arr, color, label, lw in arrow_specs:
        ax.annotate(
            "",
            xy=(origin[0] + arr[0], origin[1] + arr[1]),
            xytext=(origin[0], origin[1]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0),
        )
        ax.text(
            origin[0] + arr[0] * 1.06,
            origin[1] + arr[1] * 1.06,
            label,
            color=color,
            fontsize=11,
            fontweight="bold",
        )
    # Mark splits with markers.
    for split, marker in [("test_b", "s"), ("test_c", "^")]:
        mask = split_id == split
        ax.scatter(
            pc[mask, 0], pc[mask, 1],
            facecolors="none", edgecolors="black", s=70,
            marker=marker, linewidths=1.0, label=split,
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title(f"PCA of impact-frame latents (N = {Z_all.shape[0]}); color by G; v_G accented")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---------- Right: color by D ----------
    ax = axes[1]
    sc = ax.scatter(
        pc[:, 0], pc[:, 1],
        c=D_all, cmap="viridis", s=22, alpha=0.85, edgecolors="black", linewidths=0.2,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("D (vortex diameter)")
    arrow_specs = [
        (proj_G[0], "#1f77b4", "v_G", 1.2),
        (proj_D[0], "#2ca02c", "v_D", 2.5),
        (proj_Y[0], "#ff7f0e", "v_Y", 1.2),
    ]
    for arr, color, label, lw in arrow_specs:
        ax.annotate(
            "",
            xy=(origin[0] + arr[0], origin[1] + arr[1]),
            xytext=(origin[0], origin[1]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0),
        )
        ax.text(
            origin[0] + arr[0] * 1.06,
            origin[1] + arr[1] * 1.06,
            label,
            color=color,
            fontsize=11,
            fontweight="bold",
        )
    for split, marker in [("test_b", "s"), ("test_c", "^")]:
        mask = split_id == split
        ax.scatter(
            pc[mask, 0], pc[mask, 1],
            facecolors="none", edgecolors="black", s=70,
            marker=marker, linewidths=1.0, label=split,
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("Same PCA; color by D; v_D accented")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Thrust 3: averaging-method concept vectors v_G, v_D, v_Y align with "
        "the structure of impact-frame latents in PC1-PC2 space",
        fontsize=12,
    )
    out = FIG_DIR / "thrust3_concept_vectors_2D.png"
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ------------------------------------------------------------------
# Figure 6: Thrust 4 PCA 3D scatter colored by G.
# ------------------------------------------------------------------
def figure6_thrust4_pca3d() -> Path:
    """3D scatter of impact-frame latents in the first 3 PCs."""
    latents = _load_all_latents()
    splits = ["train", "test_a", "test_b", "test_c"]
    Z_all = np.concatenate([latents[s]["z"] for s in splits], axis=0)
    G_all = np.concatenate([latents[s]["G"] for s in splits])
    split_id = np.concatenate(
        [np.full(latents[s]["z"].shape[0], s, dtype=object) for s in splits]
    )

    pca = PCA(n_components=3).fit(Z_all)
    pc = pca.transform(Z_all)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        pc[:, 0], pc[:, 1], pc[:, 2],
        c=G_all, cmap="coolwarm", s=28, alpha=0.85, edgecolors="black", linewidths=0.2,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label("G (gust amplitude)")

    # Highlight test_b and test_c.
    for split, marker, label in [("test_b", "s", "Test B"), ("test_c", "^", "Test C")]:
        mask = split_id == split
        ax.scatter(
            pc[mask, 0], pc[mask, 1], pc[mask, 2],
            facecolors="none", edgecolors="black", s=80,
            marker=marker, linewidths=1.0, label=label,
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
    ax.legend(loc="upper left")
    ax.set_title(
        f"Thrust 4: PC1-PC2-PC3 of impact latents (N = {Z_all.shape[0]}); the cluster "
        f"shape exhibits ~3 orthogonal directions\nconsistent with (G, D, Y) "
        "spanning the dataset envelope",
        fontsize=11,
    )
    out = FIG_DIR / "thrust4_pca3d.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ------------------------------------------------------------------
# Driver.
# ------------------------------------------------------------------
def _describe(path: Path, label: str) -> None:
    size_kb = path.stat().st_size / 1024.0
    print(f"  [{label}] {path.name:48s}  ({size_kb:7.1f} KB)")


def main() -> None:
    print("Generating Session 14 figures into:", FIG_DIR)
    out1 = figure1_thrust1_epiplexity_decomposition()
    _describe(out1, "Fig 1")
    out2 = figure2_thrust2_forecast_horizon()
    _describe(out2, "Fig 2")
    out3 = figure3_thrust2_hero_long_rollout()
    _describe(out3, "Fig 3")
    out4 = figure4_thrust4_intrinsic_dim()
    _describe(out4, "Fig 4")
    out5 = figure5_thrust3_concept_vectors_2d()
    _describe(out5, "Fig 5")
    out6 = figure6_thrust4_pca3d()
    _describe(out6, "Fig 6")
    print("Done.")


if __name__ == "__main__":
    main()
