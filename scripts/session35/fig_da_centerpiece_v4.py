"""F20 composite centerpiece: phase-resolved DA evaluation of the flagship stack.

Five panels, all numbers loaded from frozen JSON artifacts at build time:
  (a) C_L RMSE (physical units) per phase x estimator     <- outputs/session35/da_phase_eval_b177.json
  (b) C_L R2 per phase x estimator (variance artifact)    <- outputs/session35/da_phase_eval_b177.json
  (c) impact-phase estimator ladder, test_b + test_c      <- da_phase_eval records,
      outputs/session34/rex_filter_tuned.json + outputs/session35/rex_filter_tuned_s{1..4}.json,
      outputs/session35/two_stage_envelope.json, outputs/session35/two_stage_addendum.json
  (d) filter vs smoother C_L RMSE (impact / relax)        <- outputs/session34/da_smoother.json
  (e) test_a NIS band calibration (NIS-refutation panel)  <- outputs/session35/nis_band_tuning.json

Protocol constants shown in labels (K=8 sensors, delay 10 frames, N=64 members,
band scales 1.77 / 1.0 / 4.0, the excluded test-peeked 0.840) are READ from the
JSON protocol/summary blocks, not typed. The only literal constant is
DT_TC = 0.05 t/c per frame (cache attr dt_tc, CLAUDE.md "Per-file structure"),
used to convert the RTS lag (5 frames, from the da_smoother summary key
"rts_lag5") to the 0.25 t/c latency quoted in the deployment rule.

Usage:  .venv/bin/python scripts/session35/fig_da_centerpiece_v4.py
Output: paper/sections/figures/results/fig_da_centerpiece_v4.{pdf,png}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session21"))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from figstyle import TEXTWIDTH_IN, use_style  # noqa: E402

DT_TC = 0.05  # t/c per frame (cache attr dt_tc; CLAUDE.md "Per-file structure")

OUT_DIR = REPO / "paper" / "sections" / "figures" / "results"

# ---- recipe palette (DA-stack figure; distinct from the family key) ----------
C_OPEN = "#9e9e9e"     # open loop (no assimilation)
C_EOBS = "#e08214"     # static E_obs inversion
C_LIN = "#2166ac"      # linear LAE Kalman stack
C_REX = "#1b7837"      # REX-EnKF (nonlinear, this work)
C_2ST = "#762a83"      # two-stage filter
C_2ST_L = "#9970ab"    # two-stage at the NIS band (lighter)
C_EXCL = "#b2182b"     # excluded test-peeked band

RECIPE_LABEL = {
    "openloop": "open loop",
    "eobs": "static inverse",
    "linear_lae": "linear-latent filter",
    "rex_enkf": "forecast-noise filter",
}
RECIPE_COLOR = {"openloop": C_OPEN, "eobs": C_EOBS, "linear_lae": C_LIN, "rex_enkf": C_REX}
PHASES = ["pre", "impact", "relax"]
TITLE_FS = 7.5


def load(path: str) -> dict:
    with open(REPO / path) as f:
        return json.load(f)


def main() -> None:
    use_style()

    # ---- data ----------------------------------------------------------------
    phase = load("outputs/session35/da_phase_eval_b177.json")
    smoother = load("outputs/session34/da_smoother.json")
    envelope = load("outputs/session35/two_stage_envelope.json")
    addendum = load("outputs/session35/two_stage_addendum.json")
    nis = load("outputs/session35/nis_band_tuning.json")
    rex_seeds = [load("outputs/session34/rex_filter_tuned.json")] + [
        load(f"outputs/session35/rex_filter_tuned_s{s}.json") for s in (1, 2, 3, 4)
    ]

    proto = phase["protocol"]           # k=8, delay=10, members=64, band_scale=4.0
    k_sensors = proto["k"]
    delay = proto["delay"]
    members = proto["members"]
    band_prefreeze = proto["band_scale"]                     # 4.0 (test-peeked, D3)
    band_cov = rex_seeds[0]["protocol"]["band_scale"]        # 1.77 coverage band
    c_star = envelope["c_star"]                              # 1.0 NIS band
    peeked_val = smoother["summary"]["rex_enkf"]["impact"]["cl_r2"]  # 0.840, excluded

    # Session 39 figure pass (D-B): split into a main-text tracking figure
    # (a, b) and an appendix calibration figure (c ladder, d smoother, e NIS).
    fig_t = plt.figure(figsize=(TEXTWIDTH_IN, 2.5))
    gs_t = fig_t.add_gridspec(1, 2, wspace=0.40, left=0.11, right=0.90,
                              top=0.88, bottom=0.22)
    ax_a = fig_t.add_subplot(gs_t[0, 0])
    ax_b = fig_t.add_subplot(gs_t[0, 1])
    fig_c = plt.figure(figsize=(TEXTWIDTH_IN, 5.1))
    gs_c = fig_c.add_gridspec(2, 2, height_ratios=[1.30, 0.98], hspace=0.55,
                              wspace=0.40, left=0.165, right=0.895,
                              top=0.95, bottom=0.11)
    ax_c = fig_c.add_subplot(gs_c[0, 0])
    ax_d = fig_c.add_subplot(gs_c[0, 1])
    ax_e = fig_c.add_subplot(gs_c[1, :])

    n_b = len(phase["records"]["rex_enkf"])  # 42 test_b encounters

    # ---- (a) C_L RMSE per phase ----------------------------------------------
    recipes = ["openloop", "eobs", "linear_lae", "rex_enkf"]
    width = 0.19
    xs = np.arange(len(PHASES))
    for i, rec in enumerate(recipes):
        vals = [phase["summary"][rec][p]["cl_rmse"] for p in PHASES]
        ax_a.bar(xs + (i - 1.5) * width, vals, width, color=RECIPE_COLOR[rec],
                 label=RECIPE_LABEL[rec], zorder=3)
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(PHASES)
    ax_a.set_ylim(0, 1.78)
    ax_a.set_ylabel(r"$C_L$ RMSE")
    ax_a.set_title(f"(a) $C_L$ RMSE by phase\n(in-distribution test, $n={n_b}$)",
                   loc="left", fontsize=TITLE_FS)
    ax_a.legend(loc="upper right", fontsize=5.5, handlelength=1.0,
                borderaxespad=0.1, labelspacing=0.25)
    # relax is the BEST phase in physical units for every assimilating estimator
    relax_best = max(phase["summary"][r]["relax"]["cl_rmse"] for r in recipes[1:])
    ax_a.annotate("relax: lowest RMSE\nof all phases",
                  xy=(2 + 0.5 * width, relax_best + 0.02), xytext=(1.18, 0.60),
                  fontsize=6, ha="center",
                  arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"))

    # ---- (b) C_L R2 per phase (the variance artifact) -------------------------
    ymin_b = -3.4
    for i, rec in enumerate(recipes):
        vals = [phase["summary"][rec][p]["cl_r2"] for p in PHASES]
        ax_b.bar(xs + (i - 1.5) * width, vals, width, color=RECIPE_COLOR[rec], zorder=3)
        for j, v in enumerate(vals):
            if v < ymin_b:  # clipped bar (open-loop relax R2 = -50)
                ax_b.annotate(f"{v:.0f} (clipped)",
                              xy=(xs[j] + (i - 1.5) * width, ymin_b + 0.15),
                              fontsize=5.2, ha="center", va="bottom", rotation=90,
                              color="white", zorder=4)
    ax_b.axhline(0.0, color="0.25", lw=0.6, zorder=2)
    ax_b.set_ylim(ymin_b, 1.25)
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels(PHASES)
    ax_b.set_ylabel(r"$C_L$ $R^2$")
    ax_b.set_title(f"(b) $C_L$ $R^2$ by phase\n(in-distribution test, $n={n_b}$)",
                   loc="left", fontsize=TITLE_FS)
    rex_relax_r2 = phase["summary"]["rex_enkf"]["relax"]["cl_r2"]
    rex_relax_rmse = phase["summary"]["rex_enkf"]["relax"]["cl_rmse"]
    ax_b.annotate(
        f"relax: $R^2<0$ at\nthe best RMSE ({rex_relax_rmse:.2f});\n"
        "variance artefact,\nnot failure",
        xy=(2 + 1.5 * width, rex_relax_r2), xytext=(-0.32, -1.55), fontsize=6,
        va="top", bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6),
        arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"), zorder=5)

    # ---- (c) impact-phase estimator ladder ------------------------------------
    def med_cat(recs: list[dict]) -> tuple[float, int]:
        r2 = np.array([r["impact"]["r2"] for r in recs])
        return float(np.median(r2)), int((r2 < -1).sum())

    eobs_med, eobs_cat = med_cat(phase["records"]["eobs"])
    lin_med, lin_cat = med_cat(phase["records"]["linear_lae"])
    seed_vals = np.array([j["aggregates"]["median_CL_r2_impact"] for j in rex_seeds])
    seed_cats = [j["aggregates"]["catastrophic_impact_lt_-1"] for j in rex_seeds]
    rex_anchor, rex_cat = seed_vals[0], seed_cats[0]           # s0 = the 0.749 anchor
    band_lo = seed_vals.mean() - seed_vals.std(ddof=1)
    band_hi = seed_vals.mean() + seed_vals.std(ddof=1)

    ts10_b = envelope["arms"]["two_stage_test_b"]["aggregates"]
    ts10_c = envelope["arms"]["two_stage_test_c"]["aggregates"]
    ts177_b = addendum["arms"]["two_stage_test_b_b177"]["aggregates"]
    ts177_c = addendum["arms"]["two_stage_test_c_b177"]["aggregates"]
    rex177_c = addendum["arms"]["rex_test_c_b177"]["aggregates"]
    n_c = ts10_c["n_encounters"]  # 40 test_c encounters

    rows = [  # top -> bottom; (label, color, test_b val, test_b cat, test_c agg or None)
        (RECIPE_LABEL["eobs"], C_EOBS, eobs_med, eobs_cat, None),
        (RECIPE_LABEL["linear_lae"], C_LIN, lin_med, lin_cat, None),
        (f"forecast-noise filter ($c={band_cov}$)", C_REX, rex_anchor, rex_cat, rex177_c),
        (f"two-stage filter ($c^*={c_star:g}$, NIS)", C_2ST_L,
         ts10_b["median_CL_r2_impact"], ts10_b["catastrophic_impact_lt_-1"], ts10_c),
        (f"two-stage filter ($c={band_cov}$)", C_2ST,
         ts177_b["median_CL_r2_impact"], ts177_b["catastrophic_impact_lt_-1"], ts177_c),
    ]
    pitch = 1.7
    x0, x1 = 0.585, 0.93
    ys = np.arange(len(rows))[::-1] * pitch  # first row on top
    for y, (label, color, val, cat, agg_c) in zip(ys, rows):
        best = label.startswith(f"two-stage filter ($c={band_cov}")
        if best:  # best protocol-clean filter: shaded row + tagged label
            ax_c.add_patch(Rectangle((x0, y - 1.20), x1 - x0, 1.70, facecolor=color,
                                     alpha=0.10, edgecolor="none", zorder=1))
            label = label + "\nbest protocol-clean filter"
        if label.startswith("forecast-noise filter"):  # 5-seed member-noise band (mean +- sd)
            ax_c.fill_betweenx([y - 0.34, y + 0.34], band_lo, band_hi,
                               color=color, alpha=0.22, lw=0, zorder=2)
            ax_c.annotate(
                f"5 seeds: {seed_vals.mean():.3f}$\\,\\pm\\,${seed_vals.std(ddof=1):.3f}",
                xy=(0.5 * (band_lo + band_hi), y + 0.42), fontsize=5.5, ha="center",
                va="bottom", color=color)
        ax_c.plot([val], [y], marker="o", ms=5.5 if best else 4.5, color=color,
                  zorder=4, mec="black" if best else color, mew=0.6)
        ax_c.annotate(f"{val:.2f} ({cat}/{n_b})", xy=(val, y - 0.40), fontsize=5.5,
                      ha="center", va="top", color="0.15")
        if agg_c is not None:
            vc = agg_c["median_CL_r2_impact"]
            cc = agg_c["catastrophic_impact_lt_-1"]
            ax_c.plot([vc], [y], marker="o", ms=4.5, mfc="none", mec=color, mew=0.9,
                      zorder=4)
            ax_c.annotate(f"C: {vc:.2f} ({cc}/{n_c})", xy=(vc, y - 0.82),
                          fontsize=5.5, ha="center", va="top", color=color)
        parts = label.split("\n")
        ax_c.text(-0.02, y + (0.26 if len(parts) > 1 else 0.0), parts[0],
                  transform=ax_c.get_yaxis_transform(), ha="right", va="center",
                  fontsize=6.5, fontweight="bold" if best else "normal")
        if len(parts) > 1:
            ax_c.text(-0.02, y - 0.34, parts[1],
                      transform=ax_c.get_yaxis_transform(), ha="right", va="center",
                      fontsize=5.5, fontstyle="italic", color=color)
    ax_c.set_yticks([])
    ax_c.set_ylim(-1.45, ys[0] + 1.95)
    ax_c.set_xlim(x0, x1)
    ax_c.set_xlabel(r"impact-phase median $C_L$ $R^2$")
    ax_c.set_title("(a) impact-phase estimator ladder",
                   loc="left", fontsize=TITLE_FS)
    hb = plt.Line2D([], [], marker="o", ls="none", color="0.2", ms=4,
                    label=f"in-distribution test ($n={n_b}$)")
    hc = plt.Line2D([], [], marker="o", ls="none", mfc="none", mec="0.2", ms=4,
                    label=f"boundary test, $|G|=4$ ($n={n_c}$)")
    ax_c.legend(handles=[hb, hc], loc="upper center", ncol=2, fontsize=5.5,
                handletextpad=0.2, borderaxespad=0.0, columnspacing=0.8)

    # ---- (d) filter vs smoother -----------------------------------------------
    ss = smoother["summary"]
    rts_lag_frames = 5  # from the da_smoother summary key "rts_lag5" used below
    assert f"rts_lag{rts_lag_frames}" in ss
    lag_tc = rts_lag_frames * DT_TC  # 0.25 t/c
    methods = [  # (summary key, label, color, is_smoother)
        ("kf", "linear\nKF", C_LIN, False),
        (f"rts_lag{rts_lag_frames}", f"+RTS\nlag {rts_lag_frames}", C_LIN, True),
        ("rex_enkf", "forecast-\nnoise", C_REX, False),
        ("rex_enks", "+EnKS", C_REX, True),
    ]
    xd = np.arange(len(methods))
    wd = 0.34
    for i, (key, label, color, smooth) in enumerate(methods):
        hatch = "////" if smooth else None
        for dx, alpha, v in ((-wd / 2, 1.0, ss[key]["impact"]["cl_rmse"]),
                             (wd / 2, 0.45, ss[key]["relax"]["cl_rmse"])):
            ax_d.bar(i + dx, v, wd, color=color, alpha=alpha, hatch=hatch,
                     lw=0.4, edgecolor="white", zorder=3)
            ax_d.annotate(f"{v:.2f}", xy=(i + dx, v + 0.005), fontsize=5.2,
                          ha="center", va="bottom")
    ax_d.set_xticks(xd)
    ax_d.set_xticklabels([m[1] for m in methods], fontsize=6)
    ax_d.set_ylabel(r"$C_L$ RMSE")
    ax_d.set_ylim(0, 0.62)
    ax_d.set_title(f"(b) filter vs smoother (in-distribution test, $n={n_b}$)",
                   loc="left", fontsize=TITLE_FS)
    handles_d = [
        Patch(facecolor="0.35", label="impact"),
        Patch(facecolor="0.35", alpha=0.45, label="relax"),
        Patch(facecolor="white", edgecolor="0.45", hatch="////", label="smoother"),
    ]
    ax_d.legend(handles=handles_d, loc="upper center", ncol=3, fontsize=5.2,
                handletextpad=0.3, handlelength=0.9, borderaxespad=0.0,
                columnspacing=0.7, labelspacing=0.25)
    ax_d.text(0.02, 0.855,
              f"online: nonlinear filter; {lag_tc:g} t/c\n"
              f"latency (RTS lag {rts_lag_frames}): linear smoother",
              transform=ax_d.transAxes, fontsize=5.8, va="top", fontstyle="italic")

    # ---- (e) NIS band calibration on test_a ------------------------------------
    bands = sorted(nis["bands"].keys(), key=float)
    bx = np.array([float(b) for b in bands])
    nis_vals = np.array([nis["bands"][b]["pooled_mean_nis_impact"] for b in bands])
    r2_vals = np.array([nis["bands"][b]["median_CL_r2_impact"] for b in bands])
    n_a = nis["bands"][bands[0]]["n_encounters"]  # 100 test_a encounters

    ax_e2 = ax_e.twinx()
    ax_e2.spines["right"].set_visible(True)
    l1, = ax_e.plot(bx, nis_vals, marker="o", color="0.25", lw=1.0,
                    label="pooled NIS")
    ax_e.axhline(1.0, color="0.25", lw=0.7, ls=(0, (4, 3)))
    ax_e.annotate(r"target $\mathrm{E}[\mathrm{NIS}]=1$", xy=(0.87, 1.03),
                  fontsize=5.8, ha="left", color="0.25")
    l2, = ax_e2.plot(bx, r2_vals, marker="s", color=C_REX, lw=1.0,
                     label=r"median $C_L$ $R^2$")
    # pre-registered selection: argmin |pooled NIS - 1| saturates at the grid edge
    isel = int(np.argmin(np.abs(nis_vals - 1.0)))
    assert bx[isel] == c_star  # matches the frozen c_star in nis_band_tuning.json
    ax_e.plot([bx[isel]], [nis_vals[isel]], marker="o", ms=9, mfc="none", mec="0.1",
              mew=0.9, zorder=5)
    ax_e.annotate(
        f"NIS $<1$ at every band: the selection\n"
        f"saturates at the grid edge ($c^*={c_star:g}$)\n"
        f"and picks the lowest-$R^2$ band",
        xy=(bx[isel] + 0.07, nis_vals[isel] - 0.03), xytext=(2.0, 0.63),
        fontsize=6, va="top", arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"))
    # excluded test-peeked band: annotate, do NOT plot its 0.840 value.
    # Text sits in the empty band between the NIS curve (below 0.42 for
    # c > 2.5) and the R^2 curve (above 0.93 there), clear of both.
    ax_e.axvline(band_prefreeze, color=C_EXCL, lw=0.8, ls=(0, (2, 2)))
    ax_e.annotate(f"$c={band_prefreeze:g}$: test-peeked {peeked_val:.3f} excluded",
                  xy=(2.60, 0.87), fontsize=6,
                  color=C_EXCL, va="top", ha="left")
    ax_e.set_xlabel("innovation-band scale $c$")
    ax_e.set_ylabel("pooled impact NIS", color="0.25")
    ax_e2.set_ylabel(r"median impact $C_L$ $R^2$", color=C_REX)
    ax_e.tick_params(axis="y", colors="0.25")
    ax_e2.tick_params(axis="y", colors=C_REX)
    ax_e.set_ylim(0, 1.14)
    ax_e2.set_ylim(0.54, 0.88)
    ax_e.set_xlim(0.8, 6.2)
    ax_e.set_title(f"(c) NIS band calibration (validation, $n={n_a}$)",
                   loc="left", fontsize=TITLE_FS)
    ax_e.legend(handles=[l1, l2], loc="center right", fontsize=6, handlelength=1.3,
                borderaxespad=0.4, labelspacing=0.25)

    # ---- footers: shared protocol, split with the panels ----------------------
    fig_t.text(0.11, 0.03,
               f"$K={k_sensors}$ wall-pressure taps, delay-{delay}, $N={members}$ "
               f"members; phase evaluation at the val-calibrated band "
               f"$c={band_prefreeze:g}$.", fontsize=5.5, color="0.3")
    fig_c.text(0.165, 0.02,
               f"(a) frozen bands only; (c) selection on the validation split only; "
               f"$K={k_sensors}$ taps, $N={members}$ members.",
               fontsize=5.5, color="0.3")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_t.savefig(OUT_DIR / "fig_da_tracking_v4.pdf")
    fig_c.savefig(OUT_DIR / "fig_da_calibration_v4.pdf")
    print("wrote fig_da_tracking_v4.pdf + fig_da_calibration_v4.pdf")


if __name__ == "__main__":
    main()
