"""F19 add-on: relative peak error across the gust envelope (scale invariance).

Panel (a): relative peak C_L error (percent of the true peak, per-|G| medians)
vs |G| for the three assimilating recipes in
``outputs/session34/da_relative_errors.json`` (rex_enkf, linear_lae, eobs).
The REX-EnKF band is roughly scale-invariant across the envelope while the
linear filter is non-uniform. An inset shows the same statistic aggregated by
gust diameter D (``peak_rel_err_pct_by_D``). The open-loop range is annotated
as text (it is off-scale).

Panel (b): absolute peak C_L error vs |G|, per-|G| medians of
|peak_value_error| over the per-encounter records in
``outputs/session34/da_phase_eval.json`` (the source records from which the
relative-error JSON medians are derived; convention verified median, not
mean), with the median true peak |C_L| as the grey growth reference. The
absolute error grows with the peak itself, which is what makes the flat
relative band in (a) meaningful.

Split: test_b (all 42 record encounters carry test_b case ids; |G| <= 3).

Run (CPU):
    OMP_NUM_THREADS=4 .venv/bin/python scripts/session35/fig_relative_errors_v4.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "session21"))

import figstyle as fs  # noqa: E402

REL_JSON = REPO_ROOT / "outputs" / "session34" / "da_relative_errors.json"
PHASE_JSON = REPO_ROOT / "outputs" / "session34" / "da_phase_eval.json"
OUT_DIR = REPO_ROOT / "paper" / "sections" / "figures" / "results"
OUT_STEM = "fig_relative_errors_v4"

# filter recipe -> display style. rex_enkf is the nonlinear headline (green),
# linear_lae the linear stack (blue), eobs the filter-free pressure readout.
FILTER_STYLE = {
    "rex_enkf": {
        "label": "forecast-noise filter",
        "color": fs.FAMILY_COLOR["jepa"],
        "marker": "o",
        "ls": "-",
    },
    "linear_lae": {
        "label": "linear-latent filter",
        "color": fs.FAMILY_COLOR["pod"],
        "marker": "^",
        "ls": "--",
    },
    "eobs": {
        "label": "static inverse",
        "color": fs.SPLIT_COLOR["test_b_boundary"],
        "marker": "s",
        "ls": "-.",
    },
}

CASE_RE = re.compile(r"G([+-][0-9.]+)_D([0-9.]+)_Y")


def case_gd(case_id: str) -> tuple[float, float]:
    m = CASE_RE.match(case_id)
    if m is None:
        raise ValueError(f"unparseable case_id: {case_id}")
    return abs(float(m.group(1))), float(m.group(2))


def main() -> int:
    fs.use_style()

    rel = json.loads(REL_JSON.read_text())
    phase = json.loads(PHASE_JSON.read_text())

    # ---- panel (b) source: per-encounter records -> per-|G| medians --------
    abs_err_by_g: dict[str, dict[float, float]] = {}
    true_peak_by_g: dict[float, float] = {}
    split_cases = sorted({r["case_id"] for r in phase["records"]["rex_enkf"]})
    for fam in FILTER_STYLE:
        by_g: dict[float, list[float]] = {}
        for r in phase["records"][fam]:
            g, _d = case_gd(r["case_id"])
            by_g.setdefault(g, []).append(abs(r["peak"]["peak_value_error"]))
        abs_err_by_g[fam] = {g: float(np.median(v)) for g, v in sorted(by_g.items())}
    peaks_by_g: dict[float, list[float]] = {}
    for r in phase["records"]["rex_enkf"]:  # true peaks identical across recipes
        g, _d = case_gd(r["case_id"])
        peaks_by_g.setdefault(g, []).append(abs(r["peak"]["true_peak_cl"]))
    true_peak_by_g = {g: float(np.median(v)) for g, v in sorted(peaks_by_g.items())}

    # open-loop annotation range (off-scale in panel (a)), read from the JSON
    ol = [v for v in rel["openloop"]["peak_rel_err_pct_by_G"].values()]
    ol_lo, ol_hi = min(ol), max(ol)

    print(f"[F19+] split: test_b ({len(split_cases)} cases: {split_cases})")
    for fam in FILTER_STYLE:
        vals = rel[fam]["peak_rel_err_pct_by_G"]
        print(
            f"[F19+] {fam:10s} rel peak err by |G|: "
            + ", ".join(f"{k}: {v:.1f}%" for k, v in vals.items())
        )
    print(f"[F19+] openloop rel peak err range: {ol_lo:.0f}-{ol_hi:.0f}%")

    # ---- figure -------------------------------------------------------------
    fig_w = fs.TEXTWIDTH_IN
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(fig_w, fig_w * 0.42), gridspec_kw={"wspace": 0.32}
    )

    # (a) relative peak error vs |G|
    for fam, sty in FILTER_STYLE.items():
        d = rel[fam]["peak_rel_err_pct_by_G"]
        g = np.array([float(k) for k in d])
        y = np.array([d[k] for k in d])
        order = np.argsort(g)
        ax_a.plot(
            g[order],
            y[order],
            color=sty["color"],
            marker=sty["marker"],
            ls=sty["ls"],
            lw=1.0,
            markersize=4.0,
            label=sty["label"],
        )
    ax_a.set_xlabel(r"$|G|$")
    ax_a.set_ylabel(r"peak $C_L$ error (% of true peak)")
    ax_a.set_ylim(0, 30)
    ax_a.set_xticks(sorted(float(k) for k in rel["rex_enkf"]["peak_rel_err_pct_by_G"]))
    ax_a.legend(
        loc="upper left", handlelength=2.2, fontsize=6.2, borderaxespad=0.2, labelspacing=0.35
    )
    ax_a.text(
        0.98,
        0.10,
        "test B, per-|G| medians",
        transform=ax_a.transAxes,
        fontsize=6.5,
        color="0.35",
        ha="right",
    )
    ax_a.text(
        0.98,
        0.03,
        f"open loop off-scale: {ol_lo:.0f}-{ol_hi:.0f}%",
        transform=ax_a.transAxes,
        fontsize=6.5,
        color="0.35",
        ha="right",
    )
    ax_a.text(-0.20, 1.02, "(a)", transform=ax_a.transAxes, fontsize=8.5, fontweight="bold")

    # (a) inset: same statistic aggregated by gust diameter D
    ax_in = ax_a.inset_axes([0.63, 0.50, 0.34, 0.36])
    for fam, sty in FILTER_STYLE.items():
        d = rel[fam]["peak_rel_err_pct_by_D"]
        dd = np.array([float(k) for k in d])
        y = np.array([d[k] for k in d])
        order = np.argsort(dd)
        ax_in.plot(
            dd[order],
            y[order],
            color=sty["color"],
            marker=sty["marker"],
            ls=sty["ls"],
            lw=0.8,
            markersize=2.8,
        )
    ax_in.set_xticks([0.5, 1.0, 1.5])
    ax_in.set_ylim(0, 25)
    ax_in.tick_params(labelsize=5.5, length=2, pad=1.5)
    ax_in.set_xlabel(r"$D$", fontsize=6, labelpad=1)
    ax_in.set_ylabel("%", fontsize=6, labelpad=1)

    # (b) absolute peak error vs |G| + true-peak growth reference
    g_sorted = sorted(true_peak_by_g)
    ax_b.plot(
        g_sorted,
        [true_peak_by_g[g] for g in g_sorted],
        color=fs.FAMILY_COLOR["oracle"],
        ls=(0, (4, 3)),
        lw=1.0,
        marker="D",
        markersize=3.2,
        label=r"true peak $|C_L|$ (median)",
    )
    for fam, sty in FILTER_STYLE.items():
        d = abs_err_by_g[fam]
        ax_b.plot(
            list(d.keys()),
            list(d.values()),
            color=sty["color"],
            marker=sty["marker"],
            ls=sty["ls"],
            lw=1.0,
            markersize=4.0,
            label=sty["label"],
        )
    ax_b.set_xlabel(r"$|G|$")
    ax_b.set_ylabel(r"median $|$peak $C_L$ error$|$")
    ax_b.set_xticks(g_sorted)
    ax_b.set_ylim(bottom=0)
    ax_b.legend(
        loc="upper left", handlelength=2.2, fontsize=6.2, borderaxespad=0.2, labelspacing=0.35
    )
    ax_b.text(-0.20, 1.02, "(b)", transform=ax_b.transAxes, fontsize=8.5, fontweight="bold")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{OUT_STEM}.pdf"
    png = OUT_DIR / f"{OUT_STEM}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    print(f"[F19+] wrote {pdf}")
    print(f"[F19+] wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
