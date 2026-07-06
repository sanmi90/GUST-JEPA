"""F13 (v4 paper): the latent-REX forecaster card.

Two-part figure:
  (a) architecture schematic: context window -> per-window instance norm ->
      arcsinh -> 2-layer LSTM -> direct multi-horizon quantile head (one shot,
      no autoregressive feedback) -> inverse transform; pinball loss formula.
  (b) the two calibration criteria side by side: the band scale c* = 1.7674
      selected as the 80 percent one-step coverage quantile on VALIDATION
      (rex_tune.json winner, "coverage-calibrated (val)") versus the Session 35
      pooled impact-phase NIS per band scale on TEST A
      (nis_band_tuning.json, "NIS on test_a") with the NIS = 1 reference and
      the pre-registered argmin |NIS - 1| selection.

Every RESULT number (winner kind / hidden / nq / band_c_star / params, the
per-band NIS values, the NIS-selected c*) is read from the JSONs at build
time. Architecture constants are literals with file:line provenance
(extracted in outputs/session35/mc_provenance.md MC-4):
  - 2-layer LSTM                    | scripts/session34/latent_rex.py:50
  - instance norm (mu, sd>=1e-3)    | latent_rex.py:59
  - arcsinh((z - mu)/sd)            | latent_rex.py:60
  - inverse sinh(.)*sd + mu         | latent_rex.py:63
  - head Linear-GELU-Linear, H*d*nq | latent_rex.py:51-54
  - horizon H = 40                  | latent_rex.py:82
  - latent d = 32                   | latent_rex.py:46 (pooled flagship latent)
  - train context L ~ U{16..30}     | scripts/session34/rex_tune.py:154
  - eval context L = 25             | latent_rex.py:133
  - quantiles q = 0.1..0.9          | rex_tune.py:47
  - pinball max(qu, (q-1)u)         | latent_rex.py:66-72
  - no autoregressive feedback      | latent_rex.py:11-13 (docstring, design)
  - c* = 80% coverage quantile      | rex_tune.py:222-232

Outputs: paper/sections/figures/results/fig_rex_card_v4.{pdf,png}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import figstyle  # noqa: E402

TUNE_JSON = REPO / "outputs/session34/rex_tune.json"
NIS_JSON = REPO / "outputs/session35/nis_band_tuning.json"
OUT_DIR = REPO / "paper/sections/figures/results"

GREEN = figstyle.FAMILY_COLOR["jepa"]
ORANGE = "#e08214"  # boundary accent, matches the v3 mechanism figure
GREY = figstyle.FAMILY_COLOR["oracle"]
RED = "#b2182b"

# Architecture literals with provenance (see module docstring / MC-4).
H_HORIZON = 40        # latent_rex.py:82
D_LATENT = 32         # latent_rex.py:46 (d=32 pooled flagship latent)
CTX_TRAIN = "16..30"  # rex_tune.py:154, rng.integers(16, 31)
CTX_EVAL = 25         # latent_rex.py:133


def _box(ax, x, y, w, h, text, fc="#f2f2f2", fs=5.8):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor=fc, edgecolor="0.25", linewidth=0.8, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=4, linespacing=1.45)


def _arrow(ax, p0, p1, rad=0.0, **kw):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=7,
        color=kw.pop("color", "0.2"), lw=kw.pop("lw", 0.9),
        linestyle=kw.pop("ls", "-"),
        connectionstyle=f"arc3,rad={rad}", zorder=2, **kw))


def panel_a(ax, winner: dict) -> None:
    """Architecture schematic. Run-selected constants come from the JSON."""
    kind = winner["kind"].upper()   # rex_tune.json winner
    hidden = int(winner["hidden"])  # rex_tune.json winner
    nq = int(winner["nq"])          # rex_tune.json winner
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y1, y2, hb = 0.68, 0.32, 0.22
    # row 1: context -> instance norm + arcsinh -> LSTM
    _box(ax, 0.00, y1, 0.31, hb,
         "context window $z_{t-L..t}$\n"
         rf"$L \sim U\{{{CTX_TRAIN}\}}$ train,  $L = {CTX_EVAL}$ eval")
    _box(ax, 0.37, y1, 0.30, hb,
         r"instance norm $(\mu, \sigma)$"
         "\n" r"$\mathrm{arcsinh}\left((z - \mu)/\sigma\right)$")
    _box(ax, 0.73, y1, 0.27, hb,
         rf"{kind} $\times$ 2 layers" + f"\n$h = {hidden}$", fc="#e7f2e9")
    _arrow(ax, (0.315, y1 + hb / 2), (0.363, y1 + hb / 2))
    _arrow(ax, (0.675, y1 + hb / 2), (0.723, y1 + hb / 2))
    # wrap-around arrow into row 2
    _arrow(ax, (0.865, y1 - 0.015), (0.115, y2 + hb / 2), rad=-0.22)
    # row 2: direct head -> inverse transform / quantile fan
    _box(ax, 0.12, y2, 0.42, hb,
         "direct multi-horizon head, one shot\n"
         rf"$H \times d \times n_q = "
         rf"{H_HORIZON} \times {D_LATENT} \times {nq}$", fc="#e7f2e9")
    _box(ax, 0.62, y2, 0.38, hb,
         r"inverse $\sinh(\cdot)\,\sigma + \mu$"
         "\n" rf"quantile fan $q = 0.1 .. 0.9$")
    _arrow(ax, (0.545, y2 + hb / 2), (0.613, y2 + hb / 2))

    # pinball loss (latent_rex.py:66-72) + params (rex_tune.json winner)
    ax.text(0.12, 0.235,
            r"pinball loss $\rho_q(u) = \max\!\left(qu, (q-1)u\right)$,"
            r" $u = z - \hat{z}_q$", fontsize=6.0, ha="left", va="top")
    ax.text(1.0, 0.235, rf"{winner['params_m']:.1f}M params",
            fontsize=6.0, ha="right", va="top", color="0.35")

    # crossed-out autoregressive feedback path (latent_rex.py:11-13)
    _arrow(ax, (0.81, 0.115), (0.06, 0.115), color=RED, ls=(0, (4, 3)))
    ax.text(0.44, 0.115, r"$\times$", color=RED, fontsize=12,
            ha="center", va="center", zorder=3, fontweight="bold")
    ax.text(0.44, 0.045, "no autoregressive feedback: rollout error "
            "cannot compound", color=RED, fontsize=5.8, ha="center", va="top")
    ax.set_title("(a) latent-REX: direct multi-horizon quantile forecaster",
                 fontsize=8, loc="left")


def panel_b(ax, winner: dict, nis: dict) -> None:
    """The two calibration criteria for the predictive-band scale c."""
    c_star_cov = winner["band_c_star"]  # 80% coverage quantile on val
    keys = sorted(nis["bands"], key=float)
    bands = [float(k) for k in keys]
    nis_vals = [nis["bands"][k]["pooled_mean_nis_impact"] for k in keys]
    c_star_nis = float(nis["c_star"])   # argmin |pooled impact NIS - 1|
    c_star_nis_val = float(nis["c_star_nis"])

    # NIS = 1 filter-consistency reference
    ax.axhline(1.0, color=GREY, ls=(0, (4, 3)), lw=0.9, zorder=1)
    ax.text(6.15, 1.025, "NIS = 1 (consistent)", ha="right", va="bottom",
            fontsize=6.0, color=GREY)

    # criterion 2: pooled impact NIS on test A per band scale
    ax.plot(bands, nis_vals, color=GREEN, marker="o", ms=3.5, lw=1.0, zorder=3)
    ax.scatter([c_star_nis], [c_star_nis_val], s=52, facecolors="none",
               edgecolors=GREEN, linewidths=1.1, zorder=4)
    ax.annotate("NIS-selected (test A):\n" rf"$c^{{*}} = {c_star_nis:g}$",
                (c_star_nis, c_star_nis_val), xytext=(0.76, 0.72),
                fontsize=6.0, color=GREEN, ha="left", linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.6,
                                shrinkB=4))

    # criterion 1: coverage-calibrated on validation (rex_tune winner)
    ax.axvline(c_star_cov, color=ORANGE, ls="-", lw=1.1, zorder=2)
    ax.scatter([c_star_cov], [0.0], s=22, color=ORANGE, marker="D",
               zorder=4, clip_on=False)
    ax.text(c_star_cov + 0.14, 0.80,
            "coverage-calibrated (val):\n"
            rf"$c^{{*}} = {c_star_cov:.4f}$"
            "\n(80% one-step coverage)",
            fontsize=6.0, color=ORANGE, ha="left", va="center",
            linespacing=1.4)

    ax.set_xlabel(r"predictive-band scale $c$")
    ax.set_ylabel("pooled impact NIS (test A)")
    ax.set_xlim(0.65, 6.3)
    ax.set_ylim(0.0, 1.14)
    ax.set_title("(b) two calibration criteria", fontsize=8, loc="left")


def main() -> int:
    figstyle.use_style()
    winner = json.loads(TUNE_JSON.read_text())["winner"]
    nis = json.loads(NIS_JSON.read_text())

    fig = plt.figure(figsize=(figstyle.TEXTWIDTH_IN, 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.25,
                          left=0.005, right=0.985, top=0.89, bottom=0.155)
    panel_a(fig.add_subplot(gs[0]), winner)
    panel_b(fig.add_subplot(gs[1]), winner, nis)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_rex_card_v4.pdf")
    fig.savefig(OUT_DIR / "fig_rex_card_v4.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT_DIR / "fig_rex_card_v4.pdf")
    print("wrote", OUT_DIR / "fig_rex_card_v4.png")
    print(f"  winner: {winner['kind']} h{winner['hidden']} q{winner['nq']} "
          f"c*_cov={winner['band_c_star']:.4f} params={winner['params_m']:.2f}M")
    print(f"  NIS-selected c*={nis['c_star']:g} (pooled impact NIS "
          f"{nis['c_star_nis']:.3f}); bands="
          + ", ".join(f"{k}:{nis['bands'][k]['pooled_mean_nis_impact']:.3f}"
                      for k in nis["bands"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
