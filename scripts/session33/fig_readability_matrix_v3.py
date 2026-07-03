"""F4 (v3): what the state carries and who supplies it (Table X visual).

Two horizontal-bar panels over the ten-model controlled matrix, with the
models grouped by wake-supervision status (wake head on vs off): (a) windowed
test_b wake-enstrophy linear readability R2 (the attribution result), and
(b) matched-predictor merit at h = 8. A narrow text column on the right lists
the decode-floor SSIM per model. Bars are coloured by objective family.

Every plotted value is read from
outputs/session33/numbers_parts/table_x.json; nothing is hand-typed.

Usage:
    taskset -c 16-23 python scripts/session33/fig_readability_matrix_v3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session21"))

from figstyle import FAMILY_COLOR, TEXTWIDTH_IN, use_style  # noqa: E402

TABLE_X = REPO / "outputs/session33/numbers_parts/table_x.json"
OUT_PDF = REPO / "outputs/session33/figures/fig_readability_matrix_v3.pdf"


def lighten(color: str, amount: float) -> tuple[float, float, float]:
    """Blend a colour toward white by ``amount`` in [0, 1]."""
    r, g, b = matplotlib.colors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


OBJECTIVE_COLOR = {
    "predictive": FAMILY_COLOR["jepa"],
    "reconstructive": FAMILY_COLOR["fukami"],
    "bvae": lighten(FAMILY_COLOR["fukami"], 0.45),
    "none": "#e08214",
    "pod": FAMILY_COLOR["pod"],
}
OBJECTIVE_LABEL = {
    "predictive": "predictive (JEPA)",
    "reconstructive": "reconstructive",
    "bvae": r"recon. + KL ($\beta$-VAE)",
    "none": "supervised only",
    "pod": "linear (POD)",
}

# (json suffix, display label, objective family, wake head on)
MODELS = [
    ("JepaWake", "JEPA", "predictive", True),
    ("SupOnly", "supervised only", "none", True),
    ("AeWake", "AE", "reconstructive", True),
    ("Bvae", r"$\beta$-VAE", "bvae", True),
    ("FukamiWake", "Fukami AE", "reconstructive", True),
    ("JepaNowake", "JEPA", "predictive", False),
    ("AeNowake", "AE", "reconstructive", False),
    ("RegAE", "reg. AE", "reconstructive", False),
    ("Fukami", "Fukami AE", "reconstructive", False),
    ("Pod", "POD", "pod", False),
]


def main() -> None:
    use_style()
    numbers = json.loads(TABLE_X.read_text())["numbers"]

    def val(prefix: str, model: str) -> float:
        return float(numbers[f"{prefix}_{model}"]["value"])

    wake = [val("x_wake", m) for m, _, _, _ in MODELS]
    merit = [val("x_merit", m) for m, _, _, _ in MODELS]
    ssim = [val("x_ssim", m) for m, _, _, _ in MODELS]
    colors = [OBJECTIVE_COLOR[obj] for _, _, obj, _ in MODELS]
    labels = [lab for _, lab, _, _ in MODELS]

    horizon = numbers["x_merit_JepaWake"]["horizon"]
    split = numbers["x_wake_JepaWake"]["split"].replace("test_b", "test B")

    # y layout: ON block on top, thin separator, OFF block below, plus two
    # header rows for the block labels.
    n_on = sum(1 for m in MODELS if m[3])
    y_on = [10.0 - i for i in range(n_on)]
    y_off = [4.0 - i for i in range(len(MODELS) - n_on)]
    ys = y_on + y_off
    y_sep = 5.45
    y_hdr_on, y_hdr_off = 11.05, 4.9

    fig, (ax_a, ax_b, ax_s) = plt.subplots(
        1, 3, figsize=(TEXTWIDTH_IN, 3.25), sharey=True,
        gridspec_kw={"width_ratios": [1.3, 1.0, 0.30], "wspace": 0.12},
    )
    fig.subplots_adjust(left=0.17, right=0.99, top=0.94, bottom=0.295)

    for ax, vals in ((ax_a, wake), (ax_b, merit)):
        ax.barh(ys, vals, height=0.62, color=colors, edgecolor="none", zorder=3)
        ax.axvline(0.0, color="0.2", lw=0.6, zorder=2)
        ax.axhline(y_sep, color="0.75", lw=0.5, zorder=1)
        lo, hi = min(min(vals), 0.0), max(vals)
        pad = 0.13 * (hi - lo)
        ax.set_xlim(lo - (0.35 * pad if lo < 0 else 0.0) - 0.02, hi + pad)
        # value labels at bar ends; negative bars label right of zero, where
        # the row is empty, so nothing collides with the tick labels
        off = 0.012 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        for y, v in zip(ys, vals):
            x_txt = v + off if v >= 0 else off
            ax.text(x_txt, y, f"{v:.2f}", ha="left", va="center",
                    fontsize=6, color="0.25", zorder=4)
        ax.set_ylim(-0.75, 11.75)
        ax.tick_params(axis="y", length=0)

    # block labels once, in panel (a)
    x0 = ax_a.get_xlim()[0]
    for y_hdr, txt in ((y_hdr_on, "wake supervision on"),
                       (y_hdr_off, "wake supervision off")):
        ax_a.text(x0, y_hdr, txt, ha="left", va="center", fontsize=6.5,
                  style="italic", color="0.3")

    ax_a.set_yticks(ys)
    ax_a.set_yticklabels(labels)
    ax_a.set_xlabel(r"wake-enstrophy probe $R^2$"
                    + "\n" + f"({split}, windowed linear)")
    ax_b.set_xlabel("matched-predictor\n" + rf"merit ($h = {horizon}$)")
    ax_a.text(0.0, 1.015, "(a)", transform=ax_a.transAxes, fontsize=8,
              style="italic", va="bottom")
    ax_b.text(0.0, 1.015, "(b)", transform=ax_b.transAxes, fontsize=8,
              style="italic", va="bottom")

    # decode-floor SSIM as a table-like text column
    ax_s.set_xlim(0, 1)
    ax_s.set_xticks([])
    ax_s.tick_params(axis="y", length=0)
    for side in ax_s.spines.values():
        side.set_visible(False)
    ax_s.axhline(y_sep, color="0.75", lw=0.5, zorder=1)
    for y, v in zip(ys, ssim):
        ax_s.text(0.5, y, f"{v:.3f}", ha="center", va="center",
                  fontsize=6.5, color="0.15")
    ax_s.set_xlabel("decode\nSSIM", fontsize=7)

    handles = [Patch(facecolor=OBJECTIVE_COLOR[o], label=OBJECTIVE_LABEL[o])
               for o in ("predictive", "reconstructive", "bvae", "none", "pod")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6,
               handlelength=1.0, handleheight=0.8, labelspacing=0.4,
               columnspacing=1.2, bbox_to_anchor=(0.55, 0.0))

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    plt.close(fig)

    print(f"wrote {OUT_PDF}")
    for (m, lab, obj, on), w, mr, s in zip(MODELS, wake, merit, ssim):
        head = "ON " if on else "OFF"
        print(f"  {head} {m:<12} ({lab:<15} {obj:<14}) "
              f"wake={w:+.3f} merit={mr:+.3f} ssim={s:.3f}")


if __name__ == "__main__":
    main()
