"""Session 31 Track F figures: the three-curve field-VRMSE forecast panel.

Reads ``outputs/session31/q2_temporal.json`` and draws, per model, the aggregated
field VRMSE against forecast horizon for the three reference curves:
decode floor (decode of the true latent, the best a fixed decoder can do), the ROM
model (decode of the rolled latent), and persistence (hold the last-context DNS
field). The ROM is useful exactly where ``floor < model < persistence``. Small
multiples share axes so the six canonical models read at a glance; built at JFM
textwidth via :mod:`scripts.session21.figstyle`.

    OMP_NUM_THREADS=8 taskset -c 0-7 python scripts/session31/make_figures.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.session21 import figstyle  # noqa: E402
from src.evaluation.report_session31 import (  # noqa: E402
    CANONICAL_MODELS,
    MODEL_LABEL,
    PRIMARY_PREDICTOR,
)

REPO = Path(__file__).resolve().parents[2]
Q2 = REPO / "outputs" / "session31" / "q2_temporal.json"
OUT_DIR = REPO / "outputs" / "session31" / "figures"

# curve -> (colour family key, linestyle, label)
CURVES = {
    "floor": ("oracle", (0, (4, 3)), "decode floor"),
    "model": ("jepa", "-", "ROM (rolled latent)"),
    "persistence": ("fukami", (0, (1, 1)), "persistence"),
}


def make_field_vrmse_panel(q2: dict, predictor: str = PRIMARY_PREDICTOR) -> plt.Figure:
    figstyle.use_style()
    w, h = figstyle.figure_size(1.0, aspect=0.66)
    fig, axes = plt.subplots(2, 3, figsize=(w, h), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, model in zip(axes, CANONICAL_MODELS):
        pred = q2["models"][model]["predictors"][predictor]
        fv = pred["field_vrmse"]
        hz = np.asarray(fv["horizons"], dtype=float)
        for key, (fam, ls, _label) in CURVES.items():
            ax.plot(
                hz,
                fv[key],
                linestyle=ls,
                color=figstyle.FAMILY_COLOR[fam],
                linewidth=1.1,
                marker="",
                label=_label,
            )
        ax.axhline(1.0, color="0.7", linewidth=0.6, linestyle=":", zorder=0)
        ax.set_title(MODEL_LABEL[model], fontsize=8)
        ax.set_xlim(hz.min(), hz.max())
        ax.margins(x=0)
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("field VRMSE")
    axes[0].legend(loc="upper left", fontsize=6.2, handlelength=1.8)
    fig.supxlabel("forecast horizon $h$ (frames)", fontsize=8, y=0.02)
    fig.suptitle(
        "Three-curve forecast field VRMSE (matched ResUNet predictor, Test B, in-window)",
        fontsize=8.2,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--q2", default=str(Q2))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--predictor", default=PRIMARY_PREDICTOR)
    args = p.parse_args(argv)

    q2 = json.loads(Path(args.q2).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = make_field_vrmse_panel(q2, predictor=args.predictor)
    for ext in ("pdf", "png"):
        path = out_dir / f"q2_field_vrmse_three_curve_v2p2.{ext}"
        fig.savefig(path)
        print(f"[fig] wrote {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
