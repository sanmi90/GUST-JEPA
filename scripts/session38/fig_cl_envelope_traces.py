"""C_L estimation traces across the gust-intensity envelope (Session 38).

One row of C_L panels, columns |G| = 1, 2, 3, 4: DNS truth vs the frozen
filter analysis (+-2 ensemble sd) vs the open-loop rollout, from the
pre-impact shedding through impact and relaxation. Data from
outputs/session38/hero_traces_envelope.json (dump_hero_traces re-run of the
FROZEN envelope filter, model jepa_pool_vec, envelope_vec configuration
verbatim; representative encounters per |G| stratum by the median rule,
test_b for |G| <= 3 and test_c for |G| = 4).

Run (CPU): taskset -c 0-15 python -m scripts.session38.fig_cl_envelope_traces
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "session21"))

import matplotlib.pyplot as plt  # noqa: E402

from figstyle import TEXTWIDTH_IN, use_style  # noqa: E402

SRC = REPO_ROOT / "outputs/session38/hero_traces_envelope.json"
OUT = REPO_ROOT / "paper/sections/figures/results"

COLORS = {
    "truth": "#000000",
    "filter": "#1b7837",
    "open_loop": "#8c8c8c",
}

CASE_RE = re.compile(r"G[+-][\d.]+_D([\d.]+)_Y([+-][\d.]+)")


def main() -> int:
    use_style()
    blob = json.loads(SRC.read_text())
    traces = sorted(blob["traces"], key=lambda tr: tr["envelope_record"]["abs_G"])
    n = len(traces)
    fig, axes = plt.subplots(
        1, n, figsize=(TEXTWIDTH_IN, 0.30 * TEXTWIDTH_IN),
        constrained_layout=True, sharex=True,
    )
    for j, tr in enumerate(traces):
        ax = axes[j]
        frames = np.asarray(tr["frames"], dtype=float)
        t = (frames - tr["t_impact"]) * tr["dt_tc"]
        truth = np.asarray(tr["truth"]["C_L"])
        filt = np.asarray(tr["filter_analysis"]["C_L"])
        std = np.asarray(tr["filter_analysis"]["C_L_ens_std"])
        ol = np.asarray(tr["open_loop"]["C_L"])
        ax.fill_between(t, filt - 2 * std, filt + 2 * std,
                        color=COLORS["filter"], alpha=0.18, lw=0)
        ax.plot(t, ol, color=COLORS["open_loop"], lw=0.9, ls="--",
                label="open-loop rollout")
        ax.plot(t, filt, color=COLORS["filter"], lw=1.1,
                label=r"filter analysis ($\pm 2\sigma$)")
        ax.plot(t, truth, color=COLORS["truth"], lw=1.1, label="DNS truth")
        ax.axvline(0.0, color="0.75", lw=0.6, zorder=0)
        rec = tr["envelope_record"]
        g = rec["abs_G"]
        split = {"test_b": "in-distribution test", "test_c": "boundary test"}.get(
            rec["split"], rec["split"])
        m = CASE_RE.match(tr["case_id"])
        sub = f", $D = {float(m.group(1)):g}$, $Y = {float(m.group(2)):+g}$" if m else ""
        ax.set_title(f"$|G| = {g:g}${sub}\n({split})", fontsize=6.5)
        ax.set_xlabel(r"$(t - t_{\mathrm{impact}})\, U_\infty / c$")
        if j == 0:
            ax.set_ylabel(r"$C_L$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.14), fontsize=6.5)
    for ext in ("pdf", "png"):
        path = OUT / f"fig_cl_envelope_traces_v4.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
