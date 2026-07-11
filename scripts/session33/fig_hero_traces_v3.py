"""F7 hero figure: assimilated encounter traces (truth vs open-loop vs
pressure-only vs filter analysis).

SESSION_33_MANUSCRIPT_V3.md Section 9 (F7). Data from
outputs/session33/hero_traces.json (frozen D220 filter, representative test_b
encounters picked at the stratum median per the representative-case rule).
Columns = encounters (|G| = 1, 1.5, 2); rows = C_L and wake enstrophy. The
filter band is +-2 ensemble standard deviations of the analysis readout.

Run (CPU):
    taskset -c 16-23 python -m scripts.session33.fig_hero_traces_v3
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "session21"))

import matplotlib.pyplot as plt  # noqa: E402

from figstyle import TEXTWIDTH_IN, use_style  # noqa: E402

S33 = REPO_ROOT / "outputs" / "session33"
OUT = S33 / "figures"

COLORS = {
    "truth": "#000000",
    "filter": "#1b7837",
    "open_loop": "#8c8c8c",
    "pressure_only": "#2166ac",
}


def main():
    use_style()
    blob = json.loads((S33 / "hero_traces.json").read_text())
    traces = blob["traces"]
    n = len(traces)
    fig, axes = plt.subplots(
        2, n, figsize=(TEXTWIDTH_IN, 0.52 * TEXTWIDTH_IN),
        constrained_layout=True, sharex="col",
    )
    if n == 1:
        axes = axes.reshape(2, 1)

    for j, tr in enumerate(traces):
        frames = np.asarray(tr["frames"], dtype=float)
        t = (frames - tr["t_impact"]) * tr["dt_tc"]
        for i, (obs, label) in enumerate(
            (("C_L", r"$C_L$"), ("wake_enstrophy", r"wake enstrophy $\Omega_w$"))
        ):
            ax = axes[i, j]
            truth = np.asarray(tr["truth"][obs])
            filt = np.asarray(tr["filter_analysis"][obs])
            std = np.asarray(tr["filter_analysis"][f"{obs}_ens_std"])
            ol = np.asarray(tr["open_loop"][obs])
            po = np.asarray(tr["pressure_only"][obs])
            ax.fill_between(t, filt - 2 * std, filt + 2 * std,
                            color=COLORS["filter"], alpha=0.18, lw=0)
            ax.plot(t, ol, color=COLORS["open_loop"], lw=0.9, ls="--",
                    label="open-loop rollout")
            ax.plot(t, po, color=COLORS["pressure_only"], lw=0.9, ls=":",
                    label="pressure-only regression")
            ax.plot(t, filt, color=COLORS["filter"], lw=1.1,
                    label=r"filter analysis ($\pm 2\sigma$)")
            ax.plot(t, truth, color=COLORS["truth"], lw=1.1, label="DNS truth")
            ax.axvline(0.0, color="0.75", lw=0.6, zorder=0)
            if i == 0:
                g = abs(tr["envelope_record"]["abs_G"])
                # Session 38 Stage 5 (memo catch 8): archive-signed case_id
                # removed from the header (rule: archive identifiers only in
                # the data appendix); D and Y parsed from it are sign-safe,
                # the G sign stays out pending the s3.5 sign audit.
                import re as _re
                _m = _re.match(r"G[+-][\d.]+_D([\d.]+)_Y([+-][\d.]+)", tr["case_id"])
                _d, _y = (float(_m.group(1)), float(_m.group(2))) if _m else (None, None)
                _sub = f", $D = {_d:g}$, $Y = {_y:+g}$" if _m else ""
                ax.set_title(f"$|G| = {g:g}${_sub}", fontsize=6.5)
            if j == 0:
                ax.set_ylabel(label)
            if i == 1:
                ax.set_xlabel(r"$(t - t_{\mathrm{impact}})\, U_\infty / c$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [labels.index(x) for x in
             ("DNS truth", r"filter analysis ($\pm 2\sigma$)",
              "pressure-only regression", "open-loop rollout")]
    fig.legend([handles[k] for k in order], [labels[k] for k in order],
               ncol=4, frameon=False, fontsize=6,
               loc="upper center", bbox_to_anchor=(0.5, 1.06))

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_hero_traces_v3.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {OUT / 'fig_hero_traces_v3.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
