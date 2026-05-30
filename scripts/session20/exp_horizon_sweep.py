"""Session 20 Track G: horizon sweep H in {1,4,8,16,32,64}.

Reads the held-out closure CSV (exp_closure_r2.py) and produces, for the B1
baselines on test_b and test_c (z_markov mode): closure R^2 and MAE vs H per
observable, the horizon at which each family drops below R^2 = 0.5, and a
graceful-vs-abrupt readout (does the predictive latent degrade smoothly while
the reconstructive one fails at the drift onset?).

Output: outputs/session20/horizon_sweep/{horizon_sweep.png,.pdf,horizon_summary.json}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "outputs/session20/closure_r2/closure_r2_heldout.csv"
OUT = REPO / "outputs/session20/horizon_sweep"
# Representative families (d=64 + the matched-capacity d=32 JEPA).
FAMILIES = {
    "jepa_d64_test1_noBN": ("JEPA d=64", "#1f77b4", "-"),
    "jepa_d32_noBN": ("JEPA d=32", "#17becf", "--"),
    "fukami_d64_noBN": ("Fukami d=64", "#d62728", "-"),
    "pod_d64_noBN": ("POD d=64", "#7f7f7f", "-"),
}
KEY_METRICS = ["wake_enstrophy", "C_L", "circulation_neg"]


def first_below(hs, r2s, thresh=0.5):
    for h, r in zip(hs, r2s):
        if r < thresh:
            return int(h)
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV)
    df = df[df["mode"] == "z_markov"]
    summary = {}
    fig, axes = plt.subplots(2, len(KEY_METRICS), figsize=(5 * len(KEY_METRICS), 8), squeeze=False)
    for r, split in enumerate(["test_b", "test_c"]):
        for c, metric in enumerate(KEY_METRICS):
            ax = axes[r][c]
            for tag, (label, color, ls) in FAMILIES.items():
                sub = df[(df.baseline == tag) & (df.split == split) & (df.metric == metric)].sort_values("horizon")
                if sub.empty:
                    continue
                hs, r2s = sub.horizon.values, sub.r2.values
                ax.plot(hs, r2s, marker="o", color=color, ls=ls, label=label, lw=1.6, ms=4)
                summary.setdefault(split, {}).setdefault(metric, {})[tag] = {
                    "horizons": hs.tolist(), "r2": r2s.tolist(),
                    "r2_below_0.5_at_H": first_below(hs, r2s),
                }
            ax.axhline(0.5, color="0.6", ls=":", lw=1)
            ax.axhline(0.0, color="0.8", ls="-", lw=0.8)
            ax.set_title(f"{split}: {metric}", fontsize=10)
            ax.set_xlabel("horizon H (frames past impact)")
            if c == 0:
                ax.set_ylabel("held-out closure $R^2$")
            ax.set_ylim(-1.0, 1.0)
            ax.grid(alpha=0.25)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, framealpha=0.7)
    fig.suptitle("Track G: held-out closure $R^2$ vs rollout horizon (z_markov)", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "horizon_sweep.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / "horizon_sweep.pdf", bbox_inches="tight")
    plt.close(fig)

    # Graceful-vs-abrupt: wake_enstrophy R^2 drop from H=8 to H=64, JEPA vs Fukami.
    def drop(tag, split, metric="wake_enstrophy"):
        s = summary.get(split, {}).get(metric, {}).get(tag)
        if not s:
            return None
        r2 = dict(zip(s["horizons"], s["r2"]))
        return {"H8": r2.get(8), "H64": r2.get(64), "drop_8_to_64": (r2.get(8, np.nan) - r2.get(64, np.nan))}
    summary["graceful_vs_abrupt_wake_testb"] = {
        t: drop(t, "test_b") for t in FAMILIES
    }
    with open(OUT / "horizon_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[track-g] wrote", OUT / "horizon_sweep.png", "and horizon_summary.json")
    for t, lab in [(k, v[0]) for k, v in FAMILIES.items()]:
        s = summary.get("test_b", {}).get("wake_enstrophy", {}).get(t, {})
        print(f"  {lab:14s} wake R^2 drops below 0.5 at H={s.get('r2_below_0.5_at_H')}")


if __name__ == "__main__":
    main()
