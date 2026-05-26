"""Session 16 Exp 4: comparison figure of latent RMSE vs horizon
across markov-only / AR-from-impact / full-context rollout modes.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "session16" / "exp4"
FIG_DIR = REPO / "outputs" / "session16" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    summary = json.loads((OUT / "markov_closure.json").read_text())
    horizons = summary["horizons"]
    splits = list(summary["per_split_summary"].keys())

    fig, axes = plt.subplots(1, len(splits), figsize=(4.5 * len(splits), 4), sharey=False)
    if len(splits) == 1:
        axes = [axes]

    for ax, split in zip(axes, splits):
        s = summary["per_split_summary"][split]
        ax.plot(horizons, s["markov"]["mean_by_horizon"], "o-", label="Markov-only (mask: z_impact + self)", linewidth=2)
        ax.plot(horizons, s["ar_from_impact"]["mean_by_horizon"], "s-", label="AR from z_impact (1-frame seed, sliding ctx)", linewidth=2)
        ax.plot(horizons, s["full_context"]["mean_by_horizon"], "^-", label="Full context (32-frame seed ending at impact)", linewidth=2)
        ax.set_xlabel("horizon H (frames post-impact)")
        ax.set_ylabel("latent RMSE  ||z_pred[t] - z_dns[t]||/sqrt(d)")
        n_first = s["markov"]["count_by_horizon"][0]
        ax.set_title(f"{split} (N={n_first})")
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
        if split == splits[0]:
            ax.legend(loc="upper left", fontsize=8)

    plt.suptitle(
        "Exp 4: Markov closure of the impact-frame latent.  z_impact is approximately a Markov "
        "sufficient statistic at short-to-medium horizons (markov ~= full_context for H<=8); "
        "AR adds value at longer horizons via accumulated predicted state."
    )
    plt.tight_layout()
    fig_path = FIG_DIR / "exp4_markov_closure.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[exp4-fig] wrote {fig_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
