"""Topology mechanism (CORRECTED): rollout-PRESERVATION of the latent encounter's
single H1 cycle. Per encounter, compare the dominant-loop prominence (max H1
lifetime / cloud diameter) of the TRUE latent trajectory (z_dns) vs the markov
ROLLOUT (z_markov). A representation that stays on its cyclic manifold under
rollout preserves the loop (ratio ~1); a reconstructive latent's loop collapses.

Compares the unconditioned JEPAs (own predictor) to the production conditioned
JEPA and Fukami references. Reuses loop_summary() from exp_persistent_homology.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "session20"))
import exp_persistent_homology as PH  # noqa: E402

SPLIT = "test_b"
PROD = REPO / "outputs/session18/exp_b1_test3"
FAMILIES = [
    ("tf-no-c", REPO / f"outputs/session27/rollouts_noc_tf/{SPLIT}.npz"),
    ("lstm-no-c", REPO / f"outputs/session27/rollouts_noc_lstm/{SPLIT}.npz"),
    ("JEPA (cond, prod)", PROD / f"rollouts_jepa_d64_test1_noBN/{SPLIT}.npz"),
    ("Fukami AE", PROD / f"rollouts_fukami_d64_noBN/{SPLIT}.npz"),
]


def main():
    res = {}
    print("=== loop preservation under rollout (test_b): rel = maxH1lifetime/clouddiam ===")
    print(f"{'family':18s} {'dns rel(med)':>12} {'markov rel(med)':>15} {'ratio mk/dns(med)':>17}")
    for name, npz in FAMILIES:
        if not npz.exists():
            print(f"  {name:18s} MISSING {npz}"); continue
        b = np.load(npz, allow_pickle=True)
        zd = b["z_dns"].astype(np.float32); zm = b["z_markov"].astype(np.float32)
        dns_rel, mk_rel, ratio = [], [], []
        for i in range(zd.shape[0]):
            sd = PH.loop_summary(zd[i]); sm = PH.loop_summary(zm[i])
            dns_rel.append(sd["max_lifetime_rel"]); mk_rel.append(sm["max_lifetime_rel"])
            if sd["max_lifetime_rel"] > 1e-6:
                ratio.append(sm["max_lifetime_rel"] / sd["max_lifetime_rel"])
        res[name] = {"dns": np.array(dns_rel), "mk": np.array(mk_rel), "ratio": np.array(ratio)}
        print(f"  {name:18s} {np.median(dns_rel):>12.3f} {np.median(mk_rel):>15.3f} "
              f"{np.median(ratio):>17.3f}", flush=True)

    names = [n for n, _ in FAMILIES if n in res]
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    data = [np.clip(res[n]["ratio"], 0, 2.0) for n in names]
    bp = ax.boxplot(data, labels=names, showfliers=False, patch_artist=True)
    cols = ["#C0520F", "#C0520F", "#2E7D4F", "#3C6FB0"]
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.45)
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6, label="loop fully preserved")
    ax.set_ylabel("loop preservation under rollout\n(markov / dns, clipped at 2)")
    ax.set_title("Latent topology: does the cycle survive the rollout? (test_b)", fontsize=9)
    ax.tick_params(axis="x", labelsize=7, rotation=15); ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = REPO / "outputs/session27/topology_uncond.pdf"
    fig.savefig(out, bbox_inches="tight"); print(f"\n[topology] figure -> {out}")


if __name__ == "__main__":
    main()
