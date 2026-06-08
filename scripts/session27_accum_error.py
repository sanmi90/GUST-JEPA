"""Accumulated-error evolution: wake-enstrophy forecast vs recursive rollout
horizon H, for the winner predictors (canonical scale).

The closure latent at impact+H is produced by H recursive one-step predictions
(z_markov). Plotting R^2 and MAE vs H shows how error compounds over the rollout
and whether the gust conditioning helps more at long horizons.

Lines:
  - conditioned transformer (z_markov)
  - unconditioned transformer cond-0 (z_markov, 3-seed mean +/- std)
  - z_dns representational ceiling (encoded latent, no rollout error)
  - predict-the-mean floor (R^2 = 0)
Output: outputs/session27/accum_error.png + .csv  (test_b)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "session20"))
import exp_closure_r2 as cr  # noqa: E402

HOR = [1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32]
SPLIT = "test_b"
OUTPNG = REPO / "outputs/session27/accum_error.png"
OUTCSV = REPO / "outputs/session27/accum_error.csv"


def curve(tag, mode):
    rows = cr.evaluate(tag, "jepa", 64, dns, tuple(HOR), [SPLIT], n_boot=50)
    r2 = {}; mae = {}
    for r in rows:
        if r["metric"] == "wake_enstrophy" and r["mode"] == mode:
            r2[r["horizon"]] = r["r2"]; mae[r["horizon"]] = r["mae"]
    return ([r2.get(h, np.nan) for h in HOR], [mae.get(h, np.nan) for h in HOR])


dns = np.load(cr.DNS_METRICS_PATH, allow_pickle=True)
cond_r2, cond_mae = curve("jepa_d64_test1_noBN", "z_markov")
ceil_r2, ceil_mae = curve("jepa_d64_test1_noBN", "z_dns")
unc = [curve(f"test1_noBN_predc0_s{s}", "z_markov") for s in (0, 1, 2)]
unc_r2 = np.array([u[0] for u in unc]); unc_mae = np.array([u[1] for u in unc])
ur2_m, ur2_s = np.nanmean(unc_r2, 0), np.nanstd(unc_r2, 0)
uma_m, uma_s = np.nanmean(unc_mae, 0), np.nanstd(unc_mae, 0)

plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
fig, (axr, axm) = plt.subplots(1, 2, figsize=(10.5, 4.0))
NAVY, ACC, GRY = "#1F3864", "#C0520F", "#8A8F98"

axr.axhline(0, color=GRY, lw=1, ls=":")
axr.plot(HOR, ceil_r2, color=GRY, lw=1.6, ls="--", label="z_dns ceiling (encoded)")
axr.plot(HOR, cond_r2, color=NAVY, lw=2.2, marker="o", ms=4, label="conditioned (with c)")
axr.plot(HOR, ur2_m, color=ACC, lw=2.2, marker="s", ms=4, label="unconditioned (no c)")
axr.fill_between(HOR, ur2_m - ur2_s, ur2_m + ur2_s, color=ACC, alpha=0.18)
axr.axvline(16, color="0.8", lw=1); axr.set_xlabel("rollout horizon H (recursive steps)")
axr.set_ylabel("held-out wake-enstrophy $R^2$"); axr.set_title("forecast skill vs horizon")
axr.legend(frameon=False, fontsize=9); axr.grid(alpha=0.25)

axm.plot(HOR, cond_mae, color=NAVY, lw=2.2, marker="o", ms=4, label="conditioned (with c)")
axm.plot(HOR, uma_m, color=ACC, lw=2.2, marker="s", ms=4, label="unconditioned (no c)")
axm.fill_between(HOR, uma_m - uma_s, uma_m + uma_s, color=ACC, alpha=0.18)
axm.plot(HOR, ceil_mae, color=GRY, lw=1.6, ls="--", label="z_dns ceiling")
axm.axvline(16, color="0.8", lw=1); axm.set_xlabel("rollout horizon H (recursive steps)")
axm.set_ylabel("wake-enstrophy MAE"); axm.set_title("accumulated error vs horizon")
axm.legend(frameon=False, fontsize=9); axm.grid(alpha=0.25)

fig.suptitle("Recursive-rollout error accumulation, wake enstrophy (test_b, canonical scale)", y=1.02)
fig.tight_layout()
OUTPNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPNG, dpi=170, bbox_inches="tight")

import csv
with open(OUTCSV, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["H", "cond_r2", "uncond_r2_mean", "uncond_r2_std", "ceil_r2",
                                   "cond_mae", "uncond_mae_mean", "uncond_mae_std", "ceil_mae"])
    for i, h in enumerate(HOR):
        w.writerow([h, round(cond_r2[i], 4), round(ur2_m[i], 4), round(ur2_s[i], 4), round(ceil_r2[i], 4),
                    round(cond_mae[i], 3), round(uma_m[i], 3), round(uma_s[i], 3), round(ceil_mae[i], 3)])
print("wrote", OUTPNG, "and", OUTCSV)
print("H   cond_R2  uncondR2(+/-)   ceil")
for i, h in enumerate(HOR):
    print(f"{h:3d}  {cond_r2[i]:+.3f}   {ur2_m[i]:+.3f}({ur2_s[i]:.3f})   {ceil_r2[i]:+.3f}")
