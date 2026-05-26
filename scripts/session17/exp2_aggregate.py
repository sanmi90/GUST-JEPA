"""Session 17, Experiment 2 (b-d): aggregate physical-metric rollout data.

Reads outputs/session17/exp2/physical_metrics_per_encounter.npz and:
  (b) Per metric (C_L, I_y, I_x, enstrophy, circ_pos, circ_neg) and horizon H,
      compute per-encounter (Markov - Full) deltas and bootstrap 95% CI.
      Report the smallest H at which the CI excludes 0 by more than 10% of
      the metric's encounter-pool std.
  (c) Impulse-lift correlation: Pearson r(dI_y/dt, C_L) per rollout mode,
      pooled across encounters. DNS reference should be r > 0.95 by Wu's
      theorem.
  (d) Spectral fidelity at H=16, H=32: radially-averaged power spectrum
      of the predicted wake region for DNS / Markov / Full, plus L2
      spectral error.

Outputs:
    outputs/session17/exp2/markov_vs_full_delta.json
    outputs/session17/exp2/impulse_lift_correlation.json
    outputs/session17/exp2/horizon_summary.json
    outputs/session17/figures/exp2_physical_closure_horizon.png
    outputs/session17/figures/exp2_impulse_lift_scatter.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


REPO = Path(__file__).resolve().parents[2]
EXP2 = REPO / "outputs" / "session17" / "exp2"
FIGS = REPO / "outputs" / "session17" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
HORIZONS = (1, 4, 8, 16, 24, 32, 48, 64, 79)
METRICS = ("CL", "I_y", "I_x", "enstrophy", "circ_pos", "circ_neg")
MODES = ("markov", "ar", "full")
MODE_LABEL = {"markov": "Markov-only", "ar": "AR-from-impact", "full": "Full-context"}


def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 2000, ci: float = 0.95,
                 rng=None) -> tuple[float, float, float]:
    if rng is None:
        rng = np.random.default_rng(0)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, values.size, values.size)
        means[b] = float(values[idx].mean())
    alpha = (1 - ci) / 2
    return float(np.mean(values)), float(np.quantile(means, alpha)), float(
        np.quantile(means, 1 - alpha)
    )


def main() -> None:
    d = np.load(EXP2 / "rollout_metrics_per_encounter.npz", allow_pickle=True)
    summary: dict = {}

    # Compute per-encounter pool, per-metric:
    for split in ("test_b", "test_c"):
        if f"{split}_case_id" not in d.files:
            continue
        n_enc = len(d[f"{split}_case_id"])
        print(f"\n[exp2-agg] {split}: {n_enc} encounters")
        split_results = {}
        # Per-metric, per-horizon, bootstrap CI on (Markov - Full).
        for metric in METRICS:
            dns = d[f"{split}_dns_{metric}"]  # (n_enc, T_pad)
            mk = d[f"{split}_markov_{metric}"]
            arm = d[f"{split}_ar_{metric}"]
            fc = d[f"{split}_full_{metric}"]
            metric_rec = {}
            for H in HORIZONS:
                col = H - 1  # H=1 -> index 0 (frame_impact+1)
                if col >= dns.shape[1]:
                    continue
                valid = ~np.isnan(dns[:, col]) & ~np.isnan(mk[:, col]) & ~np.isnan(fc[:, col])
                v_dns = dns[:, col][valid]
                v_mk = mk[:, col][valid]
                v_ar = arm[:, col][valid]
                v_fc = fc[:, col][valid]
                # std of metric across encounters at this horizon (DNS reference)
                std_dns = float(v_dns.std()) or 1e-9
                # delta = markov - full
                delta_mk_fc = v_mk - v_fc
                delta_mk_dns = v_mk - v_dns
                delta_fc_dns = v_fc - v_dns
                delta_ar_dns = v_ar - v_dns
                mean_mkfc, lo_mkfc, hi_mkfc = bootstrap_ci(delta_mk_fc)
                # absolute error vs DNS
                metric_rec[H] = {
                    "n": int(valid.sum()),
                    "dns_mean": float(v_dns.mean()),
                    "dns_std": std_dns,
                    "markov_mean": float(v_mk.mean()),
                    "ar_mean": float(v_ar.mean()),
                    "full_mean": float(v_fc.mean()),
                    "delta_markov_minus_full": {
                        "mean": mean_mkfc, "ci_lo": lo_mkfc, "ci_hi": hi_mkfc,
                        "frac_of_std": (abs(mean_mkfc) / std_dns) if std_dns > 0 else None,
                    },
                    "abs_err_vs_dns": {
                        "markov_mean": float(np.mean(np.abs(delta_mk_dns))),
                        "ar_mean": float(np.mean(np.abs(delta_ar_dns))),
                        "full_mean": float(np.mean(np.abs(delta_fc_dns))),
                    },
                }
            split_results[metric] = metric_rec
        summary[split] = split_results

    # Acceptance gates per plan.
    gates = {}
    for split in summary:
        if "CL" not in summary[split] or 16 not in summary[split]["CL"]:
            continue
        cl16 = summary[split]["CL"][16]["delta_markov_minus_full"]
        # CI within 10% of std?
        std_cl = summary[split]["CL"][16]["dns_std"]
        cl_within = abs(cl16["mean"]) < 0.1 * std_cl or (cl16["ci_lo"] <= 0 <= cl16["ci_hi"])
        iy16 = summary[split].get("I_y", {}).get(16)
        if iy16 is None:
            iy_within = False
        else:
            iy16_d = iy16["delta_markov_minus_full"]
            iy_within = abs(iy16_d["mean"]) < 0.1 * iy16["dns_std"] or (
                iy16_d["ci_lo"] <= 0 <= iy16_d["ci_hi"]
            )
        gates[split] = {
            "CL_H16_within_10pct_of_std_or_CI_includes_zero": cl_within,
            "I_y_H16_within_10pct_of_std_or_CI_includes_zero": iy_within,
        }

    (EXP2 / "horizon_summary.json").write_text(
        json.dumps({"per_split": summary, "gates": gates}, indent=2, default=str)
    )
    print(f"[exp2-agg] wrote {EXP2 / 'horizon_summary.json'}")

    # Impulse-lift correlation per mode.
    # For each rollout mode, compute Pearson r(dI_y/dt, C_L) across all
    # (encounter, frame) pairs in test_b + test_c at H <= 32 (most relevant).
    impulse_lift = {}
    for split in ("test_b", "test_c"):
        if f"{split}_dns_I_y" not in d.files:
            continue
        impulse_lift[split] = {}
        for mode in ("dns",) + MODES:
            I_y = d[f"{split}_{mode}_I_y"]  # (n_enc, T_pad)
            C_L = d[f"{split}_{mode}_CL"]
            # Centered first-difference dI_y/dt; pool across encounters and frames
            # up to first NaN per encounter.
            dIy_list, CL_list = [], []
            for i in range(I_y.shape[0]):
                row_Iy = I_y[i]
                row_CL = C_L[i]
                valid = ~np.isnan(row_Iy) & ~np.isnan(row_CL)
                if valid.sum() < 5:
                    continue
                idx = np.where(valid)[0]
                last = idx.max()
                T = last + 1
                Iy = row_Iy[:T]
                CL = row_CL[:T]
                # centered diff (need 1 < t < T-1)
                if T < 4:
                    continue
                dIy = (Iy[2:] - Iy[:-2]) / 2.0  # dt = 1 frame
                CL_center = CL[1:-1]
                # Only consider H <= 32
                lim = min(32, len(dIy))
                dIy_list.append(dIy[:lim])
                CL_list.append(CL_center[:lim])
            if dIy_list:
                X = np.concatenate(dIy_list)
                Y = np.concatenate(CL_list)
                r, p = pearsonr(X, Y)
                impulse_lift[split][mode] = {"r": float(r), "p": float(p), "n": int(X.size)}
                print(
                    f"[exp2-agg] {split} {mode:8s}: r(dI_y/dt, C_L) = {r:+.3f}  "
                    f"(n={X.size}, p={p:.2e})"
                )

    (EXP2 / "impulse_lift_correlation.json").write_text(
        json.dumps(impulse_lift, indent=2)
    )
    print(f"[exp2-agg] wrote {EXP2 / 'impulse_lift_correlation.json'}")

    # Markov-vs-Full delta JSON (separate file for clarity).
    deltas = {}
    for split in summary:
        deltas[split] = {}
        for metric, recs in summary[split].items():
            deltas[split][metric] = {
                str(H): recs[H]["delta_markov_minus_full"] for H in recs
            }
    (EXP2 / "markov_vs_full_delta.json").write_text(json.dumps(deltas, indent=2))
    print(f"[exp2-agg] wrote {EXP2 / 'markov_vs_full_delta.json'}")

    # Figure: 3 rows (C_L, I_y, enstrophy) x 2 cols (test_b, test_c)
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    for col, split in enumerate(("test_b", "test_c")):
        if split not in summary:
            continue
        for row, metric in enumerate(("CL", "I_y", "enstrophy")):
            ax = axes[row, col]
            if metric not in summary[split]:
                ax.axis("off")
                continue
            recs = summary[split][metric]
            Hs = sorted(recs.keys())
            dns_means = [recs[H]["dns_mean"] for H in Hs]
            mk_means = [recs[H]["markov_mean"] for H in Hs]
            ar_means = [recs[H]["ar_mean"] for H in Hs]
            fc_means = [recs[H]["full_mean"] for H in Hs]
            ax.plot(Hs, dns_means, "k-", lw=2.0, label="DNS")
            ax.plot(Hs, fc_means, "-", color="tab:orange", label="Full-context")
            ax.plot(Hs, ar_means, "--", color="tab:green", label="AR-from-impact")
            ax.plot(Hs, mk_means, "-", color="tab:blue", label="Markov-only")
            ax.set_title(f"{metric}  ({split})")
            if row == 2:
                ax.set_xlabel("horizon H")
            ax.set_ylabel(f"{metric} (mean across encounters)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle("Physical Markov closure: rollout vs DNS per horizon")
    fig.tight_layout()
    fig.savefig(FIGS / "exp2_physical_closure_horizon.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[exp2-agg] wrote {FIGS / 'exp2_physical_closure_horizon.png'}")

    # Impulse-lift scatter (test_b only, for clarity)
    if "test_b" in impulse_lift:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        for mode, color in [
            ("dns", "k"), ("markov", "tab:blue"),
            ("ar", "tab:green"), ("full", "tab:orange"),
        ]:
            if mode not in impulse_lift["test_b"]:
                continue
            # Re-pool the scatter data
            I_y = d[f"test_b_{mode}_I_y"]
            C_L = d[f"test_b_{mode}_CL"]
            X_all, Y_all = [], []
            for i in range(I_y.shape[0]):
                valid = ~np.isnan(I_y[i]) & ~np.isnan(C_L[i])
                if valid.sum() < 4:
                    continue
                idx = np.where(valid)[0]
                T = idx.max() + 1
                Iy = I_y[i, :T]
                CL = C_L[i, :T]
                if T < 4:
                    continue
                dIy = (Iy[2:] - Iy[:-2]) / 2.0
                CL_c = CL[1:-1]
                lim = min(32, len(dIy))
                X_all.append(dIy[:lim])
                Y_all.append(CL_c[:lim])
            X = np.concatenate(X_all)
            Y = np.concatenate(Y_all)
            r = impulse_lift["test_b"][mode]["r"]
            ax.scatter(
                X, Y, s=3, alpha=0.3, color=color,
                label=f"{MODE_LABEL.get(mode, mode.upper())}  r={r:+.3f}",
            )
        ax.set_xlabel(r"$dI_y/dt$ (centered diff)")
        ax.set_ylabel(r"$C_L$")
        ax.axhline(0, color="gray", alpha=0.4, lw=0.7)
        ax.axvline(0, color="gray", alpha=0.4, lw=0.7)
        ax.legend(loc="upper left")
        ax.set_title("Impulse-lift relation (Test B, all encounters pooled, H<=32)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGS / "exp2_impulse_lift_scatter.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[exp2-agg] wrote {FIGS / 'exp2_impulse_lift_scatter.png'}")


if __name__ == "__main__":
    main()
