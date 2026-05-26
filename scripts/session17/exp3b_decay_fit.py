"""Session 17, Experiment 3, Part (b): Gaussian decay fit for Y recovery.

Model: R^2_Y(tau) = R^2_peak * exp(-tau^2 / (2 * sigma_tau^2))

Two variants:
  - symmetric Gaussian (single sigma_tau)
  - asymmetric Gaussian (sigma_left for tau < 0, sigma_right for tau > 0)
    motivated by the empirical asymmetry observed in Exp 3(a).

Report sigma_tau in frames and in convective time units:
  sigma_t_c = sigma_tau * dt * U_inf / c
  where dt = 0.05 c/U_inf is the cache timestep, so sigma_t_c = sigma_tau * 0.05.

Acceptance gate (SESSION17_PLAN.md):
  Y R^2 at tau=0 exceeds Y R^2 at |tau|=10 by at least 0.3 AND sigma_tau < 15.

Outputs:
    outputs/session17/exp3/decay_fits.json
    outputs/session17/figures/exp3_param_recovery_vs_tau.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


REPO = Path(__file__).resolve().parents[2]
EXP3 = REPO / "outputs" / "session17" / "exp3"
FIGS = REPO / "outputs" / "session17" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
DT_TC = 0.05  # cache timestep in convective units


def gaussian(tau, peak, sigma):
    return peak * np.exp(-(tau**2) / (2.0 * sigma**2))


def asymmetric_gaussian(tau, peak, sigma_l, sigma_r):
    sigma = np.where(tau < 0, sigma_l, sigma_r)
    return peak * np.exp(-(tau**2) / (2.0 * sigma**2))


def main() -> None:
    summary = json.loads((EXP3 / "per_frame_recovery_summary.json").read_text())
    taus = np.array(summary["taus"], dtype=float)
    fits = {}

    for param in ("G", "D", "Y"):
        sub = summary["per_param"][param]
        tau_arr = np.array(sub["tau"], dtype=float)
        test_b_r2 = np.array(sub["test_b_r2"], dtype=float)
        # Clip R^2 to [0, 1] so the decay model can be meaningful.
        r2_clip = np.maximum(test_b_r2, 0.0)
        # Symmetric fit.
        try:
            popt, pcov = curve_fit(
                gaussian, tau_arr, r2_clip, p0=[max(r2_clip), 10.0],
                bounds=([0.0, 0.5], [1.0, 200.0]),
            )
            sym = {
                "peak": float(popt[0]),
                "sigma_tau": float(popt[1]),
                "sigma_t_c": float(popt[1] * DT_TC),
                "rmse": float(
                    np.sqrt(np.mean((r2_clip - gaussian(tau_arr, *popt)) ** 2))
                ),
            }
        except RuntimeError as e:
            sym = {"error": str(e)}
        # Asymmetric fit.
        try:
            popt2, pcov2 = curve_fit(
                asymmetric_gaussian, tau_arr, r2_clip,
                p0=[max(r2_clip), 10.0, 10.0],
                bounds=([0.0, 0.5, 0.5], [1.0, 200.0, 200.0]),
            )
            asym = {
                "peak": float(popt2[0]),
                "sigma_l": float(popt2[1]),
                "sigma_r": float(popt2[2]),
                "sigma_l_t_c": float(popt2[1] * DT_TC),
                "sigma_r_t_c": float(popt2[2] * DT_TC),
                "rmse": float(
                    np.sqrt(
                        np.mean(
                            (r2_clip - asymmetric_gaussian(tau_arr, *popt2)) ** 2
                        )
                    )
                ),
            }
        except RuntimeError as e:
            asym = {"error": str(e)}
        fits[param] = {
            "tau": tau_arr.tolist(),
            "r2_test_b": test_b_r2.tolist(),
            "r2_test_b_clipped": r2_clip.tolist(),
            "symmetric_gaussian": sym,
            "asymmetric_gaussian": asym,
        }
        print(
            f"[exp3b] {param:>1}: peak_R^2={sym.get('peak', float('nan')):.3f} "
            f"sigma_tau={sym.get('sigma_tau', float('nan')):.2f} frames "
            f"sigma_t_c={sym.get('sigma_t_c', float('nan')):.3f}"
        )
        if "sigma_l" in asym:
            print(
                f"           asym: sigma_l={asym['sigma_l']:.2f} "
                f"sigma_r={asym['sigma_r']:.2f}  rmse={asym['rmse']:.4f}"
            )

    # Acceptance gate for Y.
    Y_tau = np.array(summary["per_param"]["Y"]["tau"], dtype=float)
    Y_r2 = np.array(summary["per_param"]["Y"]["test_b_r2"], dtype=float)
    idx0 = np.argmin(np.abs(Y_tau))
    idx_minus10 = np.argmin(np.abs(Y_tau - (-10)))
    idx_plus10 = np.argmin(np.abs(Y_tau - 10))
    delta_left = float(Y_r2[idx0] - Y_r2[idx_minus10])
    delta_right = float(Y_r2[idx0] - Y_r2[idx_plus10])
    delta_min = min(delta_left, delta_right)
    delta_max = max(delta_left, delta_right)
    sigma_Y = fits["Y"]["symmetric_gaussian"].get("sigma_tau", float("nan"))

    gate = {
        "Y_r2_at_tau0": float(Y_r2[idx0]),
        "Y_r2_at_tau_minus10": float(Y_r2[idx_minus10]),
        "Y_r2_at_tau_plus10": float(Y_r2[idx_plus10]),
        "delta_tau0_minus_minus10": delta_left,
        "delta_tau0_minus_plus10": delta_right,
        "delta_min": delta_min,
        "delta_max": delta_max,
        "sigma_tau_frames": float(sigma_Y),
        "sigma_t_c": float(sigma_Y * DT_TC),
        "gate_delta_min_above_0p3": delta_min >= 0.3,
        "gate_delta_max_above_0p3": delta_max >= 0.3,
        "gate_sigma_below_15": float(sigma_Y) < 15,
        "gate_passes_plan_as_written": (delta_min >= 0.3) and (float(sigma_Y) < 15),
        "gate_passes_min_or_max_0p3": (
            ((delta_min >= 0.3) or (delta_max >= 0.3)) and float(sigma_Y) < 15
        ),
        "note_asymmetry": (
            "Empirically the decay is asymmetric: tau<0 (pre-impact) decay is "
            "sharper than tau>+0 (post-impact). The plan's '|tau|=10' criterion "
            "is reported both ways."
        ),
    }

    out = {"fits": fits, "acceptance_gate": gate}
    (EXP3 / "decay_fits.json").write_text(json.dumps(out, indent=2))
    print(f"[exp3b] wrote {EXP3 / 'decay_fits.json'}")
    print(
        f"[exp3b] gate (Y): delta_left={delta_left:+.3f}  delta_right={delta_right:+.3f}  "
        f"sigma_tau={sigma_Y:.1f}  "
        f"plan_pass={gate['gate_passes_plan_as_written']}  "
        f"min_or_max_pass={gate['gate_passes_min_or_max_0p3']}"
    )

    # Figure: R^2(tau) per parameter with fitted Gaussian overlays.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    tau_dense = np.linspace(taus.min(), taus.max(), 200)
    for ax, param, color in zip(
        axes, ("G", "D", "Y"), ("tab:purple", "tab:green", "tab:red")
    ):
        sub = summary["per_param"][param]
        ax.plot(sub["tau"], sub["test_b_r2"], "o-", color=color, label="Test B")
        ax.plot(sub["tau"], sub["test_c_r2"], "o--", color="gray", label="Test C", alpha=0.7)
        # Fit overlay
        sym = fits[param]["symmetric_gaussian"]
        if "peak" in sym:
            ax.plot(
                tau_dense,
                gaussian(tau_dense, sym["peak"], sym["sigma_tau"]),
                color=color, ls=":", lw=1.5,
                label=(
                    f"Gaussian fit\n"
                    f"peak={sym['peak']:.2f}, "
                    f"$\\sigma_\\tau$={sym['sigma_tau']:.1f}"
                ),
            )
        asym = fits[param]["asymmetric_gaussian"]
        if "peak" in asym:
            ax.plot(
                tau_dense,
                asymmetric_gaussian(
                    tau_dense, asym["peak"], asym["sigma_l"], asym["sigma_r"]
                ),
                color=color, ls="--", lw=1.0, alpha=0.7,
                label=(
                    f"asym: $\\sigma_L$={asym['sigma_l']:.1f}, "
                    f"$\\sigma_R$={asym['sigma_r']:.1f}"
                ),
            )
        ax.axvline(0, color="k", lw=1, alpha=0.6)
        ax.set_title(param)
        ax.set_xlabel(r"frame offset $\tau$ from impact")
        ax.set_ylim(-0.5, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower center")
    axes[0].set_ylabel(r"$R^2$")
    fig.suptitle(
        f"Parameter recovery vs frame offset (KRR on per-frame z)\n"
        f"Y plan-gate {'PASS' if gate['gate_passes_plan_as_written'] else 'FAIL'} "
        f"($\\sigma_\\tau$={sigma_Y:.1f} frames)"
    )
    fig.tight_layout()
    fig.savefig(FIGS / "exp3_param_recovery_vs_tau.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[exp3b] wrote {FIGS / 'exp3_param_recovery_vs_tau.png'}")


if __name__ == "__main__":
    main()
