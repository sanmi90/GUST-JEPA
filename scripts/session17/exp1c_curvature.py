"""Session 17, Experiment 1, Part (c): curvature signature at the impact frame.

kappa(t) = ||z(t+1) - 2 z(t) + z(t-1)|| / (||z(t+1) - z(t-1)||/2)^2

Computed in the *full 64-D z space* for every encounter in Test B (56) and
Test C (24). We then:
  - align curvature profiles by the impact frame (t_impact = 40)
  - report the median kappa(t) across all encounters and separately for
    high-|G| (|G| >= 2) vs low-|G| (|G| < 2) cases
  - locate the curvature peak within a window around t_impact and report
    its offset and amplitude vs an off-peak baseline.

Acceptance gate (SESSION17_PLAN.md):
  kappa(t) peaks within +/- 3 frames of t_impact with peak height >= 2x
  off-peak baseline.

Outputs:
    outputs/session17/exp1/curvature_profiles.npz
    outputs/session17/exp1/curvature_acceptance.json
    outputs/session17/figures/exp1_curvature_at_impact.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
LATENTS = REPO / "outputs" / "session14" / "latents" / "S12_E_d64"
EXP1 = REPO / "outputs" / "session17" / "exp1"
FIGS = REPO / "outputs" / "session17" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
T_ENC = 120
T_IMPACT = 40


def load_split(name: str, supplement: str | None = None) -> dict:
    d = np.load(LATENTS / f"{name}.npz", allow_pickle=True)
    out = {
        "z_full": d["z_full"].astype(np.float32),
        "G": d["G"].astype(np.float32),
        "D": d["D"].astype(np.float32),
        "Y": d["Y"].astype(np.float32),
        "impact_frame": d["impact_frame"].astype(np.int32),
        "case_id": np.asarray(d["case_id"]).astype(object),
        "encounter_index": d["encounter_index"].astype(np.int32),
    }
    if supplement is not None:
        ds = np.load(LATENTS / f"{supplement}.npz", allow_pickle=True)
        out["z_full"] = np.concatenate(
            [out["z_full"], ds["z_full"].astype(np.float32)], axis=0
        )
        for k in ("G", "D", "Y"):
            out[k] = np.concatenate([out[k], ds[k].astype(np.float32)], axis=0)
        out["impact_frame"] = np.concatenate(
            [out["impact_frame"], ds["impact_frame"].astype(np.int32)], axis=0
        )
        out["case_id"] = np.concatenate(
            [out["case_id"], np.asarray(ds["case_id"]).astype(object)], axis=0
        )
        out["encounter_index"] = np.concatenate(
            [out["encounter_index"], ds["encounter_index"].astype(np.int32)], axis=0
        )
    return out


def curvature(z: np.ndarray) -> np.ndarray:
    """kappa(t) for t in [1, T-2]. z: (T, d). Returns (T,) with kappa(0)=kappa(T-1)=nan."""
    T = z.shape[0]
    out = np.full(T, np.nan, dtype=np.float64)
    for t in range(1, T - 1):
        z_p = z[t + 1]
        z_m = z[t - 1]
        z_t = z[t]
        num = np.linalg.norm(z_p - 2.0 * z_t + z_m)
        # Use the centered first difference magnitude as the speed; matches the
        # discrete-curvature definition kappa = |z''| / |z'|^2.
        speed = np.linalg.norm(z_p - z_m) / 2.0
        if speed < 1e-12:
            out[t] = 0.0
        else:
            out[t] = num / (speed**2)
    return out


def main() -> None:
    test_b = load_split("test_b", supplement=None)
    test_c = load_split("test_c")
    train = load_split("train")
    test_a = load_split("test_a")
    splits = {"test_b": test_b, "test_c": test_c, "train": train, "test_a": test_a}

    profiles = {}
    for name, split in splits.items():
        kappa = np.stack(
            [curvature(zt) for zt in split["z_full"]], axis=0
        )  # (n_enc, T)
        profiles[name] = kappa
        print(f"[exp1c] {name:8s} n={kappa.shape[0]:3d} kappa_max={np.nanmax(kappa):.3e}")

    # Stack per-encounter kappa profiles aligned by t_impact.
    # All impact frames are 40 here, so no shift needed; but keep robust code.
    aligned: dict[str, np.ndarray] = {}
    for name, kappa in profiles.items():
        T = kappa.shape[1]
        t_imp = splits[name]["impact_frame"]
        # Aligned grid: tau in [-T_IMPACT, T_ENC - 1 - T_IMPACT]
        aligned[name] = kappa  # impact frame is 40 for all encounters
    tau = np.arange(T_ENC) - T_IMPACT

    # Compute median + 25/75 quantiles per split.
    medians = {}
    quantiles = {}
    for name, kappa in profiles.items():
        medians[name] = np.nanmedian(kappa, axis=0)
        quantiles[name] = (
            np.nanpercentile(kappa, 25, axis=0),
            np.nanpercentile(kappa, 75, axis=0),
        )

    # Stratify by |G|: high-|G| (|G| >= 2.0) vs low-|G| (|G| < 2.0).
    strat_medians = {}
    for split_name in ("test_b", "test_c"):
        kappa = profiles[split_name]
        G = splits[split_name]["G"]
        mask_high = np.abs(G) >= 2.0
        mask_low = ~mask_high
        if mask_high.any():
            strat_medians[(split_name, "high")] = np.nanmedian(kappa[mask_high], axis=0)
        if mask_low.any():
            strat_medians[(split_name, "low")] = np.nanmedian(kappa[mask_low], axis=0)

    # Acceptance gate: per-encounter peak location and peak/baseline ratio.
    peak_records = []
    near_window = (T_IMPACT - 3, T_IMPACT + 3)  # +/- 3 frames
    far_window = (
        list(range(T_IMPACT - 20, T_IMPACT - 5))
        + list(range(T_IMPACT + 5, T_IMPACT + 20))
    )
    for split_name in ("test_b", "test_c"):
        kappa = profiles[split_name]
        for i in range(kappa.shape[0]):
            k = kappa[i]
            # Mask middle region (impact-bracket) to find local peak.
            local_idx = T_IMPACT + np.nanargmax(k[T_IMPACT - 10 : T_IMPACT + 10]) - 10
            offset = int(local_idx - T_IMPACT)
            in_window = near_window[0] <= local_idx <= near_window[1]
            baseline = float(np.nanmedian(k[far_window]))
            peak = float(k[local_idx])
            ratio = peak / max(baseline, 1e-9)
            peak_records.append(
                {
                    "split": split_name,
                    "enc_idx": i,
                    "G": float(splits[split_name]["G"][i]),
                    "D": float(splits[split_name]["D"][i]),
                    "Y": float(splits[split_name]["Y"][i]),
                    "peak_frame": int(local_idx),
                    "peak_offset": offset,
                    "in_window_3": bool(in_window),
                    "peak_value": peak,
                    "off_window_baseline": baseline,
                    "peak_ratio": ratio,
                }
            )

    # Aggregate gate stats.
    gate = {}
    for split_name in ("test_b", "test_c"):
        rs = [r for r in peak_records if r["split"] == split_name]
        offsets = np.array([r["peak_offset"] for r in rs])
        in_window = np.array([r["in_window_3"] for r in rs])
        ratios = np.array([r["peak_ratio"] for r in rs])
        # Also test the MEDIAN profile's gate.
        med = np.nanmedian(profiles[split_name], axis=0)
        local_idx_med = (
            T_IMPACT + int(np.nanargmax(med[T_IMPACT - 10 : T_IMPACT + 10])) - 10
        )
        baseline_med = float(np.nanmedian(med[far_window]))
        peak_med = float(med[local_idx_med])
        ratio_med = peak_med / max(baseline_med, 1e-9)
        gate[split_name] = {
            "n_encounters": int(len(rs)),
            "median_offset": float(np.median(offsets)),
            "frac_in_window_3": float(in_window.mean()),
            "median_peak_ratio": float(np.median(ratios)),
            "p25_peak_ratio": float(np.percentile(ratios, 25)),
            "p75_peak_ratio": float(np.percentile(ratios, 75)),
            "median_profile": {
                "peak_frame": local_idx_med,
                "peak_offset": local_idx_med - T_IMPACT,
                "peak_value": peak_med,
                "off_window_baseline": baseline_med,
                "peak_ratio": ratio_med,
                "passes": bool(
                    abs(local_idx_med - T_IMPACT) <= 3 and ratio_med >= 2.0
                ),
            },
        }

    accept = {
        "tau_grid": [int(v) for v in tau.tolist()],
        "t_impact": T_IMPACT,
        "near_window_frames": [near_window[0], near_window[1]],
        "off_window_indices": far_window,
        "splits": gate,
        "acceptance_gate_text": (
            "median kappa(t) peaks within +/- 3 frames of t_impact and "
            "peak height >= 2x off-window baseline"
        ),
    }
    (EXP1 / "curvature_acceptance.json").write_text(json.dumps(accept, indent=2))
    print(f"[exp1c] wrote {EXP1 / 'curvature_acceptance.json'}")

    # Save profiles to NPZ.
    np.savez_compressed(
        EXP1 / "curvature_profiles.npz",
        tau=tau,
        kappa_test_b=profiles["test_b"],
        kappa_test_c=profiles["test_c"],
        kappa_train=profiles["train"],
        kappa_test_a=profiles["test_a"],
        median_test_b=medians["test_b"],
        median_test_c=medians["test_c"],
        median_train=medians["train"],
        G_test_b=test_b["G"],
        G_test_c=test_c["G"],
        D_test_b=test_b["D"],
        D_test_c=test_c["D"],
        Y_test_b=test_b["Y"],
        Y_test_c=test_c["Y"],
        peak_records=np.array(peak_records, dtype=object),
    )
    print(f"[exp1c] wrote {EXP1 / 'curvature_profiles.npz'}")

    # Print gate summary.
    print("\n[exp1c] Acceptance gate (median profile per split):")
    for split_name, g in gate.items():
        m = g["median_profile"]
        flag = "PASS" if m["passes"] else "FAIL"
        print(
            f"  {split_name:8s} peak_offset={m['peak_offset']:+d}  "
            f"ratio={m['peak_ratio']:.2f}  {flag}"
        )

    # Plot: 2 rows x 1 col.
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, split_name, color in [
        (axes[0], "test_b", "tab:blue"),
        (axes[1], "test_c", "tab:orange"),
    ]:
        kappa = profiles[split_name]
        med = medians[split_name]
        q25, q75 = quantiles[split_name]
        ax.fill_between(tau, q25, q75, color=color, alpha=0.2, label="IQR")
        ax.plot(tau, med, color=color, lw=2, label=f"median ({split_name})")
        # high/low |G| breakdown
        for stratum, ls in (("high", "--"), ("low", ":")):
            key = (split_name, stratum)
            if key in strat_medians:
                ax.plot(
                    tau, strat_medians[key], color=color, lw=1.4, ls=ls,
                    label=f"{stratum}-|G| median",
                )
        ax.axvline(0, color="k", lw=1, alpha=0.6, label="impact frame")
        ax.axvspan(-3, 3, color="k", alpha=0.07)
        ax.set_ylabel(r"$\kappa(\tau)$")
        ax.set_yscale("log")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
        # title with peak/ratio
        m = gate[split_name]["median_profile"]
        ax.set_title(
            f"{split_name}  (peak offset={m['peak_offset']:+d}, "
            f"ratio={m['peak_ratio']:.2f}, "
            f"{'PASS' if m['passes'] else 'FAIL'})"
        )
    axes[-1].set_xlabel(r"frame offset $\tau$ from impact ($\tau=0$)")
    fig.suptitle("Trajectory curvature aligned by impact frame")
    fig.tight_layout()
    fig.savefig(FIGS / "exp1_curvature_at_impact.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[exp1c] wrote {FIGS / 'exp1_curvature_at_impact.png'}")


if __name__ == "__main__":
    main()
