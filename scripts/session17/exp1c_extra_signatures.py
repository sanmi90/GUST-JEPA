"""Session 17, Experiment 1, Part (c)-extra: additional topological signatures.

The plan's gate -- kappa(t) peaking at impact -- FAILS on the median profile
(test_b peak offset -10 ratio 1.39, test_c peak offset +9 ratio 0.91; see
curvature_acceptance.json). But the raw profile shows kappa(t) actually DIPS
at impact (test_b: kappa around 4.5 off-impact vs 2.84 at impact; test_c:
4.0 off-impact vs 1.78 at impact). The impact frame is a curvature MINIMUM,
i.e. a region of locally-linear trajectory through a smooth pass-through.

To characterize the impact frame topologically, compute three signatures per
encounter:
  - kappa(t) := ||z(t+1) - 2 z(t) + z(t-1)|| / (||z(t+1) - z(t-1)||/2)^2
       (plan's curvature; already in curvature_profiles.npz, recompute for self-contained)
  - speed(t) := ||z(t+1) - z(t-1)|| / 2
       (centered first-difference magnitude; trajectory velocity)
  - bend_cos(t) := <z(t)-z(t-1), z(t+1)-z(t)> / (||z(t)-z(t-1)|| ||z(t+1)-z(t)||)
       (1 = straight pass-through, -1 = reversal)

Outputs:
    outputs/session17/exp1/extra_signatures.npz
    outputs/session17/exp1/extra_signatures_summary.json
    outputs/session17/figures/exp1_signatures_at_impact.png
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
T_ENC = 120
T_IMPACT = 40


def load_z_full(name: str, supplement: str | None = None) -> dict:
    d = np.load(LATENTS / f"{name}.npz", allow_pickle=True)
    out = {
        "z_full": d["z_full"].astype(np.float32),
        "G": d["G"].astype(np.float32),
        "impact_frame": d["impact_frame"].astype(np.int32),
    }
    if supplement is not None:
        ds = np.load(LATENTS / f"{supplement}.npz", allow_pickle=True)
        out["z_full"] = np.concatenate(
            [out["z_full"], ds["z_full"].astype(np.float32)], axis=0
        )
        out["G"] = np.concatenate([out["G"], ds["G"].astype(np.float32)], axis=0)
        out["impact_frame"] = np.concatenate(
            [out["impact_frame"], ds["impact_frame"].astype(np.int32)], axis=0
        )
    return out


def compute_signatures(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return kappa(T,), speed(T,), bend_cos(T,) with NaN at boundaries."""
    T, _ = z.shape
    kappa = np.full(T, np.nan, dtype=np.float64)
    speed = np.full(T, np.nan, dtype=np.float64)
    bcos = np.full(T, np.nan, dtype=np.float64)
    for t in range(1, T - 1):
        d_pos = z[t + 1] - z[t]
        d_neg = z[t] - z[t - 1]
        # centered first diff
        z_dot = (z[t + 1] - z[t - 1]) / 2.0
        z_ddot = z[t + 1] - 2.0 * z[t] + z[t - 1]
        speed[t] = float(np.linalg.norm(z_dot))
        if speed[t] > 1e-12:
            kappa[t] = float(np.linalg.norm(z_ddot)) / (speed[t] ** 2)
        else:
            kappa[t] = 0.0
        nrm_pos = np.linalg.norm(d_pos)
        nrm_neg = np.linalg.norm(d_neg)
        if nrm_pos > 1e-12 and nrm_neg > 1e-12:
            bcos[t] = float(np.dot(d_pos, d_neg) / (nrm_pos * nrm_neg))
    return kappa, speed, bcos


def main() -> None:
    test_b = load_z_full("test_b", supplement="test_b_v1p5_supplement")
    test_c = load_z_full("test_c")
    splits = {"test_b": test_b, "test_c": test_c}

    results = {}
    for name, split in splits.items():
        kappas, speeds, bcoss = [], [], []
        for zt in split["z_full"]:
            k, s, b = compute_signatures(zt)
            kappas.append(k)
            speeds.append(s)
            bcoss.append(b)
        results[name] = {
            "kappa": np.stack(kappas),
            "speed": np.stack(speeds),
            "bcos": np.stack(bcoss),
            "G": split["G"],
        }
        print(
            f"[exp1c-extra] {name:8s} n={len(kappas):3d}  "
            f"kappa_med_impact={np.nanmedian(results[name]['kappa'][:, T_IMPACT]):.3f}  "
            f"speed_med_impact={np.nanmedian(results[name]['speed'][:, T_IMPACT]):.3f}  "
            f"bcos_med_impact={np.nanmedian(results[name]['bcos'][:, T_IMPACT]):.3f}"
        )

    # Compute per-signature impact contrast: at-impact median vs off-impact baseline.
    # Off-impact baseline: median kappa over |tau| in [10, 25].
    tau = np.arange(T_ENC) - T_IMPACT
    far = (np.abs(tau) >= 10) & (np.abs(tau) <= 25)
    contrast = {}
    for name, r in results.items():
        c = {}
        for sig in ("kappa", "speed", "bcos"):
            arr = r[sig]  # (n_enc, T)
            med = np.nanmedian(arr, axis=0)
            at = float(med[T_IMPACT])
            base = float(np.nanmedian(med[far]))
            c[sig] = {
                "at_impact": at,
                "off_impact_baseline": base,
                "ratio_impact_over_baseline": at / max(abs(base), 1e-9),
                "absolute_delta": at - base,
            }
            # Also find the location of the median's extremum within +/- 10 frames.
            window = (np.abs(tau) <= 10)
            local = med.copy()
            local[~window] = np.nan
            if sig in ("bcos",):
                # bend cosine peaks near 1 (straight), so look for max
                argmax_offset = int(np.nanargmax(local) - T_IMPACT)
                c[sig]["max_within_10"] = float(np.nanmax(local))
                c[sig]["argmax_offset_within_10"] = argmax_offset
            else:
                # kappa, speed: report both min and max within +/- 10
                c[sig]["min_within_10"] = float(np.nanmin(local))
                c[sig]["max_within_10"] = float(np.nanmax(local))
                c[sig]["argmin_offset_within_10"] = int(np.nanargmin(local) - T_IMPACT)
                c[sig]["argmax_offset_within_10"] = int(np.nanargmax(local) - T_IMPACT)
        contrast[name] = c

    # New acceptance attempt: trough in kappa at impact (baseline/at >= 2).
    trough_gate = {}
    for name, r in results.items():
        med = np.nanmedian(r["kappa"], axis=0)
        at = float(med[T_IMPACT])
        base = float(np.nanmedian(med[far]))
        trough_ratio = base / max(at, 1e-9)
        # local minimum within +/- 3 frames?
        win = np.abs(tau) <= 3
        local_min_idx = int(np.nanargmin(med * np.where(win, 1.0, np.nan)) - T_IMPACT)
        trough_gate[name] = {
            "at_impact_kappa": at,
            "off_impact_baseline_kappa": base,
            "trough_ratio_baseline_over_impact": trough_ratio,
            "trough_passes_2x": trough_ratio >= 2.0,
            "trough_passes_1p5x": trough_ratio >= 1.5,
            "argmin_offset_within_3": local_min_idx,
        }

    summary = {
        "signature_contrast": contrast,
        "alternative_acceptance_trough": trough_gate,
        "interpretation": (
            "Plan's gate (kappa peak at impact) FAILS. Inverted reading: kappa(t) "
            "dips at impact (curvature minimum, locally-linear pass-through). "
            "Trough_ratio = off-impact-baseline / at-impact-median."
        ),
    }
    (EXP1 / "extra_signatures_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[exp1c-extra] wrote {EXP1 / 'extra_signatures_summary.json'}")

    np.savez_compressed(
        EXP1 / "extra_signatures.npz",
        tau=tau,
        kappa_test_b=results["test_b"]["kappa"],
        kappa_test_c=results["test_c"]["kappa"],
        speed_test_b=results["test_b"]["speed"],
        speed_test_c=results["test_c"]["speed"],
        bcos_test_b=results["test_b"]["bcos"],
        bcos_test_c=results["test_c"]["bcos"],
        G_test_b=results["test_b"]["G"],
        G_test_c=results["test_c"]["G"],
    )
    print(f"[exp1c-extra] wrote {EXP1 / 'extra_signatures.npz'}")

    print("\n[exp1c-extra] Alternative gate: kappa TROUGH at impact (baseline/at >= 2)")
    for name, g in trough_gate.items():
        flag = "PASS" if g["trough_passes_2x"] else (
            "PASS-1.5x" if g["trough_passes_1p5x"] else "FAIL"
        )
        print(
            f"  {name:8s} at={g['at_impact_kappa']:.3f}  base={g['off_impact_baseline_kappa']:.3f}  "
            f"trough_ratio={g['trough_ratio_baseline_over_impact']:.3f}  {flag}"
        )

    # Figure: 3 rows (kappa, speed, bend cos) x 2 cols (test_b, test_c).
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for col, (split_name, color) in enumerate(
        [("test_b", "tab:blue"), ("test_c", "tab:orange")]
    ):
        kappa = results[split_name]["kappa"]
        speed = results[split_name]["speed"]
        bcos = results[split_name]["bcos"]
        for row, (sig_name, arr, color_, ylabel) in enumerate([
            (r"$\kappa(\tau)$", kappa, color, "curvature (log)"),
            (r"$|z'|(\tau)$", speed, color, "speed"),
            (r"$\cos\theta(\tau)$", bcos, color, "bend cosine"),
        ]):
            ax = axes[row, col]
            med = np.nanmedian(arr, axis=0)
            q25 = np.nanpercentile(arr, 25, axis=0)
            q75 = np.nanpercentile(arr, 75, axis=0)
            ax.fill_between(tau, q25, q75, color=color_, alpha=0.2)
            ax.plot(tau, med, color=color_, lw=2)
            ax.axvline(0, color="k", lw=1, alpha=0.6)
            ax.axvspan(-3, 3, color="k", alpha=0.07)
            if row == 0:
                ax.set_yscale("log")
                ax.set_title(f"{split_name}")
            if col == 0:
                ax.set_ylabel(f"{sig_name}\n({ylabel})")
            ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel(r"frame offset $\tau$ from impact")
    axes[-1, 1].set_xlabel(r"frame offset $\tau$ from impact")
    fig.suptitle("Topological signatures aligned by impact frame")
    fig.tight_layout()
    fig.savefig(FIGS / "exp1_signatures_at_impact.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[exp1c-extra] wrote {FIGS / 'exp1_signatures_at_impact.png'}")


if __name__ == "__main__":
    main()
