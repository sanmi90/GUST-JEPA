#!/usr/bin/env python3
"""T1 (paper_redesign.md section 6): per-latent-coordinate spectral content.

Per-coordinate Welch PSD over the training encounters (undisturbed + gusted),
per family at d = 32: POD, predictive CLW (jepa_pool_vec), AE-LW (ae_wake_pool),
published-recipe reconstruction (fukami). Metric: spectral flatness (Wiener
entropy, geometric/arithmetic mean of the PSD) per coordinate; family median
with a CASE-CLUSTERED bootstrap CI on the difference.

GATE (D-E): the introduction's "broadband-mixing" wording for the
published-recipe latent is supported only if the published-recipe median
flatness EXCEEDS the predictive latent's with a case-clustered CI on the
difference (Fukami - JepaWake) that excludes zero. Otherwise the fallback
wording (clock + divergence evidence only) stands.

CPU-only (cached latents). Run:
    OMP_NUM_THREADS=4 taskset -c 0-15 .venv/bin/python \\
        scripts/session39/t1_spectral_flatness.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import welch
from scipy.stats import gmean

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "outputs/session34/trackc_latents"
OUT_JSON = REPO / "outputs/session39/t1_spectral_flatness.json"
OUT_FIG = REPO / "paper/sections/figures/results/fig_t1_spectra_v4.pdf"

FAMILIES = {  # paper label -> latent cache run (d = 32, matched)
    "Pod": "pod",
    "JepaWake": "jepa_pool_vec",      # predictive CLW
    "AeWake": "ae_wake_pool",         # AE-LW (matched-supervision)
    "Fukami": "fukami",               # published-recipe reconstruction
}
DT_TC = 0.05                          # cache cadence; fs = 1/dt in cycles per t/c
FS = 1.0 / DT_TC                      # = 20 ; frequency axis IS the Strouhal number
NPERSEG = 64
ST_SHED = 0.675                       # measured shedding Strouhal (DMD)
N_BOOT = 2000
SEED = 0


def load_family(run: str):
    z = np.load(CACHE / f"latents_{run}_train.npz", allow_pickle=True)
    Z = z["z_gap"].astype(np.float64)
    cid = np.asarray([str(c) for c in z["case_id"]])
    enc = np.asarray(z["encounter_index"])
    fr = np.asarray(z["frame"])
    # group into per-encounter (T, d) trajectories, sorted by frame
    keys = list(zip(cid.tolist(), enc.tolist()))
    order: dict = {}
    for i, k in enumerate(keys):
        order.setdefault(k, []).append(i)
    trajs, tcase, tbaseline = [], [], []
    for (case, _e), idx in order.items():
        idx = np.asarray(idx)[np.argsort(fr[idx])]
        trajs.append(Z[idx])                     # (T, d)
        tcase.append(case)
        tbaseline.append(("aseline" in case) or ("G+0.00" in case))
    return trajs, np.asarray(tcase), np.asarray(tbaseline)


def flatness_of(x: np.ndarray) -> float:
    """Wiener entropy of a coordinate time series (DC bin excluded)."""
    x = x - x.mean()
    if np.allclose(x, 0.0):
        return np.nan
    f, p = welch(x, fs=FS, nperseg=min(NPERSEG, len(x)), detrend="constant")
    p = p[1:]                                    # drop DC
    p = p[p > 0]
    if p.size < 4:
        return np.nan
    return float(gmean(p) / p.mean())


def mean_psd(trajs) -> tuple[np.ndarray, np.ndarray]:
    """Mean PSD over encounters, per coordinate: (d, n_freq)."""
    acc = None
    f_ref = None
    n = 0
    for x in trajs:                              # x: (T, d)
        per = []
        for j in range(x.shape[1]):
            xi = x[:, j] - x[:, j].mean()
            f, p = welch(xi, fs=FS, nperseg=min(NPERSEG, len(xi)), detrend="constant")
            per.append(p)
            f_ref = f
        acc = np.asarray(per) if acc is None else acc + np.asarray(per)
        n += 1
    return f_ref, acc / n


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    per_family = {}
    fam_psd = {}
    fam_freq = None
    for label, run in FAMILIES.items():
        trajs, tcase, tbase = load_family(run)
        # per-(encounter, coordinate) flatness
        flat_ec, case_ec = [], []
        for x, case in zip(trajs, tcase):
            for j in range(x.shape[1]):
                v = flatness_of(x[:, j])
                if not np.isnan(v):
                    flat_ec.append(v)
                    case_ec.append(case)
        flat_ec = np.asarray(flat_ec)
        case_ec = np.asarray(case_ec)
        per_family[label] = {"flat": flat_ec, "case": case_ec,
                             "median": float(np.median(flat_ec))}
        f_ref, psd = mean_psd(trajs)
        fam_freq = f_ref
        fam_psd[label] = psd
        print(f"{label:10s} median flatness = {np.median(flat_ec):.4f} "
              f"(n={flat_ec.size} coord-encounters, {len(set(case_ec))} cases)")

    # case-clustered bootstrap on the difference Fukami - JepaWake
    def median_over_cases(flat, case, cases):
        m = np.isin(case, cases)
        return np.median(flat[m]) if m.any() else np.nan

    a, b = per_family["Fukami"], per_family["JepaWake"]
    all_cases = np.array(sorted(set(a["case"]) | set(b["case"])))
    diffs = []
    for _ in range(N_BOOT):
        samp = rng.choice(all_cases, size=all_cases.size, replace=True)
        da = median_over_cases(a["flat"], a["case"], samp)
        db = median_over_cases(b["flat"], b["case"], samp)
        if not (np.isnan(da) or np.isnan(db)):
            diffs.append(da - db)
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    point = a["median"] - b["median"]
    gate = bool(lo > 0.0)                        # Fukami strictly more broadband

    result = {
        "_provenance": {"script": "scripts/session39/t1_spectral_flatness.py",
                        "cache": str(CACHE.relative_to(REPO)),
                        "families": FAMILIES, "fs_cycles_per_tc": FS,
                        "nperseg": NPERSEG, "n_boot": N_BOOT},
        "median_flatness": {k: v["median"] for k, v in per_family.items()},
        "fukami_minus_jepawake": {"point": point, "ci_lo": float(lo),
                                  "ci_hi": float(hi)},
        "gate_broadband_supported": gate,
        "interpretation": ("Fukami more broadband than the predictive latent, "
                           "CI excludes zero: broadband wording SUPPORTED"
                           if gate else
                           "difference CI includes zero or wrong sign: keep the "
                           "clock+divergence fallback wording (D-E)"),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"\nGATE broadband supported: {gate}")
    print(f"  Fukami - JepaWake median flatness = {point:.4f} "
          f"[{lo:.4f}, {hi:.4f}]")
    print(f"wrote {OUT_JSON}")

    # ---- figure: coordinate-by-Strouhal mean-PSD heatmaps, shared scale -------
    try:
        import matplotlib.pyplot as plt
        import sys
        sys.path.insert(0, str(REPO))
        from scripts.session21.figstyle import TEXTWIDTH_IN, use_style
        use_style()
        fmask = fam_freq <= 3.0                  # show St in [0, 3]
        vmax = max(np.log10(p[:, fmask] + 1e-12).max() for p in fam_psd.values())
        vmin = vmax - 4.0
        fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH_IN, 0.72 * TEXTWIDTH_IN),
                                 sharex=True, sharey=True)
        for ax, (label, psd) in zip(axes.ravel(), fam_psd.items()):
            P = np.log10(psd[:, fmask] + 1e-12)
            order = np.argsort(np.argmax(psd[:, fmask], axis=1))  # by peak freq
            im = ax.pcolormesh(fam_freq[fmask], np.arange(P.shape[0]),
                               P[order], vmin=vmin, vmax=vmax,
                               cmap="magma", shading="auto")
            ax.axvline(ST_SHED, color="c", lw=0.7, ls=(0, (3, 2)))
            ax.set_title(f"{label} (flatness {per_family[label]['median']:.2f})",
                         fontsize=7)
        for ax in axes[-1]:
            ax.set_xlabel(r"Strouhal number $St$")
        for ax in axes[:, 0]:
            ax.set_ylabel("latent coordinate")
        fig.colorbar(im, ax=axes, shrink=0.8, label=r"$\log_{10}$ PSD")
        fig.savefig(OUT_FIG, bbox_inches="tight")
        print(f"wrote {OUT_FIG}")
    except Exception as e:  # noqa: BLE001
        print(f"[t1] figure skipped: {e}")


if __name__ == "__main__":
    main()
