"""Scale-decomposition figure (fig_scale_decomp_v2p1.pdf): Gaussian scale
decomposition (sigma/c = 0.05) of the held-out reconstructions, v2.1.

Matches the v2.1 caption (fig:scale_decomp, section_4_results.tex):
  Top: large-scale vorticity for the simulation, the predictive decode, and the
       reconstructive autoencoder at H = 16 (impact + 16) on a representative
       test_b encounter; the predictive latent retains the leading-edge vortex and
       shear layer the reconstructive one smooths away.
  Bottom: large-scale wake enstrophy through the staged encounter for the
       simulation, predictive, reconstructive, and POD fields on test_b (mean over
       encounters, +/- 1 s.d. band).

The large-scale field and the wake-enstrophy time series are recomputed EXACTLY as
scripts/session28/s46_regen.py (physics_prep.large_scale_wake_enstrophy and
wake_mask, sigma/c = 0.05): the decoded omega window is mask+clip raw, and the DNS
truth is the v2p1 cache omega_z run through the same omega pipeline (mask + clip),
so the per-family mean correlations match the manuscript macros
(NumScaleDecodeCorr*). The triptych shows pipeline-normalised omega in the locked
Fig-3 convention (RdBu_r, vmin/vmax = -3/+3, black airfoil) via figstyle.vort_panel.

Representative-not-worst (CLAUDE.md): the triptych encounter is the one whose
predictive large-scale enstrophy correlation with the DNS is closest to the family
MEDIAN, a typical encounter rather than the best or worst.

CPU only; numpy/scipy + h5py for the cache. Reads the decode NPZs + the v2p1 cache;
writes only the PDF.

Output: paper/sections/figures/results/fig_scale_decomp_v2p1.pdf
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import _common as cm  # noqa: E402
import physics_prep as pp  # noqa: E402  (large_scale_wake_enstrophy / wake_mask)

fs = cm.fs

DECODE_ROOT = cm.SESSION28 / "decode"
OMEGA_MANIFEST = cm.REPO / "outputs" / "data_pipeline" / "v2p1" / "manifest.json"
# (series-key, decode-subdir, legend-label, figstyle family-colour key)
DECODE_FAMILIES = [
    ("jepa_tf_noc", "tf_noc", "predictive (JEPA)", "jepa"),
    ("fukami", "fukami", "reconstructive", "fukami"),
    ("pod", "pod", "POD", "pod"),
]
SPLIT = "test_b"
SIGMA_C = pp.SIGMA_C_DEFAULT
H16_OFFSET = 16


def cache_root() -> Path:
    base = os.environ.get(
        "VORTEX_JEPA_CACHE",
        str(
            Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
            / "data"
            / "processed"
            / "vortex-jepa"
        ),
    )
    return Path(base) / "v2p1"


def load_pipeline():
    from src.data.omega_pipeline import OmegaPipeline

    return OmegaPipeline.from_manifest(OMEGA_MANIFEST)


def dns_full(root: Path, cid: str, ei: int, pipe) -> np.ndarray:
    """Full DNS omega_z stack, pipeline-preprocessed (mask + clip, raw)."""
    import h5py

    with h5py.File(root / cid / f"encounter_{ei:02d}.h5", "r") as f:
        omega = f["omega_z"][:].astype(np.float64)
    return np.asarray(pipe.preprocess_raw(omega, cid, ei))


def main() -> None:
    fs.use_style()
    pipe = load_pipeline()
    mask = pp.wake_mask()
    sigma_px = SIGMA_C / pp.DX
    root = cache_root()
    norm_div = 3.0 * pipe.train_stats.std  # pipeline 3-sigma divisor for display

    # ---- per-family large-scale wake-enstrophy time series over the window ----
    series: dict[str, np.ndarray] = {}  # family -> (n_enc, n_frames) decoded LS enstrophy
    dns_series = None
    meta = None
    for fam, ddir, _, _famcol in DECODE_FAMILIES:
        blob = np.load(DECODE_ROOT / ddir / f"recon_{SPLIT}.npz", allow_pickle=True)
        decoded = blob["omega_decoded_window"].astype(np.float64)  # (n, 31, 192, 96)
        wframes = blob["window_frames"].astype(int)
        cids = np.array([str(c) for c in blob["case_ids"]])
        eis = blob["encounter_indices"].astype(int)
        e_dec = np.stack(
            [
                pp.large_scale_wake_enstrophy(decoded[i], sigma_px, mask)
                for i in range(decoded.shape[0])
            ]
        )
        series[fam] = e_dec
        if dns_series is None:
            dns_fields = [
                dns_full(root, cids[i], int(eis[i]), pipe) for i in range(decoded.shape[0])
            ]
            e_dns = np.stack(
                [
                    pp.large_scale_wake_enstrophy(dns_fields[i][wframes], sigma_px, mask)
                    for i in range(decoded.shape[0])
                ]
            )
            dns_series = e_dns
            meta = {
                "wframes": wframes,
                "cids": cids,
                "eis": eis,
                "decoded_jepa_h16": blob["omega_decoded_horizon"][
                    :, list(blob["horizon_offsets"]).index(H16_OFFSET)
                ].astype(np.float64),
                "h16_abs": int(
                    blob["horizon_abs_frames"][0][list(blob["horizon_offsets"]).index(H16_OFFSET)]
                ),
                "dns_fields": dns_fields,
            }

    wframes = meta["wframes"]
    impact = 40
    trel = wframes - impact

    # representative encounter for the triptych: predictive corr closest to median.
    jepa = series["jepa_tf_noc"]
    corr = np.array(
        [
            (
                np.corrcoef(jepa[i], dns_series[i])[0, 1]
                if jepa[i].std() > 1e-9 and dns_series[i].std() > 1e-9
                else np.nan
            )
            for i in range(jepa.shape[0])
        ]
    )
    finite = np.flatnonzero(np.isfinite(corr))
    rep = int(finite[np.argmin(np.abs(corr[finite] - np.nanmedian(corr)))])

    # ---- figure ----
    from scipy.ndimage import gaussian_filter

    fig = plt.figure(figsize=fs.figure_size(1.0, aspect=0.62))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25], hspace=0.35, wspace=0.08)

    # top triptych: large-scale (Gaussian-filtered) normalised omega at H=16
    # (impact + 16 = frame 56, which is in the horizon array, not the [25,55] window).
    h16_abs = meta["h16_abs"]
    fuk_h = np.load(DECODE_ROOT / "fukami" / f"recon_{SPLIT}.npz", allow_pickle=True)
    fuk_h16 = fuk_h["omega_decoded_horizon"][
        rep, list(fuk_h["horizon_offsets"]).index(H16_OFFSET)
    ].astype(np.float64)
    panels = [
        ("simulation", meta["dns_fields"][rep][h16_abs]),
        ("predictive decode", meta["decoded_jepa_h16"][rep]),
        ("reconstructive decode", fuk_h16),
    ]
    for j, (title, raw) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        large = gaussian_filter(raw, sigma=sigma_px) / norm_div
        fs.vort_panel(ax, large)
        ax.set_title(title, fontsize=7.5)

    axc = fig.add_subplot(gs[1, :])
    curves = [("dns", "oracle", "simulation", dns_series)]
    curves += [(fam, famcol, lab, series[fam]) for fam, _, lab, famcol in DECODE_FAMILIES]
    for _, famkey, lab, arr in curves:
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        col = fs.FAMILY_COLOR[famkey]
        lw = 1.6 if lab == "simulation" else 1.0
        axc.plot(
            trel,
            mean,
            color=col,
            lw=lw,
            marker=None if lab == "simulation" else "o",
            ms=2.5,
            label=lab,
            zorder=5 if lab == "simulation" else 3,
        )
        axc.fill_between(trel, mean - std, mean + std, color=col, alpha=0.10, lw=0)
    axc.axvline(0, color="0.85", lw=0.6, zorder=0)
    axc.set_xlabel("frames relative to impact")
    axc.set_ylabel(r"large-scale wake enstrophy $\Omega_w$")
    axc.legend(loc="upper left", ncol=2, columnspacing=1.2, handletextpad=0.4, borderpad=0.2)
    axc.margins(x=0.02)

    out = cm.out_path("fig_scale_decomp")
    fig.savefig(out)
    print(f"wrote {out}")
    print(
        f"  representative encounter {meta['cids'][rep]} enc {meta['eis'][rep]} "
        f"(predictive LS corr {corr[rep]:.2f}, median {np.nanmedian(corr):.2f})"
    )
    for fam, _, _, _famcol in DECODE_FAMILIES:
        c = np.array(
            [
                np.corrcoef(series[fam][i], dns_series[i])[0, 1]
                for i in range(series[fam].shape[0])
                if series[fam][i].std() > 1e-9 and dns_series[i].std() > 1e-9
            ]
        )
        print(f"  {fam}: mean LS-enstrophy corr {np.nanmean(c):+.2f}")


if __name__ == "__main__":
    main()
