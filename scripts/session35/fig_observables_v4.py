"""Figure F9 (manuscript v4): the two 80-D observables, wake vs near-body lift-element.

Panel (a): one representative test_b impact-window vorticity frame with the
wake ROI box (Mode C patch grid, 8x4 over the wake ROI bbox) and the near-body
band contour (build_nearbody_band, delta_n = 0.3c) overlaid.
Panel (b): the 64-D signed patch energies of the WAKE observable for that frame
as an 8x4 image pair (positive / negative vorticity energy).
Panel (c): same for the NEAR-BODY lift-element observable, loaded from the
precomputed nearbody_observables cache (raw, unstandardized target).
Panel (d): the two 16-bin radial spectra as line plots.

Every number and field is read from caches at build time:
  * outputs/session34/trackc_latents/fields_test_b.npz (omega_norm, window_mask,
    target_C_L) for the frame and the wake observable (computed on the fly via
    src.data.wake_observables.patch_signed_spectrum_target, CPU);
  * ${VORTEX_JEPA_CACHE}/v2p2/nearbody_observables/{case}/encounter_{k:02d}.h5
    dataset 'nearbody_lift_element' for the near-body target;
  * outputs/data_pipeline/v2p2/airfoil_adjacent_mask.npy for the band geometry.

Frame rule: within the impact window (window_mask) of the representative
encounter, the frame of max |target_C_L| (same rule as the decode-panel figure).

CPU only. Run from the repo root:
    PREVENT_ROOT=$HOME/PREVENT .venv/bin/python scripts/session35/fig_observables_v4.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "session21"))

import figstyle  # noqa: E402
from src.data.lift_element import DEFAULT_ADJACENT_MASK_PATH  # noqa: E402
from src.data.nearbody_observables import get_nearbody_band, nearbody_roi_window  # noqa: E402
from src.data.wake_observables import (  # noqa: E402
    _wake_roi_window,
    patch_signed_spectrum_target,
)

import matplotlib.pyplot as plt  # noqa: E402

CASE = "G-0.50_D1.00_Y-0.40"
ENC = 0
FIELDS_NPZ = REPO / "outputs/session34/trackc_latents/fields_test_b.npz"
PREVENT = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
CACHE = Path(os.environ.get(
    "VORTEX_JEPA_CACHE", str(PREVENT / "data" / "processed" / "vortex-jepa")))
NEARBODY_H5 = CACHE / "v2p2" / "nearbody_observables" / CASE / f"encounter_{ENC:02d}.h5"
OUT_DIR = REPO / "paper" / "sections" / "figures" / "results"
DT_TC = 0.05

# Annotation palette, distinct from the RdBu_r field colours.
C_WAKE = "#762a83"      # purple: wake observable
C_NEAR = "#1b7837"      # green: near-body lift-element observable


def main() -> None:
    figstyle.use_style()

    # ---- representative frame (impact-window peak |C_L|) --------------------
    fields = np.load(FIELDS_NPZ, allow_pickle=True)
    sel = (fields["case_id"] == CASE) & (fields["encounter_index"] == ENC)
    if not sel.any():
        raise RuntimeError(f"case {CASE} enc {ENC} not found in {FIELDS_NPZ}")
    rows = np.where(sel)[0]
    order = np.argsort(fields["frame"][rows])
    rows = rows[order]
    frames = fields["frame"][rows]
    wmask = fields["window_mask"][rows]
    cl = fields["target_C_L"][rows]
    widx = np.where(wmask)[0]
    i_peak = widx[np.argmax(np.abs(cl[widx]))]
    frame = int(frames[i_peak])
    cl_peak = float(cl[i_peak])
    omega = fields["omega_norm"][rows[i_peak]].astype(np.float32)  # (192, 96)

    # ---- wake observable (Mode C, 80-D) computed from the cached field ------
    wake_t = patch_signed_spectrum_target(
        torch.from_numpy(omega[None])).numpy()[0]  # (80,)
    wake_pos = wake_t[:32].reshape(8, 4)
    wake_neg = wake_t[32:64].reshape(8, 4)
    wake_spec = wake_t[64:]

    # ---- near-body observable (80-D) from the precomputed cache -------------
    with h5py.File(NEARBODY_H5, "r") as f:
        assert f.attrs["case_id"] == CASE and int(f.attrs["encounter_index"]) == ENC
        near_t = f["nearbody_lift_element"][frame]  # (80,)
    near_pos = near_t[:32].reshape(8, 4)
    near_neg = near_t[32:64].reshape(8, 4)
    near_spec = near_t[64:]

    # ---- geometry ------------------------------------------------------------
    wr0, wr1, wc0, wc1 = _wake_roi_window()
    band = get_nearbody_band(REPO / DEFAULT_ADJACENT_MASK_PATH)
    nr0, nr1, nc0, nc1 = nearbody_roi_window(band)

    print(f"[fig9] case={CASE} enc={ENC} frame={frame} (t/c={frame * DT_TC:.2f}) "
          f"C_L={cl_peak:.3f}")
    print(f"[fig9] wake ROI bbox (r0,r1,c0,c1)={_wake_roi_window()}, "
          f"nearbody bbox={(nr0, nr1, nc0, nc1)}")
    print(f"[fig9] wake target: patch64 max={wake_t[:64].max():.3f}, "
          f"spec16 range=[{wake_spec.min():.3f}, {wake_spec.max():.3f}]")
    print(f"[fig9] nearbody target: patch64 max={near_t[:64].max():.3f}, "
          f"spec16 range=[{near_spec.min():.3f}, {near_spec.max():.3f}]")

    # ---- layout ---------------------------------------------------------------
    fig = plt.figure(figsize=(figstyle.TEXTWIDTH_IN, 3.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.0],
                          hspace=0.42, wspace=0.30,
                          left=0.05, right=0.97, top=0.92, bottom=0.145)

    # (a) field + overlays ------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    figstyle.vort_panel(ax_a, omega)
    # wake ROI box + internal 8x4 patch grid (Mode C convention)
    ax_a.add_patch(Rectangle((wr0 - 0.5, wc0 - 0.5), wr1 - wr0, wc1 - wc0,
                             fill=False, edgecolor=C_WAKE, lw=1.1,
                             linestyle="--", zorder=6))
    for i in range(1, 8):
        x = wr0 - 0.5 + i * (wr1 - wr0) / 8.0
        ax_a.plot([x, x], [wc0 - 0.5, wc1 - 0.5], color=C_WAKE, lw=0.35,
                  linestyle=":", alpha=0.8, zorder=6)
    for j in range(1, 4):
        y = wc0 - 0.5 + j * (wc1 - wc0) / 4.0
        ax_a.plot([wr0 - 0.5, wr1 - 0.5], [y, y], color=C_WAKE, lw=0.35,
                  linestyle=":", alpha=0.8, zorder=6)
    # near-body band contour + its patch bbox and grid
    ax_a.contour((band > 0).T.astype(float), levels=[0.5], colors=C_NEAR,
                 linewidths=1.0, zorder=7)
    ax_a.add_patch(Rectangle((nr0 - 0.5, nc0 - 0.5), nr1 - nr0, nc1 - nc0,
                             fill=False, edgecolor=C_NEAR, lw=0.7,
                             linestyle=":", zorder=7))
    ax_a.text(wr1 - 3.0, wc0 + 2.0, r"wake ROI, $8{\times}4$ patches",
              color=C_WAKE, fontsize=6.5, ha="right", va="bottom")
    ax_a.text(nr0 - 2.0, nc0 - 4.0, r"near-body band ($\delta_n = 0.3c$)",
              color=C_NEAR, fontsize=6.5, ha="left", va="top")
    ax_a.set_title(rf"(a)  normalised $\omega_z$, $t/c = {frame * DT_TC:.2f}$",
                   loc="left", fontsize=8)

    # (b), (c): 8x4 signed patch-energy pairs (pos over neg, equal aspect) -------
    def patch_pair(cell, pos: np.ndarray, neg: np.ndarray, color: str,
                   label: str) -> None:
        sub = cell.subgridspec(2, 1, hspace=0.18)
        vmax = float(max(pos.max(), neg.max(), 1e-6))
        for k, (arr, cmap, sgn) in enumerate(
                [(pos, "Reds", "+"), (neg, "Blues", "-")]):
            ax = fig.add_subplot(sub[k, 0])
            im = ax.imshow(arr.T, origin="lower", cmap=cmap, vmin=0.0,
                           vmax=vmax, aspect="equal", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_edgecolor(color)
                s.set_linewidth(0.9)
            ax.set_ylabel(rf"$e^{{{sgn}}}$", fontsize=7, labelpad=2.0,
                          rotation=0, va="center")
            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="4%", pad=0.05)
            cb = fig.colorbar(im, cax=cax)
            cb.set_ticks([0.0, vmax])
            cb.ax.set_yticklabels(["0", f"{vmax:.2f}"], fontsize=5.5)
            cb.outline.set_linewidth(0.4)
            if k == 0:
                ax.set_title(label, loc="left", fontsize=8)

    patch_pair(gs[0, 1], wake_pos, wake_neg, C_WAKE,
               "(b)  wake: signed patch energies")
    patch_pair(gs[1, 0], near_pos, near_neg, C_NEAR,
               "(c)  near-body: signed patch energies")

    # (d) the two 16-bin radial spectra ------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    bins = np.arange(16)
    ax_d.plot(bins, wake_spec, color=C_WAKE, marker="o", ms=2.5, lw=1.0,
              label="wake")
    ax_d.plot(bins, near_spec, color=C_NEAR, marker="s", ms=2.5, lw=1.0,
              label="near-body lift-element")
    ax_d.set_xlabel("radial wavenumber bin", labelpad=1.5)
    ax_d.set_ylabel(r"$\log(1 + P_k)$", labelpad=1.5)
    ax_d.set_xlim(-0.4, 15.4)
    ax_d.set_ylim(0.0, 1.18 * float(wake_spec.max()))
    ax_d.set_xticks([0, 4, 8, 12, 15])
    ax_d.legend(loc="upper right", fontsize=6.5, handlelength=1.6)
    ax_d.set_title("(d)  16-bin radial spectra", loc="left", fontsize=8)

    fig.text(0.5, 0.02,
             rf"each observable: $80\mathrm{{-D}} = 64$ signed patch energies "
             rf"($8{{\times}}4 \times \{{+,-\}}$, log1p) $+$ 16-bin radial "
             rf"spectrum;  case {CASE} (test_b), encounter {ENC}",
             ha="center", va="bottom", fontsize=6.2, color="#404040")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "fig_observables_v4.pdf"
    png = OUT_DIR / "fig_observables_v4.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    print(f"[fig9] wrote {pdf}")
    print(f"[fig9] wrote {png}")


if __name__ == "__main__":
    main()
