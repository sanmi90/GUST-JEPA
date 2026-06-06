#!/usr/bin/env python3
"""Motivation animation (no JEPA): a gust encounter is a hard, fast problem.

DNS mid-plane vorticity (top) with the lift coefficient C_L evolving below, for a
strong illustrative gust. Shows the incoming vortex, the leading-edge-vortex
build-up, and the large, fast lift transient, to motivate why gust interaction is
a complex modelling problem. Pure DNS; no encoder/predictor involved.
Output: figs/gust_motivation_anim.mp4 (+ poster).
"""
import sys
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, FFMpegWriter  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import figstyle as fs  # noqa: E402
from src.data.omega_pipeline import OmegaPipeline  # noqa: E402

CACHE = REPO.parent / "PREVENT/data/processed/vortex-jepa/v2"
PIPE = REPO / "outputs/data_pipeline/v1/manifest.json"
OUT_MP4 = Path(__file__).resolve().parent / "figs" / "gust_motivation_anim.mp4"
OUT_POS = Path(__file__).resolve().parent / "figs" / "gust_motivation_poster.png"
NAVY = "#1F3864"; INK = "#242933"; ACCENT = "#C0520F"
VLIM = 2.0; DT = 0.05; LO, HI = -12, 46     # frames relative to impact


def pick_case():
    """A moderate-G gust from the TRAIN set (not extrapolation) with a clear lift swing."""
    import json
    split = json.load(open(REPO / "configs/splits/split_v2.json"))["cases"]
    train = {k for k, c in split.items() if c["split"] == "train"}
    best = (None, -1.0)
    for d in sorted(CACHE.glob("G*_D*_Y*")):
        if d.name not in train:
            continue
        try:
            G = float(d.name.split("_")[0][1:])
        except ValueError:
            continue
        if not (1.5 <= abs(G) <= 2.0):   # moderate gust, in the training envelope
            continue
        f = d / "encounter_00.h5"
        if not f.exists():
            continue
        with h5py.File(f, "r") as h:
            cl = np.asarray(h["C_L"], float)
            imp = int(h.attrs.get("impact_frame_estimate", 40))
        w = cl[max(0, imp - 10):imp + 35]
        swing = float(np.nanmax(w) - np.nanmin(w))
        if swing > best[1]:
            best = (d.name, swing)
    return best[0]


def main():
    case = pick_case()
    pipe = OmegaPipeline.from_manifest(PIPE)
    with h5py.File(CACHE / case / "encounter_00.h5", "r") as h:
        omega_raw = np.asarray(h["omega_z"], np.float32)
        cl = np.asarray(h["C_L"], float)
        imp = int(h.attrs.get("impact_frame_estimate", 40))
    omega = pipe.normalize(pipe.preprocess_raw(omega_raw, case, 0))  # (120,192,96)
    G = float(case.split("_")[0][1:]); D = float(case.split("_")[1][1:]); Y = float(case.split("_")[2][1:])
    fr = np.clip(np.arange(LO, HI + 1) + imp, 0, omega.shape[0] - 1)
    tc = (fr - imp) * DT
    print(f"motivation case {case} (G={G:+.1f},D={D:.1f},Y={Y:+.1f}) impact={imp}  "
          f"C_L swing={np.nanmax(cl[fr]) - np.nanmin(cl[fr]):.2f}")

    plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(7.6, 5.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.55, 1.0], hspace=0.32,
                          left=0.12, right=0.97, top=0.88, bottom=0.12)
    axf = fig.add_subplot(gs[0]); axl = fig.add_subplot(gs[1])
    im = fs.vort_panel(axf, omega[fr[0]], vlim=VLIM)

    axl.plot(tc, cl[fr], color=INK, lw=2.2)
    axl.axvline(0, color=ACCENT, lw=1.2, ls=":")
    axl.text(0, axl.get_ylim()[1], " gust impact", color=ACCENT, fontsize=10, va="top", ha="left")
    cur = axl.axvline(tc[0], color="0.45", lw=1.2)
    dot, = axl.plot([tc[0]], [cl[fr[0]]], "o", color=ACCENT, ms=7, zorder=6)
    axl.set_xlabel("t/c relative to impact"); axl.set_ylabel(r"lift  $C_L$")
    axl.set_xlim(tc[0], tc[-1]); axl.grid(axis="y", color="0.92")
    for s in ("top", "right"):
        axl.spines[s].set_visible(False)
    ttl = fig.text(0.5, 0.955, "", ha="center", fontsize=13, color=NAVY, fontweight="bold")

    def update(k):
        im.set_array(omega[fr[k]].T)
        cur.set_xdata([tc[k], tc[k]]); dot.set_data([tc[k]], [cl[fr[k]]])
        ttl.set_text(f"Gust encounter (G={G:+.1f}, D={D:.1f}, Y={Y:+.1f}):   t/c = {tc[k]:+.2f}")
        return [im, cur, dot]

    anim = FuncAnimation(fig, update, frames=len(fr), interval=110, blit=False)
    anim.save(str(OUT_MP4), writer=FFMpegWriter(fps=9, bitrate=3000), dpi=130)
    update(int(np.where(fr == np.clip(imp + 12, 0, 119))[0][0]))
    fig.savefig(str(OUT_POS), dpi=130)
    print("wrote", OUT_MP4, "and", OUT_POS)


if __name__ == "__main__":
    main()
