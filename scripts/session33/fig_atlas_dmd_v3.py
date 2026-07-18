"""F9 (v3 paper): physics of the latent state.

Top row: parametric atlas of the pooled JEPA latent state, PC1-PC2 projections of
the 3-PC PCA coords (outputs/session33/manifold_atlas_v2p2_coords.npz), coloured
by gust intensity G (diverging), gust size D (sequential), and phase within the
shedding cycle (cyclic; the npz stores frames since impact, converted with the
measured dominant Strouhal number).

Bottom row: latent DMD spectrum on the undisturbed baseline case
(outputs/session33/spectrum_dmd_v2p2.json), |lambda| vs St per family with the
unit-modulus line and the measured shedding frequency and subharmonic, plus the
atlas PCA explained-variance bars per family.

All plotted values come from the two data files; nothing is hand-typed.
Output: outputs/session33/figures/fig_atlas_dmd_v3.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session21"))
from figstyle import (  # noqa: E402
    FAMILY_COLOR, FAMILY_LABEL, FAMILY_MARKER, TEXTWIDTH_IN, use_style,
)

COORDS_NPZ = REPO / "outputs/session33/manifold_atlas_v2p2_coords.npz"
DMD_JSON = REPO / "outputs/session33/spectrum_dmd_v2p2.json"
OUT_PDF = REPO / "outputs/session33/figures/fig_atlas_dmd_v3.pdf"

ATLAS_FAMILY = "jepa_pool"          # the headline pooled JEPA state
DOWNSAMPLE = 3                       # every 3rd frame keeps the pdf small
DMD_FAMILIES = {"jepa_pool": "jepa", "fukami": "fukami", "pod": "pod"}
EVR_FAMILIES = {"jepa_pool": "jepa", "fukami": "fukami", "pod": "pod"}
ST_XMAX = 1.05                       # damped high-St modes are counted, not drawn
# Panel (d) legend: canonical family names for the DA/ownstack figure set
# (local override; FAMILY_LABEL in figstyle stays untouched for other figures).
DMD_LEGEND_LABEL = {"jepa": "predictive (wake)", "fukami": "reconstructive (Fukami)",
                    "pod": "linear (POD)"}


def load_atlas() -> dict:
    d = np.load(COORDS_NPZ, allow_pickle=True)
    out = {
        "coords": d[f"{ATLAS_FAMILY}_coords"][::DOWNSAMPLE],
        "G": d[f"{ATLAS_FAMILY}_G"][::DOWNSAMPLE],
        "D": d[f"{ATLAS_FAMILY}_D"][::DOWNSAMPLE],
        "phase": d[f"{ATLAS_FAMILY}_phase"][::DOWNSAMPLE].astype(np.float64),
        "evr": {fam: d[f"{fam}_explained_variance_ratio"] for fam in EVR_FAMILIES},
    }
    return out


def load_dmd() -> dict:
    d = json.loads(DMD_JSON.read_text())
    gt = d["ground_truth"]
    dt = float(d["params"]["dt_tc"])
    fams = {}
    for fam in DMD_FAMILIES:
        b = d["models"][fam]["baseline"]
        # one point per conjugate pair: keep Im(lambda) >= 0
        modes = [m for m in b["spectrum"] if m["im"] >= 0.0]
        fams[fam] = {
            "St": np.array([m["St"] for m in modes]),
            "mod": np.array([m["modulus"] for m in modes]),
            "dominant": b["dominant"],
        }
    return {"families": fams, "St_ref": float(gt["St_dominant"]),
            "St_sub": float(gt["St_subharmonic"]), "dt_tc": dt}


def atlas_panel(ax, coords, values, cmap, label, title, sym=False, vmax_cyc=None,
                order=None):
    if sym:
        v = float(np.max(np.abs(values)))
        kw = {"vmin": -v, "vmax": v}
    elif vmax_cyc is not None:
        kw = {"vmin": 0.0, "vmax": vmax_cyc}
    else:
        kw = {"vmin": float(values.min()), "vmax": float(values.max())}
    if order is None:
        order = np.arange(len(values))
    sc = ax.scatter(coords[order, 0], coords[order, 1], c=values[order], cmap=cmap,
                    s=1.2, alpha=0.45, linewidths=0, rasterized=True, **kw)
    ax.set_title(title, fontsize=7.5)
    cb = plt.colorbar(sc, ax=ax, fraction=0.052, pad=0.03)
    cb.ax.tick_params(labelsize=6)
    cb.set_label(label, fontsize=6.5)
    cb.solids.set_alpha(1.0)
    return sc


def dmd_panel(ax, dmd: dict) -> None:
    ax.axhline(1.0, color="0.4", ls=(0, (4, 3)), lw=0.8, zorder=1)
    ax.axvline(dmd["St_ref"], color="0.75", ls="-", lw=0.8, zorder=1)
    ax.axvline(dmd["St_sub"], color="0.75", ls=":", lw=0.8, zorder=1)
    ax.text(dmd["St_ref"] + 0.015, 0.56,
            rf"$St = {dmd['St_ref']:.3f}$" + "\n(measured shedding)",
            ha="left", va="bottom", fontsize=6, color="0.35")
    ax.text(dmd["St_sub"] + 0.012, 0.345, r"$St/2$",
            ha="left", va="bottom", fontsize=6, color="0.35")
    n_clip, clip_max = 0, 0.0
    for fam, key in DMD_FAMILIES.items():
        f = dmd["families"][fam]
        show = f["St"] <= ST_XMAX
        n_clip += int((~show).sum())
        if (~show).any():
            clip_max = max(clip_max, float(f["mod"][~show].max()))
        ax.scatter(f["St"][show], f["mod"][show], s=13, marker=FAMILY_MARKER[key],
                   color=FAMILY_COLOR[key], alpha=0.75, linewidths=0, zorder=3,
                   label=DMD_LEGEND_LABEL.get(key, FAMILY_LABEL[key]))
        dom = f["dominant"]
        ax.scatter([dom["St"]], [dom["modulus"]], s=52, marker=FAMILY_MARKER[key],
                   facecolor=FAMILY_COLOR[key], edgecolor="black", linewidths=0.8,
                   zorder=4)
    ax.text(ST_XMAX - 0.01, 0.32,
            f"+{n_clip} damped modes at\n" + rf"$St > {ST_XMAX}$"
            + rf" (all $|\lambda| \leq {clip_max:.2f}$)",
            ha="right", va="bottom", fontsize=6, color="0.35")
    ax.scatter([], [], s=52, marker="o", facecolor="white", edgecolor="black",
               linewidths=0.8, label="dominant latent DMD mode")
    ax.set_xlim(-0.03, ST_XMAX)
    ax.set_ylim(0.0, 1.07)
    ax.set_xlabel(r"Strouhal number $St$")
    ax.set_ylabel(r"mode modulus $|\lambda|$")
    ax.set_title("(d) latent DMD spectrum, undisturbed case", fontsize=7.5)
    ax.legend(loc="lower left", bbox_to_anchor=(0.065, 0.015), fontsize=6,
              handletextpad=0.4, borderaxespad=0.0, labelspacing=0.35)


def evr_panel(ax, evr: dict) -> None:
    n_pc = len(next(iter(evr.values())))
    width = 0.26
    for i, (fam, key) in enumerate(EVR_FAMILIES.items()):
        xs = np.arange(n_pc) + (i - 1) * width
        ax.bar(xs, evr[fam], width=width, color=FAMILY_COLOR[key], alpha=0.9)
    ax.set_xticks(np.arange(n_pc))
    ax.set_xticklabels([f"PC{i + 1}" for i in range(n_pc)])
    ax.set_ylabel("explained variance ratio")
    ax.set_title("(e) atlas PCA variance", fontsize=7.5)


def main() -> int:
    use_style()
    atlas = load_atlas()
    dmd = load_dmd()

    # phase in the npz is (frame - t_impact) in frames; convert to phase within
    # the shedding cycle using the measured dominant Strouhal number.
    cyc = np.mod(atlas["phase"] * dmd["dt_tc"] * dmd["St_ref"], 1.0)

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 3.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05],
                          hspace=0.52, wspace=0.55,
                          left=0.075, right=0.965, top=0.93, bottom=0.10)
    ax_g = fig.add_subplot(gs[0, 0])
    ax_d = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[0, 2])
    ax_dmd = fig.add_subplot(gs[1, :2])
    ax_evr = fig.add_subplot(gs[1, 2])

    evr0 = atlas["evr"][ATLAS_FAMILY]
    for ax in (ax_g, ax_d, ax_p):
        ax.set_xlabel(f"PC1 ({100 * evr0[0]:.0f}%)")
    ax_g.set_ylabel(f"PC2 ({100 * evr0[1]:.0f}%)")
    # draw order: strong |G| and large D on top for legibility; phase shuffled
    # (deterministic) so no case systematically overplots the others.
    atlas_panel(ax_g, atlas["coords"], atlas["G"], "coolwarm", r"$G$",
                r"(a) gust intensity $G$", sym=True,
                order=np.argsort(np.abs(atlas["G"]), kind="stable"))
    atlas_panel(ax_d, atlas["coords"], atlas["D"], "viridis", r"$D$",
                r"(b) gust size $D$",
                order=np.argsort(atlas["D"], kind="stable"))
    atlas_panel(ax_p, atlas["coords"], cyc, "twilight", "cycle fraction",
                "(c) shedding phase", vmax_cyc=1.0,
                order=np.random.default_rng(0).permutation(len(cyc)))
    for ax in (ax_d, ax_p):
        ax.set_yticklabels([])

    dmd_panel(ax_dmd, dmd)
    evr_panel(ax_evr, atlas["evr"])

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    plt.close(fig)
    print(f"wrote {OUT_PDF}")
    print(f"  atlas: {ATLAS_FAMILY}, {atlas['coords'].shape[0]} pts "
          f"(every {DOWNSAMPLE}rd frame), EVR PC1/PC2 = {evr0[0]:.3f}/{evr0[1]:.3f}")
    for fam in DMD_FAMILIES:
        dom = dmd["families"][fam]["dominant"]
        print(f"  {fam}: dominant St={dom['St']:.4f} |lambda|={dom['modulus']:.4f}")
    print(f"  St refs: {dmd['St_ref']} (shedding), {dmd['St_sub']} (subharmonic)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
