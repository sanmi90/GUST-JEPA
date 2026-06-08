#!/usr/bin/env python3
"""Illustrated 'Fukami-style' schematics for the talk (TEST / alternative).

Two figures in the polished autoencoder-illustration idiom (feature-map stacks +
MLP neuron layers + real vorticity fields at the ends):
  figs/fukami_ae.png    : reconstructive autoencoder (encoder CNN -> MLP/latent +
                          lift head -> decoder CNN -> reconstructed field)
  figs/fukami_jepa.png  : JEPA twin in the same style (encoder -> latent ->
                          predictor -> next latent; loss in latent space, NO decoder)

These are alternatives to the native box diagrams (slides 6 & 8), which are kept.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figs"
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import figstyle as fs  # noqa: E402
VLIM = 2.0                      # tighter range = more saturated vorticity
AP = fs.airfoil_pixels()        # NACA 0012 polygon in (192 x 96) pixel coords

NAVY = "#1F3864"; GREEN = "#2E7D4F"; RED = "#B03A2E"; ORANGE = "#C0520F"
GRAY = "#8A93A3"; CARD = "#E9ECF2"; CARDE = "#9aa3b2"; BLUE = "#3C6FB0"
GREENN = "#5FA877"; LGREEN = "#BFE0C9"

dec = np.load(REPO / "outputs/session20/decoded/test_b.npz", allow_pickle=True)
cid = np.array([str(c) for c in dec["case_ids"]]); G = np.asarray(dec["G"], float)
gi = int(np.argmax(np.abs(G))); i0 = list(dec["offsets"]).index(0)
F_IN = dec["target_norm"][gi, i0]
F_AE = dec["fukami_norm"][gi, i0]


def stack(ax, x, yc, w, h, n=4, dx=0.8, dy=1.0, fc=CARD, ec=CARDE):
    for i in range(n - 1, -1, -1):
        ax.add_patch(FancyBboxPatch((x + i * dx, yc - h / 2 + i * dy), w, h,
                     boxstyle="round,pad=0.1,rounding_size=0.7", fc=fc, ec=ec,
                     lw=1.0, zorder=3 + (n - i)))


def neurons(ax, x, ys, color, ec=NAVY, s=150):
    ax.scatter([x] * len(ys), ys, s=s, c=color, edgecolors=ec, linewidths=0.7, zorder=9)
    return [(x, y) for y in ys]


def connect(ax, c1, c2, color="#CBD3E0", lw=0.4):
    for (x1, y1) in c1:
        for (x2, y2) in c2:
            ax.plot([x1, x2], [y1, y2], color=color, lw=lw, zorder=6)


def arrow(ax, x1, x2, y, color=GRAY, lw=2.4):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))


def field(ax, x, y, w, h, F):
    for k in (2, 1):
        ax.add_patch(Rectangle((x + k * 1.1, y + k * 1.3), w, h, fc="white",
                     ec="#c2c2c2", lw=0.8, zorder=2))
    ax.imshow(F.T, origin="lower", cmap=fs.VORT_CMAP, vmin=-VLIM, vmax=VLIM,
              extent=(x, x + w, y, y + h), aspect="auto", zorder=3, interpolation="nearest")
    poly = np.column_stack([x + AP[:, 0] / 192.0 * w, y + AP[:, 1] / 96.0 * h])
    ax.add_patch(Polygon(poly, closed=True, facecolor="black", edgecolor="black",
                 lw=0.4, zorder=4))
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec="#777", lw=0.9, zorder=5))


def bracket(ax, x0, x1, y, label, color="#555"):
    ax.plot([x0, x0, x1, x1], [y - 0.8, y, y, y - 0.8], color=color, lw=1.1, zorder=5)
    ax.text((x0 + x1) / 2, y + 0.6, label, ha="center", va="bottom", fontsize=12.5,
            color=NAVY)


def new_ax(h=3.7):
    fig, ax = plt.subplots(figsize=(13, h))
    ax.set_xlim(0, 200); ax.set_ylim(-4, 47); ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    return fig, ax


# ============================ AE (Fukami style) ============================
fig, ax = new_ax(3.7)
field(ax, 2, 16, 22, 11, F_IN); ax.text(13, 13.5, "input  ω", ha="center", fontsize=11)
arrow(ax, 26, 31, 21.5)
stack(ax, 33, 22, 11, 17, 4); stack(ax, 46, 22, 9, 13, 4); stack(ax, 56, 22, 7, 9, 4)
ax.text(46, 9.5, "CNN", ha="center", fontsize=11, color=NAVY)
ax.text(68, 26, "reshape", ha="center", fontsize=9, color=GRAY, rotation=0)
arrow(ax, 64, 71, 21.5)
c1 = neurons(ax, 75, np.linspace(11, 33, 7), BLUE)
c2 = neurons(ax, 83, np.linspace(14, 30, 5), BLUE)
cb = neurons(ax, 91, np.linspace(18, 26, 3), RED, ec=RED)
c3 = neurons(ax, 99, np.linspace(14, 30, 5), BLUE)
c4 = neurons(ax, 107, np.linspace(11, 33, 7), BLUE)
connect(ax, c1, c2); connect(ax, c2, cb); connect(ax, cb, c3); connect(ax, c3, c4)
ax.text(91, 34.5, "MLP", ha="center", fontsize=11, color=NAVY)
ax.text(91, 15.5, "latent  ξ", ha="center", fontsize=10.5, color=RED)
# lift head (ξ -> C_L), below the bottleneck
ax.annotate("", xy=(90, 8.5), xytext=(91, 17.5), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4))
lh1 = neurons(ax, 88, [3.5, 6, 8.5], RED, ec=RED, s=70)
lh2 = neurons(ax, 94, [4.8, 7.2], GREENN, ec=GREEN, s=70)
lh3 = neurons(ax, 99, [6], ORANGE, ec=ORANGE, s=90)
connect(ax, lh1, lh2, color="#E3C9C4", lw=0.5); connect(ax, lh2, lh3, color="#E3C9C4", lw=0.5)
ax.text(103.5, 6, "C_L", ha="left", va="center", fontsize=10.5, color=ORANGE)
ax.text(93.5, 1.2, "lift head", ha="center", fontsize=9.5, color=ORANGE)
arrow(ax, 114, 119, 21.5)
stack(ax, 121, 22, 7, 9, 4); stack(ax, 130, 22, 9, 13, 4); stack(ax, 141, 22, 11, 17, 4)
ax.text(133, 9.5, "CNN", ha="center", fontsize=11, color=NAVY)
arrow(ax, 156, 161, 21.5)
field(ax, 163, 16, 22, 11, F_AE); ax.text(174, 13.5, "reconstructed  ω", ha="center", fontsize=11)
bracket(ax, 31, 89, 40, "Encoder"); bracket(ax, 93, 155, 40, "Decoder")
ax.text(100, 45.5, "Autoencoder: reconstruct the field (loss in pixel space)",
        ha="center", fontsize=13.5, color=RED, fontweight="bold")
ax.text(100, -2.2, r"main loss:  $\mathcal{L}_{\mathrm{AE}}=\langle\|\hat\omega-\omega\|^{2}\rangle$   (field reconstruction, pixel space)",
        ha="center", fontsize=13, color=RED)
fig.savefig(OUT / "fukami_ae.png", dpi=200)
plt.close(fig)
print("wrote", OUT / "fukami_ae.png")


# ============================ JEPA (same style) ============================
fig, ax = new_ax(3.9)
field(ax, 2, 17, 22, 11, F_IN); ax.text(13, 14.5, "field  ω(t)", ha="center", fontsize=11)
arrow(ax, 26, 31, 22.5)
stack(ax, 33, 23, 11, 17, 4); stack(ax, 46, 23, 9, 13, 4); stack(ax, 56, 23, 7, 9, 4)
ax.text(46, 10.5, "CNN + ViT", ha="center", fontsize=11, color=NAVY)
arrow(ax, 64, 71, 22.5)
zt = neurons(ax, 75, np.linspace(13, 33, 6), GREENN, ec=GREEN)
ax.text(75, 9.5, "z(t)", ha="center", fontsize=11, color=GREEN)
arrow(ax, 79, 85, 22.5)
ax.add_patch(FancyBboxPatch((85, 13), 20, 20, boxstyle="round,pad=0.2,rounding_size=1.2",
             fc="#FDEFE3", ec=ORANGE, lw=1.6, zorder=5))
ax.text(95, 23, "Predictor\n(transformer)", ha="center", va="center", fontsize=12,
        color=NAVY, fontweight="bold", zorder=6)
# (predictor input row)
arrow(ax, 105, 111, 22.5)
zp = neurons(ax, 115, np.linspace(13, 33, 6), GREENN, ec=GREEN)
ax.text(115, 9.5, "predicted z(t+1)", ha="center", fontsize=10.5, color=GREEN)
zt1 = neurons(ax, 140, np.linspace(13, 33, 6), LGREEN, ec=GREEN)
ax.text(140, 9.5, "z(t+1) target", ha="center", fontsize=10.5, color=GREEN)
ax.text(140, 6.5, "= Encoder(ω(t+1))", ha="center", fontsize=9, color=GRAY)
# latent-space loss (double arrow between predicted and target)
ax.annotate("", xy=(135, 36), xytext=(120, 36),
            arrowprops=dict(arrowstyle="<|-|>", color=NAVY, lw=1.6))
ax.text(127.5, 37.2, "match in LATENT space", ha="center", fontsize=11, color=NAVY)
bracket(ax, 31, 60, 40.5, "Encoder")
ax.text(166, 23, "no decoder", ha="center", va="center", fontsize=12, color=RED,
        fontweight="bold", rotation=0)
ax.text(166, 19, "(anti-collapse\nkeeps z informative)", ha="center", va="center",
        fontsize=9.5, color=GRAY)
ax.text(100, 45.5, "JEPA: predict the next latent in latent space, no decoder",
        ha="center", fontsize=13.5, color=GREEN, fontweight="bold")
ax.text(100, -2.2, r"main loss:  $\mathcal{L}_{\mathrm{JEPA}}=\langle\|\hat z_{t+1}-z_{t+1}\|^{2}\rangle$   (latent space)   +  anti-collapse (SIGReg)",
        ha="center", fontsize=13, color=GREEN)
fig.savefig(OUT / "fukami_jepa.png", dpi=200)
plt.close(fig)
print("wrote", OUT / "fukami_jepa.png")
