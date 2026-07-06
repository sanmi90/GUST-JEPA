"""Figure F3 (v4): architecture and the three supervision heads.

Top panel (a): matplotlib-drawn schematic of the full pipeline
  encoder chain (omega_z 192x96 -> CNN stem -> tokens -> ViT -> BatchNorm
  projection -> pooled z in R^32), the training predictor on z_t sequences
  (two variants: vector transformer per HANDOFF D250; direct REX per the
  rexpred kit runs), and the three supervision heads L / W / N branching
  off the pooled latent.

Bottom panel (b): the Chang (Proc. R. Soc. Lond. A 437, 1992) lift auxiliary
  potential phi_L on the real 192x96 cache grid, loaded from
  outputs/data_pipeline/v2p2/phi_L.npz, with the airfoil footprint filled
  black and the delta_n = 0.3c near-body band outline
  (src.data.lift_element.build_nearbody_band on
  outputs/data_pipeline/v2p2/airfoil_adjacent_mask.npy), plus a lift-direction
  inset e_L = (-sin 14 deg, cos 14 deg).

Every architecture constant drawn as a literal carries a comment citing its
source (CLAUDE.md locked decisions, configs/_kit.yaml, HANDOFF D250, or
outputs/session35/mc_provenance.md file:line entries). CPU only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session21"))
sys.path.insert(0, str(REPO))
import figstyle  # noqa: E402

from src.data.lift_element import build_nearbody_band  # noqa: E402

GREY = "#404040"
BOX_FC = "#f2f2f2"
JEPA = figstyle.FAMILY_COLOR["jepa"]

# ---------------------------------------------------------------- constants
# Encoder chain (all from CLAUDE.md "Locked decisions > Architecture"):
#   input omega_z at native cache resolution (192, 96), single channel
#   CNN stem ~3M params, 3 downsampling stages -> 24 x 12 feature map at 256
#   channels (= 288 spatial tokens); 6-layer ViT, hidden 256, 8 heads;
#   d = 32 latent via 1-layer MLP projection with BatchNorm (NOT LayerNorm).
# Predictor variant A (vector transformer): HANDOFF.md D250 (lines 10216-10218):
#   AutoregressivePredictor, cond_dim = 0, hidden 384, depth 6, heads 16,
#   RoPE, causal mask, max_seq_len 32, rolling the (B, T, 32) pooled vector.
# Predictor variant B (direct REX / rexpred): mc_provenance.md MC-4:
#   quantile LSTM, depth 2 (scripts/session34/latent_rex.py:50), tuned
#   hidden 512 (outputs/session34/rex_tune.json winner); kit
#   predictor-class rex per HANDOFF D-entry "CLN-rexpred" (Session 34).
# Objective: L_pred + 0.5 L_roll + 0.02 SIGReg(Z).
#   H_roll = 8: configs/_kit.yaml:17 (pred.horizon); lambda = 0.02:
#   configs/_kit.yaml:27 (anti_collapse.lambda, PINNED); the 0.5 rollout
#   weight and the term structure are the CLAUDE.md locked training loss.
# Heads (mc_provenance.md MC-9):
#   L: Linear(32->64) GELU Linear(64->1), smooth_l1, target C_L
#      (src/models/observable_head.py:27-49; configs/_kit.yaml:33-37)
#   W: LayerNorm-Linear(32->128)-SiLU-Linear(128->128)-SiLU-Linear(128->80),
#      target patch_signed_spectrum 80-D (observable_head.py:101-124;
#      _kit.yaml:38-43)
#   N: same WakeObservableHead class at 80-D, target nearbody_lift_element
#      (src/training/canonical_model.py:538-541; _kit.yaml:44-49)
#   All heads smooth L1 with weight 1.0 (_kit.yaml:37, :43, :49).
# Sequence length T = 32: CLAUDE.md locked data decision (sub-trajectory
#   L = 32) and src/training/train_canonical.py:130 (mc_provenance MC-9).

PHI_NPZ = REPO / "outputs/data_pipeline/v2p2/phi_L.npz"
ADJ_MASK = REPO / "outputs/data_pipeline/v2p2/airfoil_adjacent_mask.npy"
DELTA_N = 0.3  # chords; src/data/lift_element.py build_nearbody_band default
#                and src/data/nearbody_observables.py:73 (mc_provenance MC-10)
ALPHA_DEG = 14.0  # src/data/lift_element.py:56 (ALPHA_DEG = 14.0)


# ------------------------------------------------------------------ helpers
def draw_box(ax, x0, y0, w, h, text, fc=BOX_FC, ec=GREY, fs=6.0, lw=0.9,
             text_color="black", rounding=1.2):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), w, h, boxstyle=f"round,pad=0.15,rounding_size={rounding}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
    ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center",
            fontsize=fs, color=text_color, zorder=3, linespacing=1.25)


def arrow(ax, p0, p1, color=GREY, lw=0.8, ms=6, ls="-"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
        color=color, linestyle=ls, shrinkA=0.0, shrinkB=0.0, zorder=4))


# ------------------------------------------------------------- panel (a)
def draw_schematic(ax) -> None:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    # ---- left column: encoder chain (top to bottom) ----
    cx0, cw = 2.0, 22.0
    chain = [
        # CLAUDE.md locked: encoder input omega_z at native (192, 96)
        (r"$\omega_z$ input" + "\n" + r"$192 \times 96$", BOX_FC, GREY),
        # CLAUDE.md locked: CNN stem ~3M params, 3 downsampling stages
        ("CNN stem\n3 stages, ~3M", BOX_FC, GREY),
        # CLAUDE.md locked: 24 x 12 feature map at 256 channels = 288 tokens
        ("tokens\n" + r"$24 \times 12 \times 256$", BOX_FC, GREY),
        # CLAUDE.md locked: 6-layer ViT, hidden 256, 8 heads
        ("ViT, 6 layers\nhidden 256, 8 heads", BOX_FC, GREY),
        # CLAUDE.md locked: BatchNorm projection (NOT LayerNorm; SIGReg req.)
        ("BatchNorm\nprojection", BOX_FC, GREY),
        # d = 32 pooled coefficient state (CLAUDE.md locked; D250 flagship)
        ("pooled\n" + r"$z_t \in \mathbb{R}^{32}$", "#dcecdf", JEPA),
    ]
    bh, gap = 6.6, 2.6
    y_top = 57.0
    tops = [y_top - i * (bh + gap) for i in range(len(chain))]
    ax.text(cx0 + cw / 2, 59.2, "encoder (per frame, unconditional)",
            ha="center", va="center", fontsize=7)
    for (txt, fc, ec), yt in zip(chain, tops):
        draw_box(ax, cx0, yt - bh, cw, bh, txt, fc=fc, ec=ec)
    for yt in tops[:-1]:
        arrow(ax, (cx0 + cw / 2, yt - bh - 0.15), (cx0 + cw / 2, yt - bh - gap + 0.15))

    z_bot = tops[-1] - bh          # bottom of the z box
    z_mid_y = tops[-1] - bh / 2    # its vertical centre

    # ---- latent bus from z to the predictor and the heads ----
    bus_y = z_mid_y
    ax.plot([cx0 + cw + 0.2, 64.0], [bus_y, bus_y], color=JEPA, lw=1.0, zorder=1)
    ax.text(38.0, bus_y - 2.3,
            r"pooled latent sequence $z_{1:T}$,  $T = 32$",  # CLAUDE.md L = 32;
            ha="center", va="center", fontsize=6, color=JEPA)  # train_canonical.py:130

    # ---- middle block: training predictor ----
    px0, px1, py0, py1 = 30.0, 62.0, 12.0, 55.0
    ax.add_patch(FancyBboxPatch(
        (px0, py0), px1 - px0, py1 - py0,
        boxstyle="round,pad=0.15,rounding_size=1.5", facecolor="none",
        edgecolor=JEPA, linewidth=1.1, zorder=1))
    ax.text((px0 + px1) / 2, py1 - 2.4, "training predictor",
            ha="center", va="center", fontsize=7, color=JEPA)
    ax.text((px0 + px1) / 2, py1 - 5.2, r"(acts on $z_t$ sequences)",
            ha="center", va="center", fontsize=6, color=GREY)
    # variant A: HANDOFF D250 (AutoregressivePredictor cond_dim=0, hidden 384,
    # depth 6, heads 16, RoPE, causal mask)
    draw_box(ax, px0 + 2, 38.0, px1 - px0 - 4, 9.5,
             "vector transformer (D250)\n6 layers, hidden 384, 16 heads\ncausal mask, RoPE",
             fc="#eaf3ec", ec=JEPA)
    # variant B: rexpred; MC-4: LSTM depth 2 (latent_rex.py:50), hidden 512
    # (rex_tune.json winner), 9-quantile pinball head
    draw_box(ax, px0 + 2, 28.0, px1 - px0 - 4, 8.0,
             "direct REX (rexpred)\nquantile LSTM, 2 layers,\nhidden 512",
             fc="#eaf3ec", ec=JEPA)
    # open-loop rollout, H_roll = 8: configs/_kit.yaml:17; online detached
    # targets, no EMA: _kit.yaml:22 and CLAUDE.md locked training decisions
    ax.text((px0 + px1) / 2, 24.4,
            r"open-loop rollout, $H_{\mathrm{roll}} = 8$, online targets",
            ha="center", va="center", fontsize=6, color=GREY)
    # objective: 0.5 L_roll weight per CLAUDE.md locked loss; lambda = 0.02
    # PINNED at configs/_kit.yaml:27; SIGReg M = 256 / 17 knots at :28
    ax.text((px0 + px1) / 2, 19.5, "JEPA objective",
            ha="center", va="center", fontsize=6, color="black")
    ax.text((px0 + px1) / 2, 15.8,
            r"$L_{\mathrm{pred}} + 0.5\,L_{\mathrm{roll}}"
            r" + 0.02\,\mathrm{SIGReg}(Z)$",
            ha="center", va="center", fontsize=6.5, color="black")
    arrow(ax, (46.0, bus_y), (46.0, py0 - 0.2), color=JEPA)

    # ---- right block: supervision heads ----
    hx0, hx1, hy0, hy1 = 68.0, 99.0, 12.0, 55.0
    ax.add_patch(FancyBboxPatch(
        (hx0, hy0), hx1 - hx0, hy1 - hy0,
        boxstyle="round,pad=0.15,rounding_size=1.5", facecolor="none",
        edgecolor=GREY, linewidth=1.0, zorder=1))
    # all heads smooth L1, weight 1.0: _kit.yaml:37, :43, :49
    ax.text((hx0 + hx1) / 2, hy1 - 2.4, "supervision heads",
            ha="center", va="center", fontsize=7)
    ax.text((hx0 + hx1) / 2, hy1 - 5.2, "(smooth L1, weight 1.0)",
            ha="center", va="center", fontsize=6, color=GREY)
    heads = [
        # L: observable_head.py:27-49 (Linear 32->64, GELU, 64->1); target C_L
        # current frame (_kit.yaml:33-37; train_canonical.py:235 delta (0,))
        ("L : scalar lift head\nLinear 32-64-1\n" + r"$\rightarrow\ C_L$", 47.0 - 9.0),
        # W: observable_head.py:101-124 (MLP 32-128-128-80); target
        # patch_signed_spectrum, 80-D (_kit.yaml:38-43)
        ("W : wake head\nMLP 32-128-128-80\n" + r"$\rightarrow$ 80-D wake observable",
         35.5 - 9.0),
        # N: same WakeObservableHead class at 80-D, target
        # nearbody_lift_element (canonical_model.py:538-541; _kit.yaml:44-49)
        ("N : near-body Chang head\nsame MLP class\n"
         + r"$\rightarrow$ 80-D lift-element obs.", 24.0 - 9.0),
    ]
    hbh = 9.5
    for txt, yb in heads:
        draw_box(ax, hx0 + 1.6, yb, hx1 - hx0 - 3.2, hbh, txt)
    # riser from the bus fanning into the three heads
    riser_x = 64.0
    top_head_mid = heads[0][1] + hbh / 2
    ax.plot([riser_x, riser_x], [bus_y, top_head_mid], color=JEPA, lw=1.0, zorder=1)
    for _, yb in heads:
        arrow(ax, (riser_x, yb + hbh / 2), (hx0 + 1.4, yb + hbh / 2), color=JEPA)


# ------------------------------------------------------------- panel (b)
def draw_phi_panel(ax) -> None:
    with np.load(PHI_NPZ) as z:
        phi = z["phi"]                       # (192, 96), Chang lift potential
        lift_dir = z["lift_dir"]             # (-sin 14, cos 14); lift_element.py:56-58
        res = float(z["residual_linf"])      # 6.39e-13; MC-10 discretization check
    assert abs(lift_dir[0] + np.sin(np.deg2rad(ALPHA_DEG))) < 1e-9
    assert res < 1e-10

    adjacent = np.load(ADJ_MASK)             # solid + 1-cell-adjacent, 140 px
    band = build_nearbody_band(adjacent, delta_n=DELTA_N)

    # grid convention: axis 0 = x in [-1.5, 4.5], axis 1 = y in [-1.5, 1.5]
    # (src/data/lift_element.py PHYSICAL_X / PHYSICAL_Y, node coordinates)
    x = np.linspace(-1.5, 4.5, phi.shape[0])
    y = np.linspace(-1.5, 1.5, phi.shape[1])
    X, Y = np.meshgrid(x, y, indexing="ij")

    lv = np.abs(phi).max()
    levels = np.r_[np.linspace(-lv, -0.04, 8), np.linspace(0.04, lv, 8)]
    ax.contour(X, Y, phi, levels=levels, cmap="RdBu_r", vmin=-lv, vmax=lv,
               linewidths=0.55)
    # band outline: w = clip(1 - dist/0.3c, 0, 1) support boundary
    ax.contour(X, Y, band, levels=[0.02], colors="black", linewidths=0.8,
               linestyles="dashed")
    # airfoil footprint (figstyle physical polygon from Baseline.h5 airfoil_xy)
    ax.add_patch(Polygon(figstyle._airfoil_xy(), closed=True, facecolor="black",
                         edgecolor="black", lw=0.3, zorder=5))

    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 4.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([-1, 0, 1, 2, 3, 4])
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel(r"$x/c$", labelpad=1)
    ax.set_ylabel(r"$y/c$", labelpad=0)
    ax.set_title(r"auxiliary potential $\phi_L$ (Chang 1992), Neumann BC on the body",
                 fontsize=7)
    ax.annotate(r"$\delta_n = 0.3c$ band", xy=(1.35, 0.42), xytext=(2.6, 0.95),
                fontsize=6.5,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="black"))


def draw_lift_dir_inset(ax) -> None:
    a = np.deg2rad(ALPHA_DEG)
    el = np.array([-np.sin(a), np.cos(a)])   # lift_element.py:56-58 LIFT_DIR
    uinf = np.array([np.cos(a), np.sin(a)])  # freestream inclined at alpha
    o = np.array([0.0, -0.35])

    ax.plot([-1.0, 1.0], [o[1], o[1]], color=GREY, lw=0.7,
            linestyle=(0, (4, 3)))           # chord line (footprint angle 0)
    ax.annotate("", xy=o + 0.85 * uinf, xytext=o,
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=GREY,
                                mutation_scale=8))
    ax.annotate("", xy=o + 0.85 * el, xytext=o,
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color=JEPA,
                                mutation_scale=8))
    th = np.linspace(0, a, 30)
    ax.plot(o[0] + 0.5 * np.cos(th), o[1] + 0.5 * np.sin(th), color=GREY, lw=0.6)
    ax.text(o[0] + 0.62, o[1] + 0.10, r"$\alpha = 14^\circ$", fontsize=6, color=GREY)
    ax.text(*(o + 0.85 * uinf + [0.02, -0.14]), r"$U_\infty$", fontsize=7,
            color=GREY, ha="left")
    ax.text(*(o + 0.85 * el + [0.06, 0.02]), r"$\mathbf{e}_L$", fontsize=7,
            color=JEPA, ha="left")
    ax.text(0.0, -0.95, r"$\mathbf{e}_L = (-\sin\alpha,\ \cos\alpha)$",
            fontsize=6.5, ha="center", color="black")
    ax.set_xlim(-1.1, 1.35)
    ax.set_ylim(-1.15, 0.75)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("lift direction", fontsize=7)


def main() -> None:
    figstyle.use_style()
    fig = plt.figure(figsize=(figstyle.TEXTWIDTH_IN, 4.35))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.22, 1.0],
                          width_ratios=[2.45, 1.0], hspace=0.28, wspace=0.15)
    ax_top = fig.add_subplot(gs[0, :])
    ax_phi = fig.add_subplot(gs[1, 0])
    ax_ins = fig.add_subplot(gs[1, 1])

    draw_schematic(ax_top)
    draw_phi_panel(ax_phi)
    draw_lift_dir_inset(ax_ins)

    ax_top.text(-0.01, 1.02, "(a)", transform=ax_top.transAxes, fontsize=8,
                ha="left", va="bottom")
    ax_phi.text(-0.06, 1.10, "(b)", transform=ax_phi.transAxes, fontsize=8,
                ha="left", va="bottom")

    out_dir = REPO / "paper/sections/figures/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_architecture_v4.pdf")
    fig.savefig(out_dir / "fig_architecture_v4.png", dpi=200)
    print("wrote", out_dir / "fig_architecture_v4.pdf")
    print("wrote", out_dir / "fig_architecture_v4.png")


if __name__ == "__main__":
    main()
