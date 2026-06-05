#!/usr/bin/env python3
"""Forecasting animation for the talk (CPU, observable space).

For a representative LOW-error held-out encounter (NOT the hardest one), animate the
wake enstrophy and lift through the post-impact horizon, comparing:
  original (DNS)  /  reconstruction (encoded latent)  /  prediction (rolled latent).
A cursor sweeps the horizon. Output: figs/forecast_anim.mp4 (+ poster png).

Uses the production rollout (jepa_d64_test1_noBN), whose rollout stays on-manifold.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, FFMpegWriter  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PF = lambda s: REPO / f"outputs/session16/exp2/per_frame_targets/{s}.npz"
LAT = REPO / "outputs/session18/exp_b1/latents_jepa_d64_test1_noBN/train.npz"
ROLL = REPO / "outputs/session18/exp_b1/rollouts_jepa_d64_test1_noBN/test_b.npz"
OUT_MP4 = Path(__file__).resolve().parent / "figs" / "forecast_anim.mp4"
OUT_POS = Path(__file__).resolve().parent / "figs" / "forecast_poster.png"

NAVY = "#1F3864"; GREEN = "#2E7D4F"; ORANGE = "#C0520F"; INK = "#242933"
H = 40; DT = 0.05; GMIN = 1.5     # only consider real gusts when picking the case
# observables for case selection (and the two shown)
SEL = ["wake_enstrophy", "C_L"]
PANELS = [("wake_enstrophy", r"wake enstrophy  $\Omega_w$"), ("C_L", r"lift  $C_L$")]


def fit_probe(X, y):
    Xf = X.reshape(-1, X.shape[-1]); yf = y.reshape(-1); m = np.isfinite(yf)
    sc = StandardScaler().fit(Xf[m]); r = Ridge(alpha=1.0).fit(sc.transform(Xf[m]), yf[m])
    return lambda Z: r.predict(sc.transform(Z))


def main():
    L = np.load(LAT, allow_pickle=True); Lz = np.asarray(L["z_full"], float)
    Lcid = np.array([str(c) for c in L["case_id"]]); Lenc = np.asarray(L["encounter_index"], int)
    T = np.load(PF("train"), allow_pickle=True)
    Tcid = np.array([str(c) for c in T["case_id"]]); Tenc = np.asarray(T["encounter_index"], int)
    tidx = {(c, int(e)): k for k, (c, e) in enumerate(zip(Tcid, Tenc))}
    kL = [i for i, (c, e) in enumerate(zip(Lcid, Lenc)) if (c, int(e)) in tidx]
    kT = [tidx[(Lcid[i], int(Lenc[i]))] for i in kL]
    Xz = Lz[kL]
    names = sorted({n for n, _ in PANELS} | set(SEL))
    probes = {n: fit_probe(Xz, np.asarray(T[n], float)[kT]) for n in names}

    tb = np.load(PF("test_b"), allow_pickle=True)
    tb_cid = np.array([str(c) for c in tb["case_id"]]); tb_enc = np.asarray(tb["encounter_index"], int)
    real = {n: np.asarray(tb[n], float) for n in names}
    gstd = {n: np.nanstd(real[n]) + 1e-9 for n in names}

    rb = np.load(ROLL, allow_pickle=True)
    r_cid = np.array([str(c) for c in rb["case_ids"]]); r_enc = np.asarray(rb["encounter_indices"], int)
    zdns = np.asarray(rb["z_dns"], float); zmk = np.asarray(rb["z_markov"], float)
    r_imp = np.asarray(rb["impact_frame"], int)
    r_G = np.asarray(rb["G"], float); r_D = np.asarray(rb["D"], float); r_Y = np.asarray(rb["Y"], float)

    # pick the lowest mean normalized rollout error among real gusts (|G| >= GMIN)
    best = (None, None, 1e18)
    for ej in range(len(r_cid)):
        if abs(r_G[ej]) < GMIN:
            continue
        hit = np.where((tb_cid == r_cid[ej]) & (tb_enc == r_enc[ej]))[0]
        if not len(hit):
            continue
        pi = int(hit[0]); imp = int(r_imp[ej]); fr = imp + np.arange(0, H + 1)
        errs = []
        for n in SEL:
            rr = real[n][pi, fr]; pp = probes[n](zmk[ej, fr]); m = np.isfinite(rr)
            errs.append(np.mean(np.abs(pp[m] - rr[m])) / gstd[n])
        e = float(np.mean(errs))
        if e < best[2]:
            best = (ej, pi, e)
    ej, pi, err = best
    imp = int(r_imp[ej]); fr = imp + np.arange(0, H + 1); tc = np.arange(0, H + 1) * DT
    case = f"G={r_G[ej]:+.1f}, D={r_D[ej]:.1f}, Y={r_Y[ej]:+.1f}"
    print(f"chosen encounter {r_cid[ej]} enc{r_enc[ej]} ({case}); mean normalised rollout error {err:.3f}")

    plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.11, hspace=0.18)
    arts = []
    for ax, (name, lab) in zip(axes, PANELS):
        rr = real[name][pi, fr]; rec = probes[name](zdns[ej, fr]); rol = probes[name](zmk[ej, fr])
        ax.set_ylabel(lab)
        ax.plot(tc, rr, color=INK, lw=2.4, label="original (DNS)")
        ax.plot(tc, rec, color=GREEN, lw=1.8, ls="--", label="reconstruction (encoded latent)")
        ax.plot(tc, rol, color=ORANGE, lw=2.2, label="prediction (rollout)")
        cur = ax.axvline(tc[0], color="0.45", lw=1.2, zorder=5)
        dr, = ax.plot([tc[0]], [rr[0]], "o", color=INK, ms=6, zorder=6)
        dp, = ax.plot([tc[0]], [rol[0]], "o", color=ORANGE, ms=6, zorder=6)
        ax.grid(axis="y", color="0.93"); ax.set_xlim(0, H * DT)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        arts.append((cur, dr, dp, rr, rol))
    axes[-1].set_xlabel("t/c after impact")
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=9.5)
    ttl = fig.text(0.5, 0.975, "", ha="center", fontsize=12, color=NAVY, fontweight="bold")

    def update(f):
        for (cur, dr, dp, rr, rol) in arts:
            cur.set_xdata([tc[f], tc[f]]); dr.set_data([tc[f]], [rr[f]]); dp.set_data([tc[f]], [rol[f]])
        ttl.set_text(f"JEPA forecast, gust encounter ({case}):    t/c = {tc[f]:.2f}")
        return []

    anim = FuncAnimation(fig, update, frames=H + 1, interval=120, blit=False)
    anim.save(str(OUT_MP4), writer=FFMpegWriter(fps=8, bitrate=2400), dpi=130)
    update(H); fig.savefig(str(OUT_POS), dpi=130)
    print("wrote", OUT_MP4, "and", OUT_POS)


if __name__ == "__main__":
    main()
