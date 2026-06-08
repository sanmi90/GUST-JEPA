#!/usr/bin/env python3
"""Forecasting animation for the talk, UNCONDITIONED tf-no-c JEPA (GPU roll, CPU plot).

Copy of make_forecast_anim.py repointed at the fully unconditioned model
(outputs/session27/JEPA_d64_noc_tf encoder + its OWN cond_dim=0 transformer
predictor). For a representative MODERATE-G held-out (test_b) encounter, animate
the wake enstrophy and lift through the post-impact horizon, comparing:
  original (DNS)  /  reconstruction (encoded latent)  /  prediction (rolled latent).
A cursor sweeps the horizon.

Differences vs the production script (which is NOT modified):
  - latents/observable probes use the UNCONDITIONED encoder latents
    (outputs/session27/latents_own_jepa_noc_tf/train.npz, keys case_ids /
    encounter_indices), NOT the conditioned S12_E_d64 production latents.
  - the prediction trace is the model's OWN unconditioned transformer predictor
    rolled LIVE (markov-only, cond = torch.zeros(1, 0), fed RAW BatchNorm-output
    latents), seeded at the impact-frame latent. This reproduces the precomputed
    outputs/session27/rollouts_noc_tf/test_b.npz z_markov exactly.
  - the encounter is PINNED to a representative moderate-G case so it matches the
    field animation (same encounter).
  - output goes to outputs/session27/anim_noc_tf/ (NEVER paper/talk/figs/).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, FFMpegWriter  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "session17"))
import exp2_rollout_decode_metrics as E2  # noqa: E402
from src.models.predictor import AutoregressivePredictor  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402

PF = lambda s: REPO / f"outputs/session16/exp2/per_frame_targets/{s}.npz"
# UNCONDITIONED tf-no-c artefacts (read-only).
ENC_CKPT = REPO / "outputs/session27/JEPA_d64_noc_tf/checkpoint_iter020000.pt"
LAT = REPO / "outputs/session27/latents_own_jepa_noc_tf/train.npz"          # raw uncond latents
ROLL = REPO / "outputs/session27/rollouts_noc_tf/test_b.npz"                # z_dns/z_markov metadata
OUT_DIR = REPO / "outputs/session27/anim_noc_tf"
OUT_MP4 = OUT_DIR / "forecast_anim.mp4"
OUT_POS = OUT_DIR / "forecast_poster.png"

# Representative MODERATE-G held-out encounter (|G| = 1.5, lowest uncond rollout
# error among 1.0 <= |G| <= 2.0; NOT the worst case, NOT an extrapolation case).
PIN_CASE, PIN_K = "G+1.50_D1.50_Y+0.10", 0

NAVY = "#1F3864"; GREEN = "#2E7D4F"; ORANGE = "#C0520F"; INK = "#242933"
H = 40; DT = 0.05
SEL = ["wake_enstrophy", "C_L"]
PANELS = [("wake_enstrophy", r"wake enstrophy  $\Omega_w$"), ("C_L", r"lift  $C_L$")]


def fit_probe(X, y):
    Xf = X.reshape(-1, X.shape[-1]); yf = y.reshape(-1); m = np.isfinite(yf)
    sc = StandardScaler().fit(Xf[m]); r = Ridge(alpha=1.0).fit(sc.transform(Xf[m]), yf[m])
    return lambda Z: r.predict(sc.transform(Z))


def load_own_predictor(device):
    """The model's OWN unconditioned transformer predictor (cond_dim = 0)."""
    ck = torch.load(ENC_CKPT, map_location="cpu", weights_only=False)
    a = ck["args"]
    assert int(a["predictor_cond_dim"]) == 0, "expected unconditioned predictor (cond_dim=0)"
    assert a["predictor_type"] == "transformer", "expected transformer predictor"
    psd = {k[len("predictor."):]: v for k, v in ck["jepa_state_dict"].items()
           if k.startswith("predictor.")}
    p = AutoregressivePredictor(latent_dim=int(a["d"]), cond_dim=0, max_seq_len=int(a["T"]))
    p.load_state_dict(psd, strict=True)
    p.eval().to(device)
    assert p.cond_dim == 0, "predictor cond_dim must be 0"
    return p


def main():
    device = require_rtx6000(0)
    print("device", device, torch.cuda.get_device_name(device.index))

    # observable probes on the UNCONDITIONED train latents
    L = np.load(LAT, allow_pickle=True); Lz = np.asarray(L["z_full"], float)
    Lcid = np.array([str(c) for c in L["case_ids"]]); Lenc = np.asarray(L["encounter_indices"], int)
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

    rb = np.load(ROLL, allow_pickle=True)
    r_cid = np.array([str(c) for c in rb["case_ids"]]); r_enc = np.asarray(rb["encounter_indices"], int)
    zdns = np.asarray(rb["z_dns"], float)
    r_imp = np.asarray(rb["impact_frame"], int)
    r_G = np.asarray(rb["G"], float); r_D = np.asarray(rb["D"], float); r_Y = np.asarray(rb["Y"], float)

    ej = int(np.where((r_cid == PIN_CASE) & (r_enc == PIN_K))[0][0])
    pi = int(np.where((tb_cid == PIN_CASE) & (tb_enc == PIN_K))[0][0])
    imp = int(r_imp[ej]); fr = imp + np.arange(0, H + 1); tc = np.arange(0, H + 1) * DT
    case = f"G={r_G[ej]:+.1f}, D={r_D[ej]:.1f}, Y={r_Y[ej]:+.1f}"

    # Roll the OWN unconditioned predictor LIVE from the impact-frame latent.
    pred = load_own_predictor(device)
    cond0 = torch.zeros(1, 0, device=device)
    steps = 119 - imp
    z_init = torch.from_numpy(zdns[ej, imp:imp + 1].astype(np.float32)).to(device)
    zm_roll = E2.rollout_markov_only(pred, z_init, cond0, steps, device)[0].float().cpu().numpy()
    zmk = zdns[ej].copy()
    zmk[imp + 1:120] = zm_roll[1:steps + 1]
    # parity check vs the precomputed z_markov (same own predictor)
    chk = float(np.abs(zmk[fr] - np.asarray(rb["z_markov"], float)[ej, fr]).max())
    errs = []
    for n in SEL:
        rr = real[n][pi, fr]; pp = probes[n](zmk[fr]); m = np.isfinite(rr)
        gstd = np.nanstd(real[n]) + 1e-9
        errs.append(np.mean(np.abs(pp[m] - rr[m])) / gstd)
    print(f"chosen encounter {PIN_CASE} enc{PIN_K} ({case}); impact={imp}; "
          f"mean normalised rollout error {float(np.mean(errs)):.3f}; "
          f"live-roll vs precomputed z_markov max|diff|={chk:.2e} (own cond_dim=0 predictor)")

    plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.11, hspace=0.18)
    arts = []
    for ax, (name, lab) in zip(axes, PANELS):
        rr = real[name][pi, fr]; rec = probes[name](zdns[ej, fr]); rol = probes[name](zmk[fr])
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
        ttl.set_text(f"JEPA (unconditioned) forecast, gust encounter ({case}):    t/c = {tc[f]:.2f}")
        return []

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=H + 1, interval=120, blit=False)
    anim.save(str(OUT_MP4), writer=FFMpegWriter(fps=8, bitrate=2400), dpi=130)
    update(H); fig.savefig(str(OUT_POS), dpi=130)
    print("wrote", OUT_MP4, "and", OUT_POS, f"({H + 1} frames)")


if __name__ == "__main__":
    main()
