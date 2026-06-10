"""Two forecast windows for the UNCONDITIONED JEPA (own predictor, markov):
  ANTICIPATION: predict the IMPACT-frame observable by rolling L steps from a
                pre-impact seed (seed at impact-L). R^2 vs lead L answers
                'how early can we anticipate the gust impact?'
  FORECAST:     predict the observable at impact+H by rolling H steps from impact.
                R^2 vs horizon H answers 'how far after impact can we predict?'
Observables: wake enstrophy (the discriminator) and C_L (the control target).
test_b, tf-no-c and lstm-no-c. Reuses the validated rollout machinery.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "session17"))
sys.path.insert(0, str(REPO / "scripts" / "session20"))
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import exp2_rollout_decode_metrics as E2  # noqa: E402
import exp_closure_r2 as cr  # noqa: E402
import figstyle as fs  # noqa: E402  (shared deck/paper style: family colours, markers, fonts)
from src.models.predictor import AutoregressivePredictor, LSTMLatentPredictor  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402

LEADS = [1, 2, 4, 8, 12, 16]              # frames BEFORE impact (anticipation)
HORIZONS = [1, 2, 4, 8, 12, 16]           # frames AFTER impact (forecast)
DEVICE = require_rtx6000(gpu_index=0)
OBS = ["wake_enstrophy", "C_L"]
# Match the rest of the deck: the navy/green/blue family palette + per-family markers.
# tf-no-c -> JEPA green (predictive), lstm-no-c -> POD blue.
STYLE = {"tf": (fs.FAMILY_COLOR["jepa"], fs.FAMILY_MARKER["jepa"]),
         "lstm": (fs.FAMILY_COLOR["pod"], fs.FAMILY_MARKER["pod"])}


def load_pred(tag):
    ck = torch.load(REPO / f"outputs/session27/JEPA_d64_noc_{tag}/checkpoint_iter020000.pt",
                    map_location="cpu", weights_only=False)
    a = ck["args"]; ptype = a.get("predictor_type", "transformer")
    psd = {k[len("predictor."):]: v for k, v in ck["jepa_state_dict"].items() if k.startswith("predictor.")}
    if ptype == "lstm":
        p = LSTMLatentPredictor(64, cond_dim=0, hidden_dim=int(a["predictor_hidden"]),
                                num_layers=int(a["predictor_layers"]), max_seq_len=32)
    else:
        p = AutoregressivePredictor(latent_dim=64, cond_dim=0, max_seq_len=32)
    p.load_state_dict(psd, strict=True); p.eval().to(DEVICE)
    return p, ptype


def r2(yp, yt):
    yp, yt = np.asarray(yp), np.asarray(yt)
    return 1.0 - float(((yp - yt) ** 2).sum()) / max(float(((yt - yt.mean()) ** 2).sum()), 1e-12)


@torch.no_grad()
def roll(pred, ptype, z_seed, steps):
    if ptype == "transformer":
        return E2.rollout_markov_only(pred, z_seed, torch.zeros(1, 0, device=DEVICE), steps, DEVICE)
    return E2.rollout_autoregressive(pred, z_seed, torch.zeros(1, 0, device=DEVICE), steps, DEVICE)


def main():
    dns = np.load(cr.DNS_METRICS_PATH, allow_pickle=True)
    curves = {}
    for tag in ("tf", "lstm"):
        pred, ptype = load_pred(tag)
        probes = cr.fit_probes(REPO / f"outputs/session27/latents_own_jepa_noc_{tag}", dns)
        b = np.load(REPO / f"outputs/session27/rollouts_noc_{tag}/test_b.npz", allow_pickle=True)
        zd = torch.from_numpy(b["z_dns"].astype(np.float32)).to(DEVICE); imp = b["impact_frame"].astype(int)
        cid = [str(c) for c in (b["case_ids"] if "case_ids" in b.files else b["case_id"])]
        ei = [int(e) for e in (b["encounter_indices"] if "encounter_indices" in b.files else b["encounter_index"])]
        midx = cr.match_index(cid, ei, dns["test_b_case_id"], dns["test_b_encounter_index"])
        n = len(imp)
        for obs in OBS:
            pr = probes[obs]
            ant = {}
            for L in LEADS:
                yp, yt = [], []
                for i in range(n):
                    if midx[i] < 0 or imp[i] - L < 0:
                        continue
                    rr = roll(pred, ptype, zd[i, imp[i] - L:imp[i] - L + 1], L)[0, L].cpu().numpy()
                    yp.append(float(cr.apply_probe(rr[None], pr)[0]))
                    yt.append(float(dns[f"test_b_{obs}"][midx[i], imp[i]]))
                ant[L] = r2(yp, yt)
            fc = {}
            for H in HORIZONS:
                yp, yt = [], []
                for i in range(n):
                    if midx[i] < 0 or imp[i] + H >= zd.shape[1]:
                        continue
                    rr = roll(pred, ptype, zd[i, imp[i]:imp[i] + 1], H)[0, H].cpu().numpy()
                    yp.append(float(cr.apply_probe(rr[None], pr)[0]))
                    yt.append(float(dns[f"test_b_{obs}"][midx[i], imp[i] + H]))
                fc[H] = r2(yp, yt)
            curves[(tag, obs)] = (ant, fc)
            print(f"[{tag:4s} {obs:14s}] anticipation " + " ".join(f"-{L}:{ant[L]:+.2f}" for L in LEADS)
                  + " | forecast " + " ".join(f"+{H}:{fc[H]:+.2f}" for H in HORIZONS), flush=True)

    fs.use_style()
    # Larger type + thicker lines + a flatter, wider figure so it fills the slide box and
    # stays legible when projected (figstyle is tuned for the 5in paper column).
    plt.rcParams.update({
        "font.size": 15, "axes.titlesize": 18, "axes.labelsize": 16,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 15,
        "lines.linewidth": 2.4, "lines.markersize": 9,
    })
    fig, axes = plt.subplots(1, 2, figsize=fs.figure_size(2.0, aspect=0.30), sharey=True)
    for ax, obs in zip(axes, OBS):
        for tag in ("tf", "lstm"):
            c, mk = STYLE[tag]
            ant, fc = curves[(tag, obs)]
            xs = [-L for L in LEADS][::-1] + [0] + HORIZONS
            ys = [ant[L] for L in LEADS][::-1] + [1.0] + [fc[H] for H in HORIZONS]
            ax.plot(xs, ys, "-", marker=mk, color=c, label=f"{tag}-no-c")
        ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5); ax.axhline(0, color="k", lw=0.6, alpha=0.3)
        ax.set_title(fs.METRIC_LABEL.get(obs, obs.replace("_", " ")))
        ax.set_xlabel("frames relative to impact\n(<0 anticipation, >0 forecast)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"$R^2$ (test_b)"); axes[0].legend(); axes[0].set_ylim(-0.35, 1.05)
    fig.tight_layout()
    out = REPO / "outputs/session27/forecast_windows.pdf"
    fig.savefig(out, bbox_inches="tight"); print(f"\n[windows] -> {out}")


if __name__ == "__main__":
    main()
