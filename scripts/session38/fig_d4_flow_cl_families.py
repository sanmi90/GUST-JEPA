"""d = 4 flow + C_L estimation, three families side by side (Session 38).

Same encounter, same filter recipe (REX-EnKF, D261 per-family stacks: own
OSP K = 8 taps, own E_obs, own REX operator, global-Gamma update), three
reduced states at d = 4: POD, the wake-headed reconstructive AE (Fukami)
and the wake-supervised JEPA. Columns: DNS + one per family (field decoded
from that family's analysis state at t_imp + 0.4 c/U); bottom row: the C_L
trace per family against the DNS truth. Encounter = the |G| = 3 median-rule
representative of the JEPA d4 phase eval (same encounter for every family,
so the fields are directly comparable).

Run (RTX 6000): taskset -c 0-15 python -m scripts.session38.fig_d4_flow_cl_families --gpu 0
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "session21"))

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from figstyle import TEXTWIDTH_IN, use_style, vort_panel  # noqa: E402
from scripts.session34.lae_enkf_pilot import delay_embed, encounters, load_aligned  # noqa: E402
from scripts.session34.latent_rex import LatentRex  # noqa: E402

CACHE = REPO_ROOT / "outputs/session34/trackc_latents"
BAND_TO_SIGMA = 2.5631
SHOW = 8  # frames after impact
# band 1.77 = the production coverage calibration (T5, validation-only);
# the dims-grid arm ran 4.0 (test-peeked, excluded from headlines) and
# the first draft of this script ran 1.0: superseded.
DELAY, K, N, ALPHA, BAND = 10, 8, 64, 1.0, 1.77
ENCOUNTER = ("G+3.00_D1.00_Y+0.10", 3, 3.0)  # JEPA-median |G|=3 pick, shared

FAMILIES = [
    ("pod_d4", "POD", "#4d4d4d"),
    ("fukami_wake_d4", "AE + wake head", "#b2182b"),
    ("jepa_pool_vec_d4", "JEPA (wake)", "#1b7837"),
    ("cln_rexpred_d4_s0", "JEPA (lift-focused)", "#5aae61"),
]


def taps_for(model):
    for f in ("outputs/session34/osp_taps_dims.json",
              "outputs/session34/osp_taps_dims2.json"):
        d = json.loads((REPO_ROOT / f).read_text())
        if model in d:
            return np.asarray(d[model][f"K{K}"], dtype=int)
    raise KeyError(model)


def run_family(model, device, rng):
    from src.evaluation.represent import fit_linear_probe
    from src.models.decoder import SpatialLatentFieldDecoder

    tr = load_aligned(CACHE, CACHE, model, "train")
    tb = load_aligned(CACHE, CACHE, model, "test_b")
    encs_tr, encs_tb = encounters(tr), encounters(tb)
    n = tr["z"].shape[1]
    taps = taps_for(model)
    p_mu = tr["p"][:, taps].mean(axis=0)
    p_sd = tr["p"][:, taps].std(axis=0) + 1e-9
    X_tr = np.concatenate([delay_embed((tr["p"][e["rows"]][:, taps] - p_mu) / p_sd,
                                       DELAY) for e in encs_tr])
    Zt_tr = np.concatenate([tr["z"][e["rows"]] for e in encs_tr])
    W = np.linalg.solve(X_tr.T @ X_tr + ALPHA * np.eye(X_tr.shape[1]), X_tr.T @ Zt_tr)
    Gamma = np.cov((Zt_tr - X_tr @ W).T) + 1e-8 * np.eye(n)
    chol_G = np.linalg.cholesky(Gamma)
    probe = fit_linear_probe(tr["z"], tr["cl"])
    rex = LatentRex(d=n, horizon=40)
    rex.load_state_dict(torch.load(
        REPO_ROOT / f"outputs/session34/latent_rex_model_{model}.pt", map_location="cpu"))
    rex.to(device).eval()
    dec = SpatialLatentFieldDecoder(latent_dim=n, feature_h=24, feature_w=12)
    dec.load_state_dict(torch.load(
        REPO_ROOT / f"outputs/session34/trackc_decoders/decoder_{model}.pt",
        map_location="cpu"))
    dec.to(device).eval()

    @torch.no_grad()
    def rex_step(ctx_np):
        out = rex(torch.from_numpy(ctx_np[:, -30:]).float().to(device))
        s0 = out[:, 0].cpu().numpy()
        med = s0[..., s0.shape[-1] // 2]
        sig = np.clip((s0[..., -1] - s0[..., 0]) / BAND_TO_SIGMA * BAND, 1e-4, None)
        return med, sig

    cid, k, _ = ENCOUNTER
    e = next(x for x in encs_tb if x["case_id"] == cid and x["k"] == k)
    rows = e["rows"]
    cl_true = tb["cl"][rows]
    T = rows.size
    t_init = DELAY - 1
    pt = (tb["p"][rows][:, taps] - p_mu) / p_sd
    z_obs = delay_embed(pt, DELAY) @ W
    ctx = np.repeat(z_obs[None, :t_init + 1], N, axis=0)
    ctx[:, -1] += rng.standard_normal((N, n)) @ chol_G.T
    zA = np.empty((T, n)); zA[:t_init + 1] = z_obs[:t_init + 1]
    for t in range(t_init + 1, T):
        med, sig = rex_step(ctx)
        zf = med + rng.standard_normal((N, n)) * sig
        dZ = zf - zf.mean(0, keepdims=True)
        P = dZ.T @ dZ / (N - 1)
        K_g = P @ np.linalg.inv(P + Gamma)
        innov = z_obs[t][None] + rng.standard_normal((N, n)) @ chol_G.T - zf
        za = zf + innov @ K_g.T
        zA[t] = za.mean(0)
        ctx = np.concatenate([ctx, za[:, None]], axis=1)[:, -30:]
    tile = lambda z: np.repeat(np.repeat(z[:, :, None, None], 24, 2), 12, 3).astype(np.float32)
    tf = 40 + SHOW
    with torch.no_grad():
        field = dec(torch.from_numpy(tile(zA[tf:tf + 1])).to(device)) \
            .float().squeeze(1).cpu().numpy()[0]
    return cl_true, probe.predict(zA), field, T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from src.utils.device import require_rtx6000
    device = require_rtx6000(gpu_index=args.gpu)
    torch.manual_seed(args.seed)
    use_style()

    cid, k, g = ENCOUNTER
    fields_tb = np.load(CACHE / "fields_test_b.npz", allow_pickle=True)
    fkey = {(c, int(e), int(f)): i for i, (c, e, f) in enumerate(zip(
        fields_tb["case_id"].tolist(), fields_tb["encounter_index"].tolist(),
        fields_tb["frame"].tolist()))}
    dns = fields_tb["omega_norm"].astype(np.float32)[fkey[(cid, k, 40 + SHOW)]]

    results = {}
    for model, label, c in FAMILIES:
        rng = np.random.default_rng(args.seed)  # same draws per family
        results[model] = run_family(model, device, rng)
        print(f"[fam] {model} done", flush=True)

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 0.52 * TEXTWIDTH_IN))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 1.0], hspace=0.02, wspace=0.10)
    ax = fig.add_subplot(gs[0, 0])
    vort_panel(ax, dns)
    m = re.match(r"G[+-][\d.]+_D([\d.]+)_Y([+-][\d.]+)", cid)
    ax.set_title(f"DNS  ($|G| = {g:g}$, $D = {float(m.group(1)):g}$, "
                 f"$Y = {float(m.group(2)):+g}$)", fontsize=6)
    for j, (model, label, c) in enumerate(FAMILIES):
        cl_true, cl_hat, field, T = results[model]
        ax = fig.add_subplot(gs[0, j + 1])
        vort_panel(ax, field)
        ax.set_title(label, fontsize=6.5)
        ax = fig.add_subplot(gs[1, j + 1])
        tt = (np.arange(T, dtype=float) - 40) * 0.05
        sel = np.arange(T) >= DELAY
        ax.plot(tt[sel], cl_true[sel], color="k", lw=1.0)
        ax.plot(tt[sel], cl_hat[sel], color=c, lw=1.0)
        ax.axvline(0.0, color="0.75", lw=0.6, zorder=0)
        ax.axvline(tt[40 + SHOW], color="0.55", lw=0.6, ls=":", zorder=0)
        lo, hi = cl_true.min(), cl_true.max()
        pad = 0.35 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(r"$(t - t_{\mathrm{imp}})\,U_\infty/c$", fontsize=6.5)
        if j == 0:
            ax.set_ylabel(r"$C_L$", fontsize=6.5)
    axl = fig.add_subplot(gs[1, 0]); axl.axis("off")
    axl.plot([], [], color="k", lw=1.0, label="DNS truth")
    for model, label, c in FAMILIES:
        axl.plot([], [], color=c, lw=1.0, label=label)
    axl.legend(fontsize=6, frameon=False, loc="center")
    fig.suptitle("$d = 4$ estimation from eight wall taps, one filter for all "
                 "families (REX-EnKF), same encounter; fields = decoded "
                 "analysis at the dotted instant", fontsize=7, y=1.00)
    out = REPO_ROOT / "outputs/session38"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_d4_flow_cl_families.{ext}", bbox_inches="tight", dpi=300)
        print(f"wrote {out / ('fig_d4_flow_cl_families.' + ext)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
