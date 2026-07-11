"""d = 4 flow + C_L estimation across gust intensities (Session 38).

The wake-supervised JEPA at d = 4 (jepa_pool_vec_d4) under the REX-EnKF of
the D261 per-family protocol (own OSP K = 8 taps, own E_obs, own REX
operator, deployment-clean global Gamma): per |G| in {1, 2, 3} one
representative \TestSplit{} encounter (median impact analysis R2 on the
frozen d4 phase-eval rex_enkf records), showing the DNS vorticity, the
field DECODED FROM THE FOUR-COEFFICIENT ANALYSIS STATE, and the C_L trace.
The |G| = 4 boundary is not cached at d = 4 (trackc caches are
train/test_b) and is stated out of frame.

Run (RTX 6000): taskset -c 0-15 python -m scripts.session38.fig_d4_flow_cl --gpu 0
"""
from __future__ import annotations

import argparse
import json
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

MODEL = "jepa_pool_vec_d4"
CACHE = "outputs/session34/trackc_latents"
BAND_TO_SIGMA = 2.5631
SHOW_FRAME_OFFSET = 8  # t_imp + 0.4 c/u_inf
DELAY, K, N, ALPHA, BAND = 10, 8, 64, 1.0, 1.0


def pick_representatives():
    d = json.loads((REPO_ROOT / f"outputs/session34/da_phase_dim_{MODEL}.json").read_text())
    recs = d["records"]["rex_enkf"]
    picks = []
    for g in (1.0, 2.0, 3.0):
        cand = [r for r in recs
                if abs(abs(float(r["case_id"].split("_")[0][1:])) - g) < 1e-6
                and r["impact"]["r2"] is not None]
        vals = np.array([r["impact"]["r2"] for r in cand])
        med = np.median(vals)
        r = cand[int(np.argmin(np.abs(vals - med)))]
        picks.append((r["case_id"], int(r["encounter_index"]), g))
    return picks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from src.models.decoder import SpatialLatentFieldDecoder
    from src.utils.device import require_rtx6000

    device = require_rtx6000(gpu_index=args.gpu)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    use_style()

    tr = load_aligned(REPO_ROOT / CACHE, REPO_ROOT / CACHE, MODEL, "train")
    tb = load_aligned(REPO_ROOT / CACHE, REPO_ROOT / CACHE, MODEL, "test_b")
    encs_tr, encs_tb = encounters(tr), encounters(tb)
    n = tr["z"].shape[1]
    taps = np.asarray(json.loads(
        (REPO_ROOT / "outputs/session34/osp_taps_dims.json").read_text())
        [MODEL][f"K{K}"], dtype=int)
    p_mu = tr["p"][:, taps].mean(axis=0)
    p_sd = tr["p"][:, taps].std(axis=0) + 1e-9

    X_tr = np.concatenate([delay_embed((tr["p"][e["rows"]][:, taps] - p_mu) / p_sd,
                                       DELAY) for e in encs_tr])
    Zt_tr = np.concatenate([tr["z"][e["rows"]] for e in encs_tr])
    W = np.linalg.solve(X_tr.T @ X_tr + ALPHA * np.eye(X_tr.shape[1]), X_tr.T @ Zt_tr)
    Gamma = np.cov((Zt_tr - X_tr @ W).T) + 1e-8 * np.eye(n)
    chol_G = np.linalg.cholesky(Gamma)
    from src.evaluation.represent import fit_linear_probe
    probe = fit_linear_probe(tr["z"], tr["cl"])

    rex = LatentRex(d=n, horizon=40)
    rex.load_state_dict(torch.load(
        REPO_ROOT / f"outputs/session34/latent_rex_model_{MODEL}.pt", map_location="cpu"))
    rex.to(device).eval()

    decoder = SpatialLatentFieldDecoder(latent_dim=n, feature_h=24, feature_w=12)
    decoder.load_state_dict(torch.load(
        REPO_ROOT / f"outputs/session34/trackc_decoders/decoder_{MODEL}.pt",
        map_location="cpu"))
    decoder.to(device).eval()
    tile = lambda z: np.repeat(np.repeat(z[:, :, None, None], 24, 2), 12, 3).astype(np.float32)

    fields = np.load(REPO_ROOT / CACHE + "/fields_test_b.npz" if False else
                     REPO_ROOT / CACHE / "fields_test_b.npz", allow_pickle=True)
    fkey = {(c, int(e), int(f)): i for i, (c, e, f) in enumerate(zip(
        fields["case_id"].tolist(), fields["encounter_index"].tolist(),
        fields["frame"].tolist()))}
    omega = fields["omega_norm"].astype(np.float32)

    @torch.no_grad()
    def rex_step(ctx_np):
        out = rex(torch.from_numpy(ctx_np[:, -30:]).float().to(device))
        s0 = out[:, 0].cpu().numpy()
        med = s0[..., s0.shape[-1] // 2]
        sig = np.clip((s0[..., -1] - s0[..., 0]) / BAND_TO_SIGMA * BAND, 1e-4, None)
        return med, sig

    picks = pick_representatives()
    print("picks:", picks)
    t_init = DELAY - 1
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 0.78 * TEXTWIDTH_IN))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 1.15, 1.0], hspace=0.12,
                          wspace=0.06)

    for j, (cid, k, g) in enumerate(picks):
        e = next(x for x in encs_tb if x["case_id"] == cid and x["k"] == k)
        rows = e["rows"]
        frames = tb["frame"][rows] if "frame" in tb else np.arange(rows.size)
        cl_true = tb["cl"][rows]
        wmask = tb["wmask"][rows]
        T = rows.size
        t_imp = int(np.nonzero(wmask)[0].min()) + 0  # window start ~ impact-2
        # cache impact frame estimate = 40; use the standard 40 for display
        t_imp = 40
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
        cl_hat = probe.predict(zA)

        tf = t_imp + SHOW_FRAME_OFFSET
        with torch.no_grad():
            dec = decoder(torch.from_numpy(tile(zA[tf:tf + 1])).to(device)) \
                .float().squeeze(1).cpu().numpy()[0]
        dns = omega[fkey[(cid, k, tf)]]

        ax = fig.add_subplot(gs[0, j])
        vort_panel(ax, dns)
        m = __import__("re").match(r"G[+-][\d.]+_D([\d.]+)_Y([+-][\d.]+)", cid)
        ax.set_title(f"$|G| = {g:g}$, $D = {float(m.group(1)):g}$, "
                     f"$Y = {float(m.group(2)):+g}$", fontsize=6.5)
        if j == 0:
            ax.set_ylabel("DNS", fontsize=6.5)
        ax = fig.add_subplot(gs[1, j])
        vort_panel(ax, dec)
        if j == 0:
            ax.set_ylabel("decoded from\n$d=4$ analysis", fontsize=6.5)
        ax = fig.add_subplot(gs[2, j])
        tt = (np.arange(T, dtype=float) - t_imp) * 0.05
        sel = np.arange(T) >= DELAY
        ax.plot(tt[sel], cl_true[sel], color="k", lw=1.0, label="DNS truth")
        ax.plot(tt[sel], cl_hat[sel], color="#1b7837", lw=1.0,
                label="analysis ($d = 4$)")
        ax.axvline(0.0, color="0.75", lw=0.6, zorder=0)
        ax.axvline(tt[t_imp + SHOW_FRAME_OFFSET], color="0.55", lw=0.6,
                   ls=":", zorder=0)
        ax.set_xlabel(r"$(t - t_{\mathrm{imp}})\,U_\infty/c$", fontsize=6.5)
        if j == 0:
            ax.set_ylabel(r"$C_L$", fontsize=6.5)
            ax.legend(fontsize=5.5, frameon=False, loc="upper left")

    fig.suptitle(r"Four-coefficient estimation: flow and $C_L$ from eight "
                 r"wall taps (REX-EnKF, \TestSplit-equivalent split test\_b)"
                 .replace(r"\TestSplit-equivalent split test\_b",
                          "in-distribution test"), fontsize=7.5, y=0.98)
    out = REPO_ROOT / "outputs/session38"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_d4_flow_cl.{ext}", bbox_inches="tight", dpi=300)
        print(f"wrote {out / ('fig_d4_flow_cl.' + ext)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
