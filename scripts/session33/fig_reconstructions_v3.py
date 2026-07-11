"""Field-reconstruction galleries on v2.2 pooled states (audit replacement).

Session 33 / D247: the v2.1 decode gallery (figD) and pressure-recovered field
figure (figG) showed previous-generation d=64 spatial models. This regenerates
both on the v2.2 pooled d=32 states, from the frozen Q1 caches, with the same
diagnostic-decoder recipe as the decode floor (represent.fit_decode_floor_decoder,
6000 steps) so the panels are consistent with every SSIM number in the paper.

fig_reconstructions_v3.pdf : DNS vs decoded-from-encoded-latent, rows = DNS,
    predictive (jepa_pool), reconstructive (fukami), linear (pod); columns =
    pre-impact, impact, relaxation frames of the representative test_b
    encounter (the hero pick at |G| = 1.5). Per-panel SSIM vs DNS annotated
    (Wang constants, L = 8.487).
fig_field_recovery_v3.pdf  : pressure-recovered decode vs decode ceiling vs
    DNS at the impact frame, one row per family; the pressure step is the
    windowed KRR recovery at each family's own OSP K = 8 taps (the envelope's
    static-recovery estimator).

Run (RTX 6000; ~15-25 min, decoder training dominates):
    taskset -c 0-15 python -m scripts.session33.fig_reconstructions_v3 --gpu 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "session21"))

import matplotlib.pyplot as plt  # noqa: E402

from figstyle import TEXTWIDTH_IN, use_style, vort_panel  # noqa: E402

CASE = ("G+1.50_D1.50_Y+0.10", 1)  # hero representative pick, |G| = 1.5
FAMS = [
    # Session 38 audit: predictive entry re-pointed to the vec flagship
    ("jepa_pool_vec", "predictive (JEPA)", "outputs/session33/q1_vec_latents"),
    ("fukami", "reconstructive (Fukami)", "outputs/session31/q1_latents"),
    ("pod", "linear (POD)", "outputs/session31/q1_latents"),
]
SSIM_L = 8.487
OUT = REPO_ROOT / "outputs" / "session33" / "figures"


def ssim_panel(a, b):
    from skimage.metrics import structural_similarity

    return float(structural_similarity(a, b, data_range=SSIM_L))


def rows_for(cache_dir, model, case, enc):
    d = np.load(Path(cache_dir) / f"latents_{model}_test_b.npz", allow_pickle=True)
    sel = (d["case_id"].astype(str) == case) & (d["encounter_index"].astype(int) == enc)
    order = np.argsort(d["frame"][sel])
    z = np.asarray(d["z_spatial"], dtype=np.float32)[sel][order]
    frames = d["frame"][sel][order]
    return z, frames


def main(argv=None):
    import torch

    ap = argparse.ArgumentParser(description="v2.2 field-reconstruction galleries")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--decoder-steps", type=int, default=6000)
    ap.add_argument("--windows", default="outputs/session31/windows_v2p2.json")
    ap.add_argument("--osp-taps", default="outputs/session32/osp_taps_v2p2.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from src.evaluation.pressure_infer import load_pressure_for_alignment
    from src.evaluation.represent import decode_fields, fit_decode_floor_decoder
    from src.utils.device import require_rtx6000
    from scripts.session32.envelope_by_gust import build_o1_recovery
    from scripts.session32.track_b_pilot import windowed_features

    device = require_rtx6000(gpu_index=args.gpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_style()

    case, enc = CASE
    key = f"{case}/{enc:02d}"
    windows = json.loads((REPO_ROOT / args.windows).read_text())["windows"]
    t_imp = int(windows[key]["t_impact"]) if "t_impact" in windows[key] else int(
        windows[key]["impact"][0])
    col_frames = [t_imp - 8, t_imp + 8, t_imp + 24]
    col_names = [r"$t - t_{\mathrm{imp}} = -0.4\,c/u_\infty$", r"$+0.4$", r"$+1.2$"]

    # DNS fields for the encounter (normalised, from the shared field cache)
    fd = np.load(REPO_ROOT / "outputs/session31/q1_latents/fields_test_b.npz",
                 allow_pickle=True)
    sel = (fd["case_id"].astype(str) == case) & (fd["encounter_index"].astype(int) == enc)
    order = np.argsort(fd["frame"][sel])
    dns = np.asarray(fd["omega_norm"], dtype=np.float32)[sel][order]

    osp = json.loads((REPO_ROOT / args.osp_taps).read_text())
    # vec flagship taps live in the session33 OSP file; merge both sources
    osp.update(json.loads(
        (REPO_ROOT / "outputs/session33/osp_taps_vec.json").read_text()))
    # test_b pressure rows aligned to the encounter
    ltb = np.load(REPO_ROOT / "outputs/session33/q1_vec_latents/latents_jepa_pool_vec_test_b.npz",
                  allow_pickle=True)
    p_tb = load_pressure_for_alignment(
        ltb["case_id"].astype(str), ltb["encounter_index"], ltb["frame"],
        partition="v2p2")
    sel_p = (ltb["case_id"].astype(str) == case) & (ltb["encounter_index"].astype(int) == enc)
    p_enc = p_tb[sel_p][np.argsort(ltb["frame"][sel_p])]

    decoded = {}
    recovered = {}
    for model, label, cache_dir in FAMS:
        print(f"[recon-v3] === {model}: training diagnostic decoder ===", flush=True)
        dtr = np.load(REPO_ROOT / cache_dir / f"latents_{model}_train.npz",
                      allow_pickle=True)
        ftr = np.load(REPO_ROOT / cache_dir / "fields_train.npz", allow_pickle=True)
        grid = tuple(int(x) for x in dtr["latent_grid"])
        dec = fit_decode_floor_decoder(
            np.asarray(dtr["z_spatial"], dtype=np.float32),
            np.asarray(ftr["omega_norm"], dtype=np.float32),
            grid, device=device, steps=args.decoder_steps, seed=args.seed,
            verbose=False,
        )
        z_enc, frames = rows_for(REPO_ROOT / cache_dir, model, case, enc)
        decoded[model] = decode_fields(dec, z_enc, device)

        # pressure-recovered state -> tiled -> decode (impact frame only)
        taps = osp[model]["K8"]
        ctx_args = SimpleNamespace(
            train_cache=str(REPO_ROOT / cache_dir / "latents_%s_train.npz"),
            partition="v2p2", seed=args.seed)
        est = build_o1_recovery(model, ctx_args, np.asarray(taps))
        win = windowed_features(p_enc[:, np.sort(taps)], 30)
        z_rec = np.asarray(est.predict(win), dtype=np.float32)  # (T, d)
        z_rec_sp = np.repeat(
            np.repeat(z_rec[:, :, None, None], grid[0], axis=2), grid[1], axis=3)
        recovered[model] = decode_fields(dec, z_rec_sp, device)
        del dec

    # ---------------- gallery A: decode ceiling ----------------
    n_rows = 1 + len(FAMS)
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(TEXTWIDTH_IN, 0.185 * TEXTWIDTH_IN * n_rows),
        constrained_layout=True)
    for j, f in enumerate(col_frames):
        vort_panel(axes[0, j], dns[f])
        axes[0, j].set_title(col_names[j], fontsize=7)
        for i, (model, label, _) in enumerate(FAMS):
            ax = axes[i + 1, j]
            vort_panel(ax, decoded[model][f])
            s = ssim_panel(dns[f], decoded[model][f])
            ax.text(0.02, 0.04, f"SSIM {s:.2f}", transform=ax.transAxes,
                    fontsize=5.5, color="black",
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.6))
    axes[0, 0].set_ylabel("DNS", fontsize=7)
    for i, (_, label, _) in enumerate(FAMS):
        axes[i + 1, 0].set_ylabel(label, fontsize=7)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_reconstructions_v3.pdf", dpi=200)
    plt.close(fig)
    print(f"[recon-v3] wrote {OUT / 'fig_reconstructions_v3.pdf'}", flush=True)

    # ---------------- gallery B: pressure-recovered field ----------------
    f = col_frames[1]  # impact-side frame
    fig, axes = plt.subplots(
        len(FAMS), 3, figsize=(TEXTWIDTH_IN, 0.185 * TEXTWIDTH_IN * len(FAMS)),
        constrained_layout=True)
    for i, (model, label, _) in enumerate(FAMS):
        for j, (field, name) in enumerate((
                (recovered[model][f], "recovered from 8 taps"),
                (decoded[model][f], "decode ceiling"),
                (dns[f], "DNS"))):
            ax = axes[i, j]
            vort_panel(ax, field)
            if i == 0:
                ax.set_title(name, fontsize=7)
            if j < 2:
                s = ssim_panel(dns[f], field)
                ax.text(0.02, 0.04, f"SSIM {s:.2f}", transform=ax.transAxes,
                        fontsize=5.5, color="black",
                        bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.6))
        axes[i, 0].set_ylabel(label, fontsize=7)
    fig.savefig(OUT / "fig_field_recovery_v3.pdf", dpi=200)
    plt.close(fig)
    print(f"[recon-v3] wrote {OUT / 'fig_field_recovery_v3.pdf'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
