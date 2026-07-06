"""Figure F10 (manuscript v4): decode floor, truth vs CLW vs CLN vs AE-LW.

7 x 3 grid on one representative test_b encounter (case G-0.50_D1.00_Y-0.40,
encounter 0): rows = DNS truth, CLW decode, CLN decode, AE-LW decode, then the
three signed error rows (decode - truth); columns = pre-impact, impact peak
(max |target_C_L| within window_mask), relax (impact-window end + 16 frames).

Decode protocol copied from scripts/session34/da_phase_eval.py (decode-floor
decoders saved by the C3 chain, frozen-encoder protocol): the pooled z_gap
(N, 32) is tiled to (N, 32, 24, 12) and passed through
SpatialLatentFieldDecoder(latent_dim=32, feature_h=24, feature_w=12).

Every field is read from caches at build time:
  * truth: outputs/session34/trackc_latents/fields_test_b.npz (omega_norm);
  * latents: outputs/session34/trackc_latents/latents_{run}_test_b.npz (z_gap);
  * decoders: outputs/session34/trackc_decoders/decoder_{run}.pt.

GPU (RTX 6000, card 0). Run from the repo root:
    OMP_NUM_THREADS=4 taskset -c 0-7 .venv/bin/python \
        scripts/session35/fig_decode_panels_v4.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "session21"))

import figstyle  # noqa: E402
from src.models.decoder import SpatialLatentFieldDecoder  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

CASE = "G-0.50_D1.00_Y-0.40"
ENC = 0
DT_TC = 0.05
PRE_OFFSET = 10     # frames before the impact-window start
RELAX_OFFSET = 16   # frames after the impact-window end
ERR_VLIM = 1.5

LATENTS_DIR = REPO / "outputs/session34/trackc_latents"
DECODER_DIR = REPO / "outputs/session34/trackc_decoders"
OUT_DIR = REPO / "paper" / "sections" / "figures" / "results"

# (run tag, paper label); all three are seed s0 checkpoints.
MODELS = [
    ("jepa_pool_vec", "CLW"),
    ("jepa_pool_ln_s0", "CLN"),
    ("ae_wake_pool", "AE-LW"),
]


def row_index(npz, case: str, enc: int) -> np.ndarray:
    """Rows of a trackc npz for one encounter, sorted by frame."""
    sel = (npz["case_id"] == case) & (npz["encounter_index"] == enc)
    rows = np.where(sel)[0]
    if rows.size == 0:
        raise RuntimeError(f"{case} enc {enc} not found")
    return rows[np.argsort(npz["frame"][rows])]


def main() -> None:
    figstyle.use_style()
    device = require_rtx6000(gpu_index=0)
    print(f"[fig10] device={device} ({torch.cuda.get_device_name(device.index)})")

    # ---- truth fields + frame selection --------------------------------------
    fields = np.load(LATENTS_DIR / "fields_test_b.npz", allow_pickle=True)
    frows = row_index(fields, CASE, ENC)
    frames_all = fields["frame"][frows]
    wmask = fields["window_mask"][frows]
    cl = fields["target_C_L"][frows]
    widx = np.where(wmask)[0]
    f_peak = int(frames_all[widx[np.argmax(np.abs(cl[widx]))]])
    f_pre = max(int(frames_all[widx[0]]) - PRE_OFFSET, 0)
    f_relax = min(int(frames_all[widx[-1]]) + RELAX_OFFSET, int(frames_all[-1]))
    sel_frames = [f_pre, f_peak, f_relax]
    col_titles = ["pre-impact", "impact peak", "relax"]
    print(f"[fig10] case={CASE} enc={ENC} window="
          f"[{frames_all[widx[0]]}, {frames_all[widx[-1]]}] "
          f"frames={sel_frames} (t/c={[f * DT_TC for f in sel_frames]}) "
          f"C_L at peak={cl[widx[np.argmax(np.abs(cl[widx]))]]:.3f}")

    frame_to_row = {int(f): r for f, r in zip(frames_all, frows)}
    truth = np.stack([fields["omega_norm"][frame_to_row[f]].astype(np.float32)
                      for f in sel_frames])  # (3, 192, 96)
    cl_sel = [float(fields["target_C_L"][frame_to_row[f]]) for f in sel_frames]

    # ---- decode the three models (da_phase_eval protocol) --------------------
    tile = lambda z: np.repeat(np.repeat(z[:, :, None, None], 24, 2), 12, 3) \
        .astype(np.float32)  # noqa: E731

    decoded: dict[str, np.ndarray] = {}
    for run, label in MODELS:
        lat = np.load(LATENTS_DIR / f"latents_{run}_test_b.npz", allow_pickle=True)
        lrows = row_index(lat, CASE, ENC)
        lframe_to_row = {int(f): r for f, r in zip(lat["frame"][lrows], lrows)}
        z = np.stack([lat["z_gap"][lframe_to_row[f]] for f in sel_frames])  # (3, 32)
        decoder = SpatialLatentFieldDecoder(latent_dim=z.shape[1],
                                            feature_h=24, feature_w=12)
        decoder.load_state_dict(torch.load(
            DECODER_DIR / f"decoder_{run}.pt", map_location="cpu"))
        decoder.to(device).eval()
        with torch.no_grad():
            zb = torch.from_numpy(tile(z)).to(device)
            decoded[run] = decoder(zb).float().squeeze(1).cpu().numpy()
        print(f"[fig10] {label} ({run}): model_name={lat['model_name']}, "
              f"z_gap d={z.shape[1]}, decoded shape={decoded[run].shape}")

    # ---- per-panel full-field RMSE (normalised omega units) -------------------
    rmse = {run: np.sqrt(((decoded[run] - truth) ** 2).mean(axis=(1, 2)))
            for run, _ in MODELS}
    for run, label in MODELS:
        print(f"[fig10] RMSE {label}: "
              + ", ".join(f"{c}={v:.3f}" for c, v in zip(col_titles, rmse[run])))

    # ---- figure ---------------------------------------------------------------
    n_rows, n_cols = 1 + len(MODELS) + len(MODELS), 3
    fig = plt.figure(figsize=(figstyle.TEXTWIDTH_IN, 6.15))
    gs = fig.add_gridspec(n_rows, n_cols + 1,
                          width_ratios=[1.0] * n_cols + [0.045],
                          hspace=0.10, wspace=0.06,
                          left=0.075, right=0.93, top=0.945, bottom=0.045)

    row_labels = (["DNS"] + [f"{lab}" for _, lab in MODELS]
                  + [f"{lab} $-$ DNS" for _, lab in MODELS])
    field_rows = [truth] + [decoded[run] for run, _ in MODELS]
    error_rows = [decoded[run] - truth for run, _ in MODELS]

    im_field = im_err = None
    for r in range(n_rows):
        is_err = r >= 1 + len(MODELS)
        data = error_rows[r - 1 - len(MODELS)] if is_err else field_rows[r]
        vlim = ERR_VLIM if is_err else figstyle.VORT_VLIM
        for c in range(n_cols):
            ax = fig.add_subplot(gs[r, c])
            im = figstyle.vort_panel(ax, data[c], vlim=vlim)
            if is_err:
                im_err = im
            else:
                im_field = im
            if c == 0:
                ax.text(-0.06, 0.5, row_labels[r], transform=ax.transAxes,
                        ha="right", va="center", fontsize=7.5, rotation=90)
            if r == 0:
                ax.set_title(f"{col_titles[c]}\n"
                             rf"$t/c = {sel_frames[c] * DT_TC:.2f}$, "
                             rf"$C_L = {cl_sel[c]:.2f}$",
                             fontsize=7, pad=2.5)
            if is_err:
                run = MODELS[r - 1 - len(MODELS)][0]
                ax.text(0.985, 0.04, f"RMSE {rmse[run][c]:.2f}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=6.0, color="#404040")

    cax_f = fig.add_subplot(gs[0:1 + len(MODELS), n_cols])
    cb_f = fig.colorbar(im_field, cax=cax_f)
    cb_f.set_label(r"$\omega_z$ (normalised)", fontsize=7)
    cb_f.ax.tick_params(labelsize=6)
    cax_e = fig.add_subplot(gs[1 + len(MODELS):, n_cols])
    cb_e = fig.colorbar(im_err, cax=cax_e)
    cb_e.set_label(r"error $\Delta\omega_z$", fontsize=7)
    cb_e.ax.tick_params(labelsize=6)

    fig.text(0.5, 0.008,
             f"case {CASE} (test_b), encounter {ENC}; seed s0 per model; "
             "pooled z tiled to the 24 x 12 grid, "
             "frozen-encoder decode-floor decoders",
             ha="center", va="bottom", fontsize=6.2, color="#404040")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "fig_decode_panels_v4.pdf"
    png = OUT_DIR / "fig_decode_panels_v4.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    print(f"[fig10] wrote {pdf}")
    print(f"[fig10] wrote {png}")


if __name__ == "__main__":
    main()
