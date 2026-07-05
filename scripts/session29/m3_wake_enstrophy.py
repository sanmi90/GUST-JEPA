"""M3: large-scale wake-enstrophy reconstruction check (decoded field vs DNS).

A reconstruction-side enstrophy metric computed from the DECODED field, OUTSIDE
the supervised wake family, to break the Parseval circularity of the wake-R^2
probe. For each config we decode the impact-window (frames [25,55)) from the TRUE
encoded latent (recon endpoint), low-pass the wake ROI by 24x12 average pooling
(the "large-scale" wake field), take its enstrophy mean((pooled omega)^2) per
frame, and compare to the same quantity on the DNS reference.

Configs (all d=64, test_b):
  nowake : JEPA no-wake-head encoder + its wake-loss decoder (dec_nowake_s0);
           decoded from z_dns in the rollout we already generated.
  wake   : production wake-supervised JEPA (decode/tf_noc/recon_test_b.npz).
  fukami : reconstructive AE (decode/fukami/recon_test_b.npz).

Reports, per config: mean relative error of the large-scale wake enstrophy
(|E_dec - E_ref| / E_ref) and the Pearson correlation between decoded and
reference enstrophy across (encounter, frame). RTX 6000 only.

Usage: python scripts/session29/m3_wake_enstrophy.py --device cuda:3
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "session28"))

import h5py  # noqa: E402
import decode_operating_points as DOP  # noqa: E402
from src.data.wake_observables import _wake_roi_window  # noqa: E402

DECODE_DIR = REPO / "outputs" / "session28" / "decode"
ROLL_DIR = REPO / "outputs" / "session28" / "rollouts"
RUNS = REPO / "outputs" / "runs" / "session28"
PARTS_DIR = REPO / "outputs" / "session28" / "numbers_parts"
REPORT_DIR = REPO / "outputs" / "session29" / "reports"
WIN_LO, WIN_HI = 25, 55
POOL_H, POOL_W = 24, 12

CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE",
             str(Path(os.environ["PREVENT_ROOT"]) / "data" / "processed" / "vortex-jepa")))


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip()
    except Exception:
        return "unknown"


def _assert_rtx6000(device):
    name = torch.cuda.get_device_name(device.index if device.index is not None else 0)
    if not ("RTX" in name and "6000" in name):
        raise SystemExit(f"[m3] refusing non-RTX-6000 device {device}: {name!r}")
    print(f"[m3] device {device} = {name}")


def large_scale_wake_enstrophy(omega: np.ndarray) -> np.ndarray:
    """(n, F, 192, 96) raw omega -> (n, F) large-scale wake enstrophy.

    Crop the wake ROI, average-pool to 24x12 (low-pass = large scales), then take
    the ROI-mean of the squared pooled field.
    """
    r0, r1, c0, c1 = _wake_roi_window()
    t = torch.from_numpy(np.asarray(omega, dtype=np.float32))
    n, Fr, H, W = t.shape
    roi = t[:, :, r0:r1, c0:c1].reshape(n * Fr, 1, r1 - r0, c1 - c0)
    pooled = F.adaptive_avg_pool2d(roi, (POOL_H, POOL_W))     # large-scale field
    ens = (pooled ** 2).mean(dim=(1, 2, 3)).reshape(n, Fr)    # mean omega^2
    return ens.numpy()


def reference_window(case_ids, enc_idx, win_frames) -> np.ndarray:
    """Load DNS reference omega_z at win_frames (raw) per encounter from the v2.1 cache."""
    wf = np.asarray(win_frames, dtype=np.int64)
    out = np.zeros((len(case_ids), wf.size, 192, 96), dtype=np.float32)
    for i, (cid, ei) in enumerate(zip(case_ids, enc_idx)):
        p = CACHE / "v2p1" / str(cid) / f"encounter_{int(ei):02d}.h5"
        with h5py.File(p, "r") as f:
            out[i] = f["omega_z"][wf].astype(np.float32)
    return out


def decode_nowake_window(device, pipe):
    """Decode the recon window from z_dns of the no-wake rollout we generated."""
    roll = np.load(ROLL_DIR / "ctrl_pred_vit_nowake_s0" / "test_b.npz", allow_pickle=True)
    z = roll["z_dns"].astype(np.float32)                 # (n,120,d)
    ckpt = RUNS / "dec_nowake_s0" / "decoder_iter030000.pt"
    dec = DOP.build_decoder_from_ckpt(ckpt, latent_dim=z.shape[2], device=device)
    win = np.arange(DOP.WIN_LO, DOP.WIN_HI)              # same window as decode_family
    n = z.shape[0]
    fields = np.zeros((n, win.size, 192, 96), dtype=np.float32)
    for i in range(n):
        fields[i] = DOP.decode_latents(dec, z[i, win], pipe, device)
    cids = roll["case_ids"] if "case_ids" in roll.files else roll["case_id"]
    eidx = roll["encounter_indices"] if "encounter_indices" in roll.files else roll["encounter_index"]
    return fields, np.asarray(cids), np.asarray(eidx), win


def load_existing(fam):
    d = np.load(DECODE_DIR / fam / "recon_test_b.npz", allow_pickle=True)
    return (d["omega_decoded_window"].astype(np.float32),
            np.asarray(d["case_ids"]), np.asarray(d["encounter_indices"]),
            np.asarray(d["window_frames"]))


def metrics(dec_fields, case_ids, enc_idx, win_frames):
    ref_fields = reference_window(case_ids, enc_idx, win_frames)
    e_dec = large_scale_wake_enstrophy(dec_fields)
    e_ref = large_scale_wake_enstrophy(ref_fields)
    eps = 1e-8
    rel_err = np.abs(e_dec - e_ref) / (e_ref + eps)
    a, b = e_dec.ravel(), e_ref.ravel()
    r = float(np.corrcoef(a, b)[0, 1])
    return {
        "rel_err_mean": float(rel_err.mean()),
        "rel_err_median": float(np.median(rel_err)),
        "pearson_r": r,
        "e_dec_mean": float(e_dec.mean()),
        "e_ref_mean": float(e_ref.mean()),
        "n_enc": int(dec_fields.shape[0]),
        "n_frames": int(dec_fields.shape[1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:3")
    args = ap.parse_args()
    device = torch.device(args.device)
    _assert_rtx6000(device)
    pipe, _ = DOP.load_pipeline()

    print("[m3] decoding no-wake recon window from z_dns ...")
    nw_fields, nw_cids, nw_eidx, nw_wf = decode_nowake_window(device, pipe)
    wk_fields, wk_cids, wk_eidx, wk_wf = load_existing("tf_noc")
    fk_fields, fk_cids, fk_eidx, fk_wf = load_existing("fukami")

    res = {
        "nowake (JEPA no wake head + wake-loss decoder)": metrics(nw_fields, nw_cids, nw_eidx, nw_wf),
        "wake (production JEPA)": metrics(wk_fields, wk_cids, wk_eidx, wk_wf),
        "fukami (reconstructive AE)": metrics(fk_fields, fk_cids, fk_eidx, fk_wf),
    }

    part = {
        "part": "m3_wake_enstrophy",
        "numbers": {
            "m3_lswake_enstrophy_relerr_nowake": {
                "value": res["nowake (JEPA no wake head + wake-loss decoder)"]["rel_err_median"],
                "fmt": "%.2f", "split": "test_b",
                "source": "m3_wake_enstrophy", "note": "median rel-err of large-scale (24x12) wake-ROI enstrophy, decoded vs DNS, no-wake encoder",
            },
            "m3_lswake_enstrophy_relerr_wake": {
                "value": res["wake (production JEPA)"]["rel_err_median"],
                "fmt": "%.2f", "split": "test_b",
                "source": "m3_wake_enstrophy", "note": "same, production wake-supervised JEPA",
            },
            "m3_lswake_enstrophy_relerr_fukami": {
                "value": res["fukami (reconstructive AE)"]["rel_err_median"],
                "fmt": "%.2f", "split": "test_b",
                "source": "m3_wake_enstrophy", "note": "same, reconstructive Fukami AE",
            },
        },
    }
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    (PARTS_DIR / "m3_wake_enstrophy.json").write_text(json.dumps(part, indent=2, sort_keys=True))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "# M3: large-scale wake-enstrophy reconstruction (decoded field vs DNS)\n",
        f"- git sha: `{_git_sha()}`",
        f"- UTC: {ts}",
        f"- recon endpoint (true encoded latent), window [25,55), wake ROI pooled to {POOL_H}x{POOL_W}",
        "- metric outside the supervised wake family (breaks the Parseval loop of wake-R^2)\n",
        "| config | rel-err median | rel-err mean | Pearson r | E_dec | E_ref | n_enc x n_fr |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in res.items():
        lines.append(
            f"| {name} | {m['rel_err_median']:.3f} | {m['rel_err_mean']:.3f} | "
            f"{m['pearson_r']:.3f} | {m['e_dec_mean']:.3f} | {m['e_ref_mean']:.3f} | "
            f"{m['n_enc']}x{m['n_frames']} |"
        )
    lines += [
        "\n## Reading\n",
        "Lower rel-err and higher Pearson r = the decoded field carries the correct "
        "large-scale wake energy. This is an enstrophy computed from the DECODED FIELD "
        "(not read off the supervised 80-d spectrum head), so it is independent of the "
        "wake-R^2 circularity.\n",
    ]
    (REPORT_DIR / "m3_wake_enstrophy.md").write_text("\n".join(lines))

    print("\n[m3] LARGE-SCALE WAKE ENSTROPHY (test_b, recon endpoint):")
    for name, m in res.items():
        print(f"  {name:48s} rel-err(med)={m['rel_err_median']:.3f}  r={m['pearson_r']:.3f}")
    print(f"[m3] wrote {PARTS_DIR / 'm3_wake_enstrophy.json'} and {REPORT_DIR / 'm3_wake_enstrophy.md'}")


if __name__ == "__main__":
    main()
