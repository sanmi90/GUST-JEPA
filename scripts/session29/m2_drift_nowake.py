"""M2: latent-drift diagnostic for the NO-WAKE-HEAD predictive encoder.

Answers "does the wake loss hurt the predictivity of the JEPA latent?" by
comparing the rollout drift of the production wake-supervised JEPA (jepa_d64)
against the matched no-wake-head predictive control (ctrl_pred_vit_nowake_s0),
under the IDENTICAL horizon, reference cloud, covariance estimator, and rollout
protocol used for every family in scripts/session28/drift_ce1.py. We do NOT
reimplement any drift formula: this imports drift_ce1.compute_key, so the
no-wake number is apples-to-apples with the published jepa/fukami rows.

Step 1 (GPU): build the no-wake config's OWN co-trained AR-transformer predictor
from the checkpoint blob (hidden/depth read from the stored args -> strict load),
roll out from the impact-frame latents of the encoded no-wake test_b latents, and
save in the production rollout format (z_dns/z_markov/z_full + metadata).
Step 2 (CPU): drift_ce1.compute_key on the no-wake key, plus jepa_d64 (wake) and
fukami_d64 reference rows.

RTX 6000 only: the device name is asserted to contain RTX + 6000 (the L40S cards
must stay free for asolera). Rollout is light (42 encounters); drift is CPU.

Usage: python scripts/session29/m2_drift_nowake.py --device cuda:3
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "session17"))
sys.path.insert(0, str(REPO / "scripts" / "session28"))

import exp2_rollout_decode_metrics as E2  # noqa: E402
import drift_ce1  # noqa: E402
from src.models.predictor import AutoregressivePredictor  # noqa: E402

T_TOTAL = 120
NOWAKE_KEY = "ctrl_pred_vit_nowake_s0"          # rollout dir + drift key
NOWAKE_LAT = "ctrl_pred_vit_nowake_s0"          # encoded-latent dir name
NOWAKE_CKPT = (
    REPO / "outputs" / "runs" / "session28" / "ctrl_pred_vit_nowake_s0"
    / "encoder" / "checkpoint_iter020000.pt"
)
LAT_DIR = REPO / "outputs" / "session28" / "latents"
ROLL_DIR = REPO / "outputs" / "session28" / "rollouts"
PARTS_DIR = REPO / "outputs" / "session28" / "numbers_parts"
REPORT_DIR = REPO / "outputs" / "session29" / "reports"
SOURCE = "m2_drift_nowake (no-wake-head predictivity check)"


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip()
    except Exception:
        return "unknown"


def _assert_rtx6000(device: torch.device) -> None:
    name = torch.cuda.get_device_name(device.index if device.index is not None else 0)
    if not ("RTX" in name and "6000" in name):
        raise SystemExit(f"[m2] refusing non-RTX-6000 device {device}: {name!r}")
    print(f"[m2] device {device} = {name}")


def load_nowake_predictor(device: torch.device):
    ck = torch.load(NOWAKE_CKPT, map_location="cpu", weights_only=False)
    psd = {k[len("predictor."):]: v for k, v in ck["jepa_state_dict"].items()
           if k.startswith("predictor.")}
    # Infer dims from the state dict (the stored args are misleading): embed.weight
    # is (hidden, latent); depth = number of distinct blocks.N.* indices.
    hidden = int(psd["embed.weight"].shape[0])
    depth = 1 + max(int(k.split(".")[1]) for k in psd if k.startswith("blocks."))
    p = AutoregressivePredictor(latent_dim=64, cond_dim=0, hidden_dim=hidden,
                                depth=depth, max_seq_len=32)
    p.load_state_dict(psd, strict=True)
    p.eval().to(device)
    print(f"[m2] no-wake predictor: transformer hidden={hidden} depth={depth}, "
          f"{len(psd)} tensors, strict load OK")
    return p


@torch.no_grad()
def gen_rollout(pred, device) -> dict:
    b = np.load(LAT_DIR / NOWAKE_LAT / "test_b.npz", allow_pickle=True)
    zf = b["z_full"].astype(np.float32)                      # (n,120,d) raw latents = z_dns
    impact = b["impact_frame"].astype(np.int64)
    n, T, d = zf.shape
    z_dns = torch.from_numpy(zf).to(device)
    z_markov = z_dns.clone()
    z_fullc = z_dns.clone()
    cond0 = torch.zeros(1, 0, device=device)
    for i in range(n):
        ti = int(impact[i]); steps = T_TOTAL - ti - 1
        if steps <= 0:
            continue
        z_init = z_dns[i, ti:ti + 1]
        zm = E2.rollout_markov_only(pred, z_init, cond0, steps, device)
        z_markov[i, ti + 1:T_TOTAL] = zm[0, 1:steps + 1].float()
        lo = max(0, ti + 1 - int(pred.max_seq_len)); seed = z_dns[i, lo:ti + 1]
        zfu = E2.rollout_autoregressive(pred, seed, cond0, steps, device)
        z_fullc[i, ti + 1:T_TOTAL] = zfu[0, -steps:].float()
    out = {"z_dns": zf, "z_markov": z_markov.cpu().numpy(), "z_full": z_fullc.cpu().numpy()}
    for k in ("G", "D", "Y", "impact_frame"):
        out[k] = b[k]
    out["case_ids"] = b["case_ids"] if "case_ids" in b.files else b["case_id"]
    out["encounter_indices"] = (b["encounter_indices"] if "encounter_indices" in b.files
                                else b["encounter_index"])
    return out


def headline(fam_key: str) -> dict:
    r = drift_ce1.compute_key(fam_key)
    hidx = drift_ce1.HORIZONS.index(drift_ce1.HEADLINE_H)
    return {
        "rel_l2": drift_ce1.nanmean(r["rel_l2"][:, hidx]),
        "maha_ratio": drift_ce1.nanmean(r["maha_ratio"][:, hidx]),
        "near_null_frac": r["spec_near_null_departure_frac"],
        "n_null_dirs": r["spec_n_null_dirs"],
        "enc_topk_var": float(r["encoded_topk_var_frac"][hidx]),
        "n_enc": r["n_enc"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:3")
    args = ap.parse_args()
    device = torch.device(args.device)
    _assert_rtx6000(device)

    roll_out = ROLL_DIR / NOWAKE_KEY
    roll_npz = roll_out / "test_b.npz"
    if not roll_npz.exists():
        print(f"[m2] generating no-wake rollout -> {roll_npz}")
        pred = load_nowake_predictor(device)
        res = gen_rollout(pred, device)
        roll_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(roll_npz, **res)
        print(f"[m2] wrote {roll_npz}  z_full{res['z_full'].shape}")
    else:
        print(f"[m2] reusing existing rollout {roll_npz}")

    # CPU drift, identical protocol to drift_ce1.
    drift_ce1.PRED_LAT[NOWAKE_KEY] = NOWAKE_LAT
    H = drift_ce1.HEADLINE_H
    hidx = drift_ce1.HORIZONS.index(H)
    r = drift_ce1.compute_key(NOWAKE_KEY)
    nowake = {
        "rel_l2": drift_ce1.nanmean(r["rel_l2"][:, hidx]),
        "maha_ratio": drift_ce1.nanmean(r["maha_ratio"][:, hidx]),
        "near_null_frac": r["spec_near_null_departure_frac"],
        "n_null_dirs": r["spec_n_null_dirs"],
        "enc_topk_var": float(r["encoded_topk_var_frac"][hidx]),
        "n_enc": r["n_enc"],
    }
    jepa = headline("jepa_d64")       # wake-supervised production
    fukami = headline("fukami_d64")   # reconstructive reference

    part = {
        "part": "m2_drift_nowake",
        "numbers": {
            "drift_rel_l2_H24_d64_nowake": {
                "macro": "DriftRelLtwoNowake", "value": nowake["rel_l2"], "fmt": "%.2f",
                "n": nowake["n_enc"], "split": "test_b", "horizon": H, "source": SOURCE,
                "note": ("mean rel-l2 rollout deviation, no-wake-head predictive d=64, "
                         f"own co-trained predictor, H={H}; identical protocol to drift_ce1"),
            },
            "drift_maha_ratio_H24_d64_nowake": {
                "macro": "DriftMahaNowake", "value": nowake["maha_ratio"], "fmt": "%.2f",
                "n": nowake["n_enc"], "split": "test_b", "horizon": H, "source": SOURCE,
                "note": ("Mahalanobis ratio to per-frame TRAIN encoded cloud (v2 floored), "
                         f"no-wake-head predictive d=64, H={H}"),
            },
            "drift_spec_nearnull_frac_d64_nowake": {
                "macro": "DriftSpecNearNullNowake", "value": nowake["near_null_frac"],
                "fmt": "%.2f", "split": "test_b", "horizon": H, "source": SOURCE,
                "note": (f"near-null departure fraction ({nowake['n_null_dirs']} dirs), "
                         "no-wake-head predictive d=64"),
            },
        },
    }
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    (PARTS_DIR / "m2_drift_nowake.json").write_text(json.dumps(part, indent=2, sort_keys=True))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = [
        ("JEPA wake-supervised (production)", jepa),
        ("JEPA NO wake head (control)", nowake),
        ("Fukami reconstructive (ref)", fukami),
    ]
    lines = [
        "# M2: does the wake loss hurt latent predictivity? (drift, wake vs no-wake)\n",
        f"- git sha: `{_git_sha()}`",
        f"- UTC: {ts}",
        "- command: `python scripts/session29/m2_drift_nowake.py`",
        f"- horizon H={H}, d=64, split=test_b, full-context rollout; protocol = drift_ce1.py",
        f"- no-wake rollout: `{roll_npz}` (own co-trained predictor)",
        f"- part: `{PARTS_DIR / 'm2_drift_nowake.json'}`\n",
        "## Drift comparison (same protocol, H=24, d=64, test_b)\n",
        "| family | rel-l2 (Fig-5) | Maha ratio | near-null dep frac | n null dirs | enc top-5 var | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in rows:
        lines.append(
            f"| {name} | {m['rel_l2']:.3f} | {m['maha_ratio']:.3f} | "
            f"{m['near_null_frac']:.3f} | {m['n_null_dirs']} | {m['enc_topk_var']:.3f} | {m['n_enc']} |"
        )
    lines += [
        "\n## Reading\n",
        "If the no-wake control's rel-l2 and Mahalanobis ratio are close to the "
        "wake-supervised JEPA's, the wake loss does NOT degrade the latent's "
        "dynamical/predictive quality (the supervision is 'free' predictively). If "
        "the no-wake drift is markedly lower, the wake head trades predictive "
        "tightness for readability; if higher, it does not even help drift. "
        "Both stay on-manifold (low Maha ratio) only if anti-collapse holds.\n",
    ]
    (REPORT_DIR / "m2_drift_nowake.md").write_text("\n".join(lines))

    print("\n[m2] DRIFT COMPARISON (H=24, d=64, test_b):")
    for name, m in rows:
        print(f"  {name:38s} rel-l2={m['rel_l2']:.3f}  Maha={m['maha_ratio']:.3f}  "
              f"nearnull={m['near_null_frac']:.3f}")
    print(f"[m2] wrote {PARTS_DIR / 'm2_drift_nowake.json'} and {REPORT_DIR / 'm2_drift_nowake.md'}")


if __name__ == "__main__":
    main()
