"""Roll a JEPA's OWN co-trained predictor (from the encoder checkpoint blob) over
its frozen latents, in the production rollout format (z_full). This is the
end-to-end model's actual forecaster, as opposed to a bolted-on matched
predictor. Predictor dims are inferred from the state dict (the stored args are
unreliable). RTX 6000 only.

Usage: roll_own_predictor.py <encoder_ckpt> <test_b_latents.npz> <out_dir> [--device cuda:3]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "session17"))
import exp2_rollout_decode_metrics as E2  # noqa: E402
from src.models.predictor import AutoregressivePredictor  # noqa: E402

T_TOTAL = 120


def assert_rtx(dev):
    name = torch.cuda.get_device_name(dev.index if dev.index is not None else 0)
    if not ("RTX" in name and "6000" in name):
        raise SystemExit(f"[roll-own] refusing non-RTX device {dev}: {name!r}")
    print(f"[roll-own] device {dev} = {name}")


def load_own_predictor(ckpt, device):
    b = torch.load(ckpt, map_location="cpu", weights_only=False)
    psd = {k[len("predictor."):]: v for k, v in b["jepa_state_dict"].items()
           if k.startswith("predictor.")}
    hidden = int(psd["embed.weight"].shape[0])
    depth = 1 + max(int(k.split(".")[1]) for k in psd if k.startswith("blocks."))
    latent = int(psd["embed.weight"].shape[1])
    p = AutoregressivePredictor(latent_dim=latent, cond_dim=0, hidden_dim=hidden,
                                depth=depth, max_seq_len=32)
    p.load_state_dict(psd, strict=True)
    p.eval().to(device)
    print(f"[roll-own] predictor d={latent} hidden={hidden} depth={depth} (strict load OK)")
    return p


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("latents"); ap.add_argument("out_dir")
    ap.add_argument("--device", default="cuda:3")
    a = ap.parse_args()
    dev = torch.device(a.device); assert_rtx(dev)
    pred = load_own_predictor(a.ckpt, dev)

    b = np.load(a.latents, allow_pickle=True)
    zf = b["z_full"].astype(np.float32)
    cid = b["case_ids"] if "case_ids" in b.files else b["case_id"]
    enc = b["encounter_indices"] if "encounter_indices" in b.files else b["encounter_index"]
    imp = b["impact_frame"].astype(np.int64)
    z_dns = torch.from_numpy(zf).to(dev)
    z_full = z_dns.clone()
    cond0 = torch.zeros(1, 0, device=dev)
    for i in range(zf.shape[0]):
        ti = int(imp[i]); steps = T_TOTAL - ti - 1
        if steps <= 0:
            continue
        lo = max(0, ti + 1 - int(pred.max_seq_len)); seed = z_dns[i, lo:ti + 1]
        zfu = E2.rollout_autoregressive(pred, seed, cond0, steps, dev)
        z_full[i, ti + 1:T_TOTAL] = zfu[0, -steps:].float()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "test_b.npz", z_full=z_full.cpu().numpy(), z_dns=zf,
                        impact_frame=imp, case_ids=np.asarray(cid),
                        encounter_indices=np.asarray(enc))
    print(f"[roll-own] wrote {out/'test_b.npz'}  z_full{tuple(z_full.shape)}")


if __name__ == "__main__":
    main()
