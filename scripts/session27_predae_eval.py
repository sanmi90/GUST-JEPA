"""Closure eval for the PredAE: encode with the PredAE's own Fukami encoder, roll
its own LSTM, probe wake-enstrophy R^2@16 (markov + fullctx). Same canonical probe
as everything else. Compares predictive-PIXEL (PredAE) to predictive-LATENT (JEPA).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import h5py
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "session20"))
import exp_closure_r2 as cr  # noqa: E402
from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.baselines.fukami_ae import FukamiCNNEncoder  # noqa: E402
from src.models.predictor import LSTMLatentPredictor  # noqa: E402

CACHE = REPO.parent / "PREVENT/data/processed/vortex-jepa/v2"
PIPE = REPO / "outputs/data_pipeline/v1/manifest.json"
MAX_SEQ, H, HMAX = 32, 16, 18


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/session27/predae_d64/checkpoint_iter015000.pt")
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    device = torch.device(args.device)
    ck = torch.load(REPO / args.ckpt, map_location="cpu", weights_only=False)
    d = int(ck.get("d", 64))
    enc = FukamiCNNEncoder(d); enc.load_state_dict(ck["enc"]); enc.eval().to(device)
    lstm = LSTMLatentPredictor(d, cond_dim=0, hidden_dim=256, num_layers=3, max_seq_len=MAX_SEQ)
    lstm.load_state_dict(ck["lstm"]); lstm.eval().to(device)
    pipe = OmegaPipeline.from_manifest(PIPE)
    dns = np.load(cr.DNS_METRICS_PATH, allow_pickle=True)

    @torch.no_grad()
    def encode_split(split):
        cids = [str(c) for c in dns[f"{split}_case_id"]]; eis = [int(e) for e in dns[f"{split}_encounter_index"]]
        Z, imp, keep = [], [], []
        for i, (cid, ei) in enumerate(zip(cids, eis)):
            f = CACHE / cid / f"encounter_{ei:02d}.h5"
            if not f.exists():
                continue
            with h5py.File(f, "r") as h:
                om = np.asarray(h["omega_z"], np.float32); ip = int(h.attrs.get("impact_frame_estimate", 40))
            on = pipe.normalize(pipe.preprocess_raw(om, cid, ei))           # (120,H,W)
            x = torch.from_numpy(on).unsqueeze(1).to(device)               # (120,1,H,W)
            with torch.autocast("cuda", torch.bfloat16):
                z = enc(x)
            Z.append(z.float().cpu().numpy()); imp.append(ip); keep.append(i)
        return np.stack(Z), np.array(imp), np.array(keep)

    ztr, _, ktr = encode_split("train")
    probe = cr.fit_ridge(ztr.reshape(-1, d),
                         dns["train_wake_enstrophy"][ktr].reshape(-1))

    @torch.no_grad()
    def free_run(zseed, steps):
        zf = zseed
        for _ in range(steps):
            zf = torch.cat([zf, lstm(zf[:, -MAX_SEQ:], None)[:, -1:, :]], dim=1)
        return zf

    def r2(yp, yt):
        return 1.0 - float(((yp - yt) ** 2).sum()) / max(float(((yt - yt.mean()) ** 2).sum()), 1e-12)

    print(f"  [train latent scale] mean|z|={np.mean(np.linalg.norm(ztr.reshape(-1, d), axis=1)):.3f}")
    print("=== PredAE own-LSTM wake R2@16 (dns=ceiling, no rollout) ===")
    for split in ("test_b", "test_c"):
        Z, imp, keep = encode_split(split)
        Zt = torch.from_numpy(Z).to(device)
        for mode in ("dns", "markov", "fullctx"):
            yp, yt, nrm = [], [], []
            for j in range(len(keep)):
                if mode == "dns":
                    zr = Zt[j, imp[j] + H].cpu().numpy()
                elif mode == "markov":
                    seed = Zt[j, imp[j]:imp[j] + 1].unsqueeze(0)
                    zr = free_run(seed, HMAX)[0, H].cpu().numpy()
                else:
                    lo = max(0, imp[j] + 1 - MAX_SEQ); seed = Zt[j, lo:imp[j] + 1].unsqueeze(0)
                    zr = free_run(seed, HMAX)[0, seed.shape[1] - 1 + H].cpu().numpy()
                nrm.append(float(np.linalg.norm(zr)))
                yp.append(float(cr.apply_probe(zr[None], probe)[0]))
                yt.append(float(dns[f"{split}_wake_enstrophy"][keep[j], imp[j] + H]))
            print(f"  {split:7s} {mode:8s} R2={r2(np.array(yp), np.array(yt)):+.3f}  mean|z|={np.mean(nrm):.2f}")
    print("Reference: JEPA predictive-latent own-predictor test_b ~0.45-0.53; "
          "cond-0 closure on frozen Fukami latents -0.72.")


if __name__ == "__main__":
    main()
