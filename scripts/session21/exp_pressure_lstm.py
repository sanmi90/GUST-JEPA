"""LSTM pressure-window state estimator, compared to kernel ridge.

The pre-impact pressure window is a time series (30 frames x K taps), so a
recurrent model that consumes the sequence is the natural estimator. We train a
small LSTM (K -> hidden -> JEPA d=64 latent) on the v2 train split with a held-out
validation slice for early stopping, and report held-out state recovery R^2 at
K=2/4/8/16 against the kernel-ridge baseline. RTX 6000.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from src.utils.device import require_rtx6000  # noqa: E402
from _oneoff_baseline_pressure_obs import gather_pressure_and_z  # noqa: E402

LAT = REPO / "outputs/session18/exp_b1/latents_jepa_d64_test1_noBN"
PICKS = json.load(open(REPO / "outputs/session21/pressure_v2/sensor_picks_v2.json"))["TCSI"]
KS = [2, 4, 8, 16]
KRR = {2: 0.80, 4: 0.87, 8: 0.89, 16: 0.84}   # kernel-ridge test_b R2_z (from CSV)


class PressureLSTM(nn.Module):
    def __init__(self, k, d, hidden=48, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(k, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, d))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])


def r2(yh, y):
    return float(np.mean(1 - ((yh - y) ** 2).sum(0) / np.maximum(((y - y.mean(0)) ** 2).sum(0), 1e-9)))


def main():
    dev = require_rtx6000(gpu_index=0)
    print("device:", torch.cuda.get_device_name(dev.index))
    tr = np.load(LAT / "train.npz", allow_pickle=True)
    tb = np.load(LAT / "test_b.npz", allow_pickle=True)
    tc = np.load(LAT / "test_c.npz", allow_pickle=True)
    Xtr, ztr, _ = gather_pressure_and_z(tr, "train")        # (n, 30, 192)
    Xb, zb, _ = gather_pressure_and_z(tb, "test_b")
    Xc, zc, _ = gather_pressure_and_z(tc, "test_c")
    d = ztr.shape[1]
    # standardisation from train
    xmu, xsd = Xtr.mean(), Xtr.std() + 1e-6
    zmu, zsd = ztr.mean(0), ztr.std(0) + 1e-6
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Xtr)); nval = max(20, len(Xtr) // 5)
    vi, ti = perm[:nval], perm[nval:]

    results = {}
    for K in KS:
        s = PICKS[str(K)]
        def prep(X):
            return torch.tensor(((X[:, :, s] - xmu) / xsd), dtype=torch.float32, device=dev)
        Ytr = torch.tensor(((ztr - zmu) / zsd), dtype=torch.float32, device=dev)
        Xall = prep(Xtr)
        Xtr_t, Ytr_t = Xall[ti], Ytr[ti]
        Xval_t, Yval_t = Xall[vi], Ytr[vi]
        Xb_t, Xc_t = prep(Xb), prep(Xc)

        torch.manual_seed(0)
        net = PressureLSTM(K, d).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
        lossf = nn.MSELoss()
        best, best_state, patience = 1e9, None, 0
        for ep in range(600):
            net.train(); opt.zero_grad()
            loss = lossf(net(Xtr_t), Ytr_t); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                vl = lossf(net(Xval_t), Yval_t).item()
            if vl < best - 1e-4:
                best, best_state, patience = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                patience += 1
                if patience > 60:
                    break
        net.load_state_dict(best_state); net.eval()
        with torch.no_grad():
            zb_hat = net(Xb_t).cpu().numpy() * zsd + zmu
            zc_hat = net(Xc_t).cpu().numpy() * zsd + zmu
        rb, rc = r2(zb_hat, zb), r2(zc_hat, zc)
        results[K] = {"lstm_test_b": rb, "lstm_test_c": rc, "krr_test_b": KRR[K]}
        print(f"  K={K:2d}: LSTM test_b R2_z={rb:+.3f} (KRR {KRR[K]:+.2f})  test_c={rc:+.3f}")
    out = REPO / "outputs/session21/pressure_v2/lstm_vs_krr.json"
    json.dump(results, open(out, "w"), indent=0)
    print("wrote", out)


if __name__ == "__main__":
    main()
