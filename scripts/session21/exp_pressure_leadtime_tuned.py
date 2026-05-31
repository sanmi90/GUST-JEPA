"""Tuned lead-time impact prediction: validation-selected LSTM and kernel ridge.

Both estimators are now hyperparameter-tuned on a held-out validation slice (20%
of train) at K=8, lead=0, separately for the state target (impact JEPA d=64
latent) and the scalar lift target (impact C_L). The selected architectures are
then evaluated across leads tau in {0,2,4,6,8}. This replaces the single fixed
LSTM config with a fair, val-selected comparison. RTX 6000.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session21"))
from exp_pressure_leadtime import gather, r2, LEADS  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402  (REPO already on path via gather import)

LAT = REPO / "outputs/session18/exp_b1/latents_jepa_d64_test1_noBN"
OUT = REPO / "outputs/session21/pressure_v2/leadtime.json"
SEL = REPO / "outputs/session21/pressure_v2/leadtime_tuned_configs.json"

LSTM_GRID = list(itertools.product([24, 48, 96], [1, 2], [0.2, 0.4], [1e-3, 2e-3], [1e-4, 1e-3]))
KRR_GRID = list(itertools.product([0.003, 0.01, 0.03, 0.1], [0.01, 0.1, 1.0]))  # (gamma, alpha)


class LSTMNet(nn.Module):
    def __init__(self, k, dout, hidden, layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(k, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, dout))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])


def train_lstm(cfg, Xtr, ytr, Xv, yv, Xte, dev):
    hidden, layers, dropout, lr, wd = cfg
    xmu, xsd = Xtr.mean(), Xtr.std() + 1e-6
    ymu, ysd = ytr.mean(0), ytr.std(0) + 1e-6
    def T(a): return torch.tensor((a - xmu) / xsd, dtype=torch.float32, device=dev)
    Ytr = torch.tensor((ytr.reshape(len(ytr), -1) - ymu) / ysd, dtype=torch.float32, device=dev)
    Yv = torch.tensor((yv.reshape(len(yv), -1) - ymu) / ysd, dtype=torch.float32, device=dev)
    Xtr_t, Xv_t, Xte_t = T(Xtr), T(Xv), T(Xte)
    torch.manual_seed(0)
    net = LSTMNet(Xtr.shape[2], ytr.reshape(len(ytr), -1).shape[1], hidden, layers, dropout).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd); lf = nn.MSELoss()
    best, bs, pat = 1e9, None, 0
    for _ in range(600):
        net.train(); opt.zero_grad(); lf(net(Xtr_t), Ytr).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = lf(net(Xv_t), Yv).item()
        if vl < best - 1e-4: best, bs, pat = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            pat += 1
            if pat > 60: break
    net.load_state_dict(bs); net.eval()
    with torch.no_grad():
        yv_hat = net(Xv_t).cpu().numpy() * ysd + ymu
        yte_hat = net(Xte_t).cpu().numpy() * ysd + ymu
    return r2(yv_hat, yv.reshape(len(yv), -1)), yte_hat


def krr_pred(gamma, alpha, Xtr, ytr, Xte):
    Xtr = Xtr.reshape(len(Xtr), -1); Xte = Xte.reshape(len(Xte), -1)
    sx = StandardScaler().fit(Xtr); sy = StandardScaler().fit(ytr.reshape(len(ytr), -1))
    m = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
    m.fit(sx.transform(Xtr), sy.transform(ytr.reshape(len(ytr), -1)))
    return sy.inverse_transform(m.predict(sx.transform(Xte)))


def main():
    dev = require_rtx6000(gpu_index=0)
    print("device:", torch.cuda.get_device_name(dev.index))
    tr = np.load(LAT / "train.npz", allow_pickle=True)
    tb = np.load(LAT / "test_b.npz", allow_pickle=True)

    # cache pressure/targets per lead (h5 reads once)
    cache = {}
    for lead in LEADS:
        Xtr, ztr, cltr = gather(tr, "train", lead)
        Xb, zb, clb = gather(tb, "test_b", lead)
        cache[lead] = (Xtr, ztr, cltr, Xb, zb, clb)
        print(f"  loaded lead {lead}", flush=True)

    # train/val split (fixed)
    rng = np.random.default_rng(0)
    n = len(cache[0][0]); perm = rng.permutation(n); nv = max(20, n // 5)
    vi, ti = perm[:nv], perm[nv:]

    Xtr0, ztr0, cltr0, _, _, _ = cache[0]
    sel = {}
    # --- tune LSTM (state) and (lift) at lead 0 ---
    for name, y in (("state", ztr0), ("lift", cltr0.reshape(-1, 1))):
        best_vr2, best_cfg = -1e9, None
        for cfg in LSTM_GRID:
            vr2, _ = train_lstm(cfg, Xtr0[ti], y[ti], Xtr0[vi], y[vi], Xtr0[vi], dev)
            if vr2 > best_vr2: best_vr2, best_cfg = vr2, cfg
        sel[f"lstm_{name}"] = {"cfg": best_cfg, "val_r2": best_vr2}
        print(f"  LSTM {name}: {best_cfg} val_R2={best_vr2:+.3f}", flush=True)
    # --- tune KRR (state) and (lift) at lead 0 ---
    for name, y in (("state", ztr0), ("lift", cltr0.reshape(-1, 1))):
        best_vr2, best_cfg = -1e9, None
        for (g, a) in KRR_GRID:
            yhv = krr_pred(g, a, Xtr0[ti], y[ti], Xtr0[vi])
            vr2 = r2(yhv, y[vi].reshape(len(vi), -1))
            if vr2 > best_vr2: best_vr2, best_cfg = vr2, (g, a)
        sel[f"krr_{name}"] = {"cfg": best_cfg, "val_r2": best_vr2}
        print(f"  KRR  {name}: gamma,alpha={best_cfg} val_R2={best_vr2:+.3f}", flush=True)
    json.dump(sel, open(SEL, "w"), indent=0)

    # --- evaluate the selected configs across leads ---
    res = {}
    for lead in LEADS:
        Xtr, ztr, cltr, Xb, zb, clb = cache[lead]
        ok = ~np.isnan(clb)
        _, zk = None, None
        # state
        zk = krr_pred(*sel["krr_state"]["cfg"], Xtr, ztr, Xb)
        _, zl = train_lstm(sel["lstm_state"]["cfg"], Xtr[ti], ztr[ti], Xtr[vi], ztr[vi], Xb, dev)
        # lift
        ck = krr_pred(*sel["krr_lift"]["cfg"], Xtr, cltr.reshape(-1, 1), Xb).ravel()
        _, cl = train_lstm(sel["lstm_lift"]["cfg"], Xtr[ti], cltr[ti].reshape(-1, 1),
                           Xtr[vi], cltr[vi].reshape(-1, 1), Xb, dev)
        cl = cl.ravel()
        res[lead] = dict(
            r2z_krr=r2(zk, zb), r2z_lstm=r2(zl, zb),
            cl_mae_krr=float(np.mean(np.abs(ck[ok] - clb[ok]))), cl_r2_krr=r2(ck[ok], clb[ok]),
            cl_mae_lstm=float(np.mean(np.abs(cl[ok] - clb[ok]))), cl_r2_lstm=r2(cl[ok], clb[ok]))
        print(f"  lead={lead}: R2z KRR={res[lead]['r2z_krr']:+.3f} LSTM={res[lead]['r2z_lstm']:+.3f} | "
              f"C_L MAE KRR={res[lead]['cl_mae_krr']:.3f} LSTM={res[lead]['cl_mae_lstm']:.3f}", flush=True)
    json.dump({str(k): v for k, v in res.items()}, open(OUT, "w"), indent=0)
    print("wrote", OUT, "and", SEL)


if __name__ == "__main__":
    main()
