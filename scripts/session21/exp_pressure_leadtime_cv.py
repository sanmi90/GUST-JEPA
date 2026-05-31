"""Cross-validated estimator selection for the lead-time study (overfitting guard).

The single 45-sample validation split is too noisy to choose among 48 LSTM
configs (the tuned LSTM showed val R^2=0.945 vs test 0.852, and the search picked
the largest model: selection overfitting). Here we select both estimators by
5-fold cross-validation on train (test never touched), which favours configs that
generalise, then evaluate the CV-selected estimators across leads. We print the
CV-mean R^2 next to the test R^2 as the overfitting check. RTX 6000.
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
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session21"))
from exp_pressure_leadtime import gather, r2, LEADS  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402

LAT = REPO / "outputs/session18/exp_b1/latents_jepa_d64_test1_noBN"
OUT = REPO / "outputs/session21/pressure_v2/leadtime.json"
SEL = REPO / "outputs/session21/pressure_v2/leadtime_cv_configs.json"
# smaller, regularisation-leaning grid (guards small-sample overfitting)
LSTM_GRID = list(itertools.product([16, 32, 64], [1], [0.3, 0.5], [1e-3], [1e-3, 3e-3]))
KRR_GRID = list(itertools.product([0.003, 0.01, 0.03, 0.1], [0.1, 1.0, 3.0]))
EPOCHS = 350


class LSTMNet(nn.Module):
    def __init__(self, k, dout, hidden, layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(k, hidden, layers, batch_first=True, dropout=0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, dout))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])


def fit_lstm(cfg, Xtr, ytr, Xte, dev, epochs=EPOCHS):
    hidden, layers, dropout, lr, wd = cfg
    xmu, xsd = Xtr.mean(), Xtr.std() + 1e-6
    ymu, ysd = ytr.mean(0), ytr.std(0) + 1e-6
    Xtr_t = torch.tensor((Xtr - xmu) / xsd, dtype=torch.float32, device=dev)
    Xte_t = torch.tensor((Xte - xmu) / xsd, dtype=torch.float32, device=dev)
    Ytr = torch.tensor((ytr.reshape(len(ytr), -1) - ymu) / ysd, dtype=torch.float32, device=dev)
    torch.manual_seed(0)
    net = LSTMNet(Xtr.shape[2], ytr.reshape(len(ytr), -1).shape[1], hidden, layers, dropout).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd); lf = nn.MSELoss()
    for _ in range(epochs):
        net.train(); opt.zero_grad(); lf(net(Xtr_t), Ytr).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(Xte_t).cpu().numpy() * ysd + ymu


def fit_krr(g, a, Xtr, ytr, Xte):
    Xtr = Xtr.reshape(len(Xtr), -1); Xte = Xte.reshape(len(Xte), -1)
    sx = StandardScaler().fit(Xtr); sy = StandardScaler().fit(ytr.reshape(len(ytr), -1))
    m = KernelRidge(alpha=a, kernel="rbf", gamma=g).fit(sx.transform(Xtr), sy.transform(ytr.reshape(len(ytr), -1)))
    return sy.inverse_transform(m.predict(sx.transform(Xte)))


def cv_select(kind, X, y, dev):
    kf = KFold(5, shuffle=True, random_state=0)
    grid = LSTM_GRID if kind == "lstm" else KRR_GRID
    best = (-1e9, None)
    for cfg in grid:
        scores = []
        for tr_i, te_i in kf.split(X):
            yh = (fit_lstm(cfg, X[tr_i], y[tr_i], X[te_i], dev) if kind == "lstm"
                  else fit_krr(cfg[0], cfg[1], X[tr_i], y[tr_i], X[te_i]))
            scores.append(r2(yh, y[te_i].reshape(len(te_i), -1)))
        m = float(np.mean(scores))
        if m > best[0]: best = (m, cfg)
    return best  # (cv_mean_r2, cfg)


def main():
    dev = require_rtx6000(gpu_index=0)
    print("device:", torch.cuda.get_device_name(dev.index))
    tr = np.load(LAT / "train.npz", allow_pickle=True)
    tb = np.load(LAT / "test_b.npz", allow_pickle=True)
    cache = {l: (gather(tr, "train", l), gather(tb, "test_b", l)) for l in LEADS}
    (Xtr0, ztr0, cltr0), _ = cache[0]

    sel = {}
    for kind in ("lstm", "krr"):
        for name, y in (("state", ztr0), ("lift", cltr0.reshape(-1, 1))):
            cvr2, cfg = cv_select(kind, Xtr0, y, dev)
            sel[f"{kind}_{name}"] = {"cfg": cfg, "cv_r2": cvr2}
            print(f"  {kind} {name}: {cfg}  CV_R2={cvr2:+.3f}", flush=True)
    json.dump(sel, open(SEL, "w"), indent=0)

    res = {}
    for lead in LEADS:
        (Xtr, ztr, cltr), (Xb, zb, clb) = cache[lead]
        ok = ~np.isnan(clb)
        zk = fit_krr(*sel["krr_state"]["cfg"], Xtr, ztr, Xb)
        zl = fit_lstm(sel["lstm_state"]["cfg"], Xtr, ztr, Xb, dev)
        ck = fit_krr(*sel["krr_lift"]["cfg"], Xtr, cltr.reshape(-1, 1), Xb).ravel()
        cl = fit_lstm(sel["lstm_lift"]["cfg"], Xtr, cltr.reshape(-1, 1), Xb, dev).ravel()
        res[lead] = dict(
            r2z_krr=r2(zk, zb), r2z_lstm=r2(zl, zb),
            cl_mae_krr=float(np.mean(np.abs(ck[ok] - clb[ok]))), cl_r2_krr=r2(ck[ok], clb[ok]),
            cl_mae_lstm=float(np.mean(np.abs(cl[ok] - clb[ok]))), cl_r2_lstm=r2(cl[ok], clb[ok]))
        print(f"  lead={lead}: R2z KRR={res[lead]['r2z_krr']:+.3f} LSTM={res[lead]['r2z_lstm']:+.3f} | "
              f"C_L MAE KRR={res[lead]['cl_mae_krr']:.3f} LSTM={res[lead]['cl_mae_lstm']:.3f}", flush=True)
    print(f"  [overfit check] state: LSTM CV_R2={sel['lstm_state']['cv_r2']:+.3f} vs test_b "
          f"R2={res[0]['r2z_lstm']:+.3f}")
    json.dump({str(k): v for k, v in res.items()}, open(OUT, "w"), indent=0)
    print("wrote", OUT, SEL)


if __name__ == "__main__":
    main()
