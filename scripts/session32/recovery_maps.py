"""Track O -- sensor->latent recovery mappings (DIRECTIVE 2).

The windowed wall pressure is a time series (W frames x K taps), so recovery is
offered by three mappings, CV-selected per model+K (encounter-grouped 5-fold),
matching the Session-21 LSTM/KRR CV-selection precedent:

    krr  : StandardScaler -> Nystroem(RBF) -> Ridge on the flattened K*W window
           (Session 31 pressure_infer.fit_pressure_estimator).
    mlp  : torch MLP on the flattened K*W window.
    lstm : PressureLSTM (scripts/session21/exp_pressure_lstm) over the (W, K)
           sequence.

Each mapping exposes ``fit(Xseq, Xflat, Y, groups)`` -> a predictor with
``predict(Xseq, Xflat) -> (n, d)``. Torch mappings standardise pressure (global
scalar) and the target (per-dim), early-stop on an encounter-grouped val slice.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "session21"))

from exp_pressure_lstm import PressureLSTM  # noqa: E402


# --------------------------------------------------------------------------- KRR
class KRRPredictor:
    def __init__(self, est):
        self.est = est

    def predict(self, Xseq, Xflat):
        return np.asarray(self.est.predict(Xflat), dtype=np.float32)


def fit_krr(Xseq, Xflat, Y, groups, *, n_components: int, seed: int):
    from src.evaluation.pressure_infer import fit_pressure_estimator

    est = fit_pressure_estimator(Xflat, Y, n_components=n_components, seed=seed, groups=groups)
    return KRRPredictor(est)


# --------------------------------------------------------------------------- torch
def _grouped_val_split(groups, frac, seed):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    rng.shuffle(ug)
    nval = max(1, int(round(frac * len(ug))))
    val = set(ug[:nval].tolist())
    m = np.array([g in val for g in groups])
    return ~m, m


class _TorchPredictor:
    def __init__(self, net, device, xmu, xsd, ymu, ysd, kind):
        self.net, self.device = net, device
        self.xmu, self.xsd, self.ymu, self.ysd = xmu, xsd, ymu, ysd
        self.kind = kind

    def predict(self, Xseq, Xflat):
        import torch

        X = Xseq if self.kind == "lstm" else Xflat
        Xn = ((X - self.xmu) / self.xsd).astype(np.float32)
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, Xn.shape[0], 8192):
                xb = torch.from_numpy(Xn[i : i + 8192]).to(self.device)
                out.append(self.net(xb).cpu().numpy())
        yh = np.concatenate(out, axis=0)
        return (yh * self.ysd + self.ymu).astype(np.float32)


def _train_torch(net, Xtr, Ytr, Xval, Yval, device, *, epochs, patience, lr, wd, batch, seed):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.MSELoss()
    Xtr_t = torch.from_numpy(Xtr).to(device)
    Ytr_t = torch.from_numpy(Ytr).to(device)
    Xval_t = torch.from_numpy(Xval).to(device)
    Yval_t = torch.from_numpy(Yval).to(device)
    n = Xtr_t.shape[0]
    best, best_state, pat = 1e18, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = lossf(net(Xtr_t[idx]), Ytr_t[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(Xval_t), Yval_t).item()
        if vl < best - 1e-5:
            best, best_state, pat = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            pat += 1
            if pat > patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net


class MLPRecovery:
    def __init__(self, f, d, hidden=256, dropout=0.2):
        import torch.nn as nn

        self.net = nn.Sequential(
            nn.Linear(f, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, d)
        )


def _standardise(Xseq_tr, Xflat_tr, Y_tr, kind):
    xmu = float(np.mean(Xflat_tr))
    xsd = float(np.std(Xflat_tr) + 1e-6)
    ymu = Y_tr.mean(axis=0, keepdims=True).astype(np.float32)
    ysd = (Y_tr.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    return xmu, xsd, ymu, ysd


def fit_mlp(Xseq, Xflat, Y, groups, *, device, seed, epochs=200, patience=25, batch=4096):
    import torch.nn as nn

    tr_m, val_m = _grouped_val_split(groups, 0.15, seed)
    xmu, xsd, ymu, ysd = _standardise(Xseq[tr_m], Xflat[tr_m], Y[tr_m], "mlp")
    f, d = Xflat.shape[1], Y.shape[1]
    net = nn.Sequential(nn.Linear(f, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, d)).to(device)
    Xn = ((Xflat - xmu) / xsd).astype(np.float32)
    Yn = ((Y - ymu) / ysd).astype(np.float32)
    net = _train_torch(
        net,
        Xn[tr_m],
        Yn[tr_m],
        Xn[val_m],
        Yn[val_m],
        device,
        epochs=epochs,
        patience=patience,
        lr=1e-3,
        wd=1e-4,
        batch=batch,
        seed=seed,
    )
    return _TorchPredictor(net, device, xmu, xsd, ymu, ysd, "mlp")


def fit_lstm(Xseq, Xflat, Y, groups, *, device, seed, epochs=200, patience=25, batch=4096):
    tr_m, val_m = _grouped_val_split(groups, 0.15, seed)
    xmu, xsd, ymu, ysd = _standardise(Xseq[tr_m], Xflat[tr_m], Y[tr_m], "lstm")
    k, d = Xseq.shape[2], Y.shape[1]
    net = PressureLSTM(k, d, hidden=48, dropout=0.3).to(device)
    Xn = ((Xseq - xmu) / xsd).astype(np.float32)
    Yn = ((Y - ymu) / ysd).astype(np.float32)
    net = _train_torch(
        net,
        Xn[tr_m],
        Yn[tr_m],
        Xn[val_m],
        Yn[val_m],
        device,
        epochs=epochs,
        patience=patience,
        lr=2e-3,
        wd=1e-4,
        batch=batch,
        seed=seed,
    )
    return _TorchPredictor(net, device, xmu, xsd, ymu, ysd, "lstm")


MAPPINGS = {"krr": fit_krr, "mlp": fit_mlp, "lstm": fit_lstm}
