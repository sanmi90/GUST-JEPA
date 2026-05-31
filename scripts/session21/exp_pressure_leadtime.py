"""Predicting the impact before it occurs, from pre-impact wall pressure.

The pressure window ends tau frames BEFORE impact; we predict the impact-frame
predictive latent and the impact lift C_L from it, for tau in {0,2,4,6,8} (up to
8 instants ahead). Estimators: kernel ridge and an LSTM over the pre-impact
window. This isolates how early the imminent impact state and load are readable
from sparse wall sensors. K=8 TCSI taps, v2 production JEPA d=64 latent. RTX 6000.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "session20"))
from src.utils.device import require_rtx6000  # noqa: E402
from exp_closure_r2 import match_index  # noqa: E402
from exp_pressure_lstm import PressureLSTM  # noqa: E402

LAT = REPO / "outputs/session18/exp_b1/latents_jepa_d64_test1_noBN"
CACHE = Path("/home/carlos/PREVENT/data/processed/vortex-jepa/v1")
DNS = np.load(REPO / "outputs/session17/exp2/dns_physical_metrics.npz", allow_pickle=True)
TCSI8 = json.load(open(REPO / "outputs/session21/pressure_v2/sensor_picks_v2.json"))["TCSI"]["8"]
WINDOW = 30
LEADS = [0, 2, 4, 6, 8]
OUT = REPO / "outputs/session21/pressure_v2/leadtime.json"


def r2(yh, y):
    return float(np.mean(1 - ((yh - y) ** 2).sum(0) / np.maximum(((y - y.mean(0)) ** 2).sum(0), 1e-9)))


def gather(npz, split, lead):
    """Pressure window (n, WINDOW, K) ending at impact-lead; impact z; impact C_L."""
    cid = (npz["case_id"] if "case_id" in npz.files else npz["case_ids"]).astype(str)
    ei = (npz["encounter_index"] if "encounter_index" in npz.files else npz["encounter_indices"]).astype(int)
    imp = npz["impact_frame"].astype(int)
    z = npz["z"].astype(np.float64)
    di = match_index(cid, ei, DNS[f"{split}_case_id"], DNS[f"{split}_encounter_index"])
    X = np.zeros((len(cid), WINDOW, len(TCSI8)), np.float64)
    cl = np.full(len(cid), np.nan)
    for i in range(len(cid)):
        end = int(imp[i]) - lead
        with h5py.File(CACHE / cid[i] / f"encounter_{int(ei[i]):02d}.h5") as f:
            pw = np.asarray(f["p_wall"], np.float32)[:, TCSI8]    # (120, K)
        w = pw[max(0, end - WINDOW):end]
        if w.shape[0] < WINDOW:
            w = np.vstack([np.zeros((WINDOW - w.shape[0], len(TCSI8)), np.float32), w])
        X[i] = w
        if di[i] >= 0:
            cl[i] = DNS[f"{split}_C_L"][di[i], int(imp[i])]
    return X, z, cl


def krr(Xtr, ytr, Xte):
    sx = StandardScaler().fit(Xtr); sy = StandardScaler().fit(ytr.reshape(len(ytr), -1))
    m = KernelRidge(alpha=0.1, kernel="rbf", gamma=0.01)
    m.fit(sx.transform(Xtr), sy.transform(ytr.reshape(len(ytr), -1)))
    return sy.inverse_transform(m.predict(sx.transform(Xte)))


def lstm_fit(Xtr, ytr, Xte, dev, dout):
    xmu, xsd = Xtr.mean(), Xtr.std() + 1e-6
    ymu, ysd = ytr.mean(0), ytr.std(0) + 1e-6
    rng = np.random.default_rng(0); perm = rng.permutation(len(Xtr)); nv = max(20, len(Xtr) // 5)
    vi, ti = perm[:nv], perm[nv:]
    def t(a): return torch.tensor((a - xmu) / xsd, dtype=torch.float32, device=dev)
    Y = torch.tensor((ytr.reshape(len(ytr), -1) - ymu) / ysd, dtype=torch.float32, device=dev)
    Xtr_t, Xv_t, Xte_t = t(Xtr[ti]), t(Xtr[vi]), t(Xte)
    torch.manual_seed(0)
    net = PressureLSTM(Xtr.shape[2], dout).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4); lf = nn.MSELoss()
    best, bs, pat = 1e9, None, 0
    for _ in range(600):
        net.train(); opt.zero_grad(); lf(net(Xtr_t), Y[ti]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = lf(net(Xv_t), Y[vi]).item()
        if vl < best - 1e-4: best, bs, pat = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            pat += 1
            if pat > 60: break
    net.load_state_dict(bs); net.eval()
    with torch.no_grad():
        return net(Xte_t).cpu().numpy() * ysd + ymu


def main():
    dev = require_rtx6000(gpu_index=0)
    print("device:", torch.cuda.get_device_name(dev.index))
    tr = np.load(LAT / "train.npz", allow_pickle=True)
    tb = np.load(LAT / "test_b.npz", allow_pickle=True)
    res = {}
    for lead in LEADS:
        Xtr, ztr, cltr = gather(tr, "train", lead)
        Xb, zb, clb = gather(tb, "test_b", lead)
        Xtr_f, Xb_f = Xtr.reshape(len(Xtr), -1), Xb.reshape(len(Xb), -1)
        # state z: KRR + LSTM
        r2_krr = r2(krr(Xtr_f, ztr, Xb_f), zb)
        r2_lstm = r2(lstm_fit(Xtr, ztr, Xb, dev, ztr.shape[1]), zb)
        # impact C_L: KRR + LSTM
        clk = krr(Xtr_f, cltr, Xb_f).ravel()
        cll = lstm_fit(Xtr, cltr.reshape(-1, 1), Xb, dev, 1).ravel()
        ok = ~np.isnan(clb)
        res[lead] = dict(
            r2z_krr=r2_krr, r2z_lstm=r2_lstm,
            cl_mae_krr=float(np.mean(np.abs(clk[ok] - clb[ok]))),
            cl_r2_krr=r2(clk[ok], clb[ok]),
            cl_mae_lstm=float(np.mean(np.abs(cll[ok] - clb[ok]))),
            cl_r2_lstm=r2(cll[ok], clb[ok]))
        print(f"  lead={lead}: R2z KRR={r2_krr:+.3f} LSTM={r2_lstm:+.3f} | "
              f"C_L MAE KRR={res[lead]['cl_mae_krr']:.3f} (R2={res[lead]['cl_r2_krr']:+.2f}) "
              f"LSTM={res[lead]['cl_mae_lstm']:.3f}", flush=True)
    json.dump({str(k): v for k, v in res.items()}, open(OUT, "w"), indent=0)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
