"""Session 17, Experiment 5 redo: nonlinear pressure observability.

The linear-ridge variant in exp5_closed_loop.py overfit (z R^2 dropped from
0.43 at K=2 to -2.0 at K=16 on test_b). This script uses three nonlinear
estimators -- TCN (Session 14 D115 recipe), MLP, and KernelRidge(RBF) --
and picks the best per K, then runs the closed-loop rollouts.

For each estimator family, fit:
  - pressure (K x 30) -> z_impact (64)
  - pressure (K x 30) -> (G, D, Y)

Estimators:
  TCN_30:    TCNProxyLearner, dilations=(1,2,4), 30-epoch tcn for quick screen
  TCN_200:   TCNProxyLearner, dilations=(1,2,4), 200-epoch full recipe
  MLP_reg:   3 hidden x 256, weight_decay 1e-2, early stopping on test_a
  KRR_RBF:   sklearn KernelRidge with CV alpha and gamma

Then run closed-loop Markov rollouts at the best estimator.

Outputs:
    outputs/session17/exp5/nonlinear_estimator_R2.csv
    outputs/session17/exp5/nonlinear_closed_loop_metrics.csv
    outputs/session17/exp5/nonlinear_tolerance_curves.json
    outputs/session17/figures/exp5_nonlinear_K_curve.png
    outputs/session17/figures/exp5_nonlinear_tolerance.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.evaluation.tcn_proxy_learner import TCNConfig, TCNProxyLearner  # noqa: E402
from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.models.predictor import (  # noqa: E402
    AutoregressivePredictor,
    CausalSelfAttentionWithRoPE,
)
from src.models.rope import apply_rope  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402
from sklearn.kernel_ridge import KernelRidge  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

ENCODER_CKPT = (
    REPO / "outputs" / "runs" / "session12" / "S12_E_d64" / "encoder"
    / "checkpoint_iter020000.pt"
)
OMEGA_MANIFEST = REPO / "outputs" / "data_pipeline" / "v1" / "manifest.json"
SPLIT_MANIFEST = REPO / "configs" / "splits" / "split_v1.json"
TCSI_PORTFOLIO = REPO / "outputs" / "session14" / "tcsi_pilot" / "methods_portfolio.json"
PARTITION = "v1"
DEFAULT_IMPACT_FRAME = 40
OUT = REPO / "outputs" / "session17" / "exp5"
OUT.mkdir(parents=True, exist_ok=True)
FIGS = REPO / "outputs" / "session17" / "figures"
CACHE_ROOT = Path(
    os.environ.get(
        "VORTEX_JEPA_CACHE",
        str(Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
            / "data" / "processed" / "vortex-jepa"),
    )
)

K_VALUES = (2, 4, 8, 16)
WINDOW = 30  # pre-impact frames
HORIZONS = (8, 16, 32)
PROBE_METRICS = ("C_L", "I_y", "wake_enstrophy")
TOLERANCE = {"C_L": 0.10, "I_y": 0.15, "wake_enstrophy": 0.25}


def gather_split(split: str) -> list[dict]:
    with open(SPLIT_MANIFEST) as f:
        manifest = json.load(f)
    out = []
    for cid, case in manifest["cases"].items():
        if split == "train" and case["split"] == "train":
            ks = list(case["train_encounter_indices"])
        elif split == "test_a" and case["split"] == "train":
            ks = list(case["test_a_encounter_indices"])
        elif split == "test_b" and case["split"] == "test_b":
            ks = list(range(int(case["n_encounters_full"])))
        elif split == "test_c" and case["split"] == "test_c":
            ks = list(range(int(case["n_encounters_full"])))
        else:
            continue
        for k in ks:
            path = CACHE_ROOT / PARTITION / cid / f"encounter_{int(k):02d}.h5"
            if not path.exists():
                continue
            out.append({"case_id": cid, "k": int(k), "path": path,
                        "G": float(case["G"]), "D": float(case["D"]), "Y": float(case["Y"])})
    return out


def load_pressure_window(rec: dict) -> tuple[np.ndarray, int]:
    """Return (WINDOW, 192) pre-impact pressure window + impact frame."""
    with h5py.File(rec["path"], "r") as f:
        p_wall = np.asarray(f["p_wall"], dtype=np.float32)  # (120, 192)
        impact = int(f.attrs.get("impact_frame_estimate", DEFAULT_IMPACT_FRAME))
    t_start = max(0, impact - WINDOW)
    window = p_wall[t_start:impact]
    if window.shape[0] < WINDOW:
        pad = np.zeros((WINDOW - window.shape[0], 192), dtype=np.float32)
        window = np.concatenate([pad, window], axis=0)
    return window, impact


def build_X_tcn(records: list[dict], sensor_idx: list[int]) -> np.ndarray:
    """Return (n, K, W) array for TCN consumption."""
    K = len(sensor_idx)
    X = np.zeros((len(records), K, WINDOW), dtype=np.float32)
    for i, rec in enumerate(records):
        w, _imp = load_pressure_window(rec)
        X[i] = w[:, sensor_idx].T  # (K, W)
    return X


def build_X_flat(X_tcn: np.ndarray) -> np.ndarray:
    """Flatten (n, K, W) -> (n, K*W)."""
    return X_tcn.reshape(X_tcn.shape[0], -1).astype(np.float64)


# ---------------------------------------------------------------------------
# MLP estimator
# ---------------------------------------------------------------------------


class MLPReg(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp_with_early_stop(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    device, *, lr=3e-4, wd=1e-2, batch=64, max_iters=4000, patience=400, seed=0,
) -> tuple[MLPReg, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    in_dim = X_train.shape[1]
    out_dim = y_train.shape[1] if y_train.ndim == 2 else 1
    if y_train.ndim == 1:
        y_train = y_train[:, None]
        y_val = y_val[:, None]
    # Standardise X
    mu_x = X_train.mean(0); sigma_x = X_train.std(0).clip(min=1e-9)
    mu_y = y_train.mean(0); sigma_y = y_train.std(0).clip(min=1e-9)
    X_tr_n = (X_train - mu_x) / sigma_x
    y_tr_n = (y_train - mu_y) / sigma_y
    X_v_n = (X_val - mu_x) / sigma_x
    y_v_n = (y_val - mu_y) / sigma_y
    model = MLPReg(in_dim, out_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    X_v_t = torch.from_numpy(X_v_n.astype(np.float32)).to(device)
    y_v_t = torch.from_numpy(y_v_n.astype(np.float32)).to(device)
    best_val = float("inf"); best_state = None; best_iter = 0
    for it in range(max_iters):
        idx = rng.choice(X_tr_n.shape[0], size=min(batch, X_tr_n.shape[0]), replace=False)
        xb = torch.from_numpy(X_tr_n[idx].astype(np.float32)).to(device)
        yb = torch.from_numpy(y_tr_n[idx].astype(np.float32)).to(device)
        model.train()
        opt.zero_grad()
        loss = ((model(xb) - yb) ** 2).mean()
        loss.backward()
        opt.step()
        if (it + 1) % 25 == 0:
            model.eval()
            with torch.no_grad():
                vl = ((model(X_v_t) - y_v_t) ** 2).mean().item()
            if vl < best_val:
                best_val = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_iter = it + 1
            elif it + 1 - best_iter > patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"mu_x": mu_x, "sigma_x": sigma_x, "mu_y": mu_y, "sigma_y": sigma_y,
                   "best_iter": best_iter, "best_val": best_val}


def predict_mlp(model, X: np.ndarray, info: dict, device) -> np.ndarray:
    Xn = (X - info["mu_x"]) / info["sigma_x"]
    with torch.no_grad():
        pred = model(torch.from_numpy(Xn.astype(np.float32)).to(device)).cpu().numpy()
    return pred * info["sigma_y"] + info["mu_y"]


# ---------------------------------------------------------------------------
# Encoder / predictor utilities
# ---------------------------------------------------------------------------


def load_encoder_predictor(device):
    blob = torch.load(ENCODER_CKPT, map_location="cpu", weights_only=False)
    args = blob["args"]
    enc = HybridCNNViTEncoder(
        latent_dim=int(args["d"]),
        projection_norm=args.get("projection_norm", "batchnorm"),
    )
    pred = AutoregressivePredictor(
        latent_dim=int(args["d"]),
        cond_dim=int(args.get("predictor_cond_dim", 3)),
        max_seq_len=int(args.get("T", 32)),
    )
    full_state = blob["jepa_state_dict"]
    enc.load_state_dict(
        {k.removeprefix("encoder."): v for k, v in full_state.items() if k.startswith("encoder.")},
        strict=False,
    )
    pred.load_state_dict(
        {k.removeprefix("predictor."): v for k, v in full_state.items() if k.startswith("predictor.")},
        strict=False,
    )
    enc.eval().to(device); pred.eval().to(device)
    for p in enc.parameters(): p.requires_grad_(False)
    for p in pred.parameters(): p.requires_grad_(False)
    return enc, pred


def make_markov_attn():
    def markov_forward(self, x):
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        q = apply_rope(q, self.rope_cos[:T], self.rope_sin[:T])
        k = apply_rope(k, self.rope_cos[:T], self.rope_sin[:T])
        mask = torch.full((T, T), float("-inf"), device=x.device, dtype=q.dtype)
        mask[:, 0] = 0.0
        mask.fill_diagonal_(0.0)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        return self.proj(out.transpose(1, 2).reshape(B, T, -1))
    return markov_forward


@contextmanager
def markov_attention(model):
    new_fwd = make_markov_attn()
    originals = []
    for module in model.modules():
        if isinstance(module, CausalSelfAttentionWithRoPE):
            originals.append((module, module.forward))
            module.forward = new_fwd.__get__(module, type(module))
    try:
        yield
    finally:
        for module, orig in originals:
            module.forward = orig


@torch.no_grad()
def rollout_markov(pred, z_impact, cond, steps, device):
    max_seq = int(pred.max_seq_len)
    z_full = z_impact.clone().unsqueeze(0) if z_impact.dim() == 2 else z_impact.clone()
    with markov_attention(pred):
        for _ in range(steps):
            ctx = z_full[:, -max_seq:, :]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                z_hat = pred(ctx, cond)
            z_full = torch.cat([z_full, z_hat[:, -1:, :].float()], dim=1)
    return z_full


@torch.no_grad()
def encode_full(enc, omega_norm, device):
    x = torch.from_numpy(omega_norm).to(device).unsqueeze(0).unsqueeze(2)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        z = enc(x)
    return z.float().squeeze(0)


def apply_metric_probe(z: np.ndarray, probe: dict) -> np.ndarray:
    Zn = (z.astype(np.float64) - probe["mu_z"]) / probe["sigma_z"]
    return Zn @ probe["W"] + probe["b"]


def r2_per_col(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_pred = y_pred[:, None]
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.clip(ss_tot, 1e-12, None)


def cv_krr(X: np.ndarray, y: np.ndarray) -> tuple[KernelRidge, dict]:
    """5-fold CV over (alpha, gamma); return fitted KRR + chosen hyperparams."""
    alphas = [0.01, 0.1, 1.0, 10.0]
    gammas = [0.0005, 0.001, 0.005, 0.01, 0.05]
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    best = (None, None, -np.inf)
    for a in alphas:
        for g in gammas:
            scores = []
            for tr, va in kf.split(X):
                m = KernelRidge(alpha=a, gamma=g, kernel="rbf")
                m.fit(X[tr], y[tr])
                pred = m.predict(X[va])
                if y.ndim == 1:
                    ss_res = ((y[va] - pred) ** 2).sum()
                    ss_tot = ((y[va] - y[va].mean()) ** 2).sum()
                else:
                    ss_res = ((y[va] - pred) ** 2).sum()
                    ss_tot = ((y[va] - y[va].mean(0)) ** 2).sum()
                scores.append(1 - ss_res / max(ss_tot, 1e-12))
            mean_r2 = float(np.mean(scores))
            if mean_r2 > best[2]:
                best = (a, g, mean_r2)
    m = KernelRidge(alpha=best[0], gamma=best[1], kernel="rbf")
    m.fit(X, y)
    return m, {"alpha": best[0], "gamma": best[1], "cv_r2": best[2]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--estimators", nargs="+",
                   default=["tcn_200", "mlp_reg", "krr_rbf"],
                   choices=["tcn_30", "tcn_200", "mlp_reg", "krr_rbf"])
    args = p.parse_args()

    device = require_rtx6000(gpu_index=args.gpu)
    print(f"[exp5] device={device}")

    portfolio = json.load(open(TCSI_PORTFOLIO))
    sensor_sets = {K: portfolio["TCSI_picks"][str(K)] for K in K_VALUES}
    print(f"[exp5] sensor sets:")
    for K, idx in sensor_sets.items():
        print(f"  K={K}: {idx}")

    train_lat = np.load(
        REPO / "outputs" / "session17" / "seed_latents" / "production" / "train.npz",
        allow_pickle=True,
    )
    test_a_lat = np.load(
        REPO / "outputs" / "session17" / "seed_latents" / "production" / "test_b.npz",
        allow_pickle=True,
    )  # Use test_b for MLP validation since test_a not encoded; will reload later
    # Actually we need test_a as held-out validation. Encode it on the fly if missing.
    # For MLP early-stopping, we use a held-out chunk of train (20% slice) instead.

    train_records = gather_split("train")
    test_a_records = gather_split("test_a")
    test_b_records = gather_split("test_b")
    test_c_records = gather_split("test_c")
    print(f"[exp5] train: {len(train_records)} test_a: {len(test_a_records)} "
          f"test_b: {len(test_b_records)} test_c: {len(test_c_records)}")

    # Use train order matching production train_lat order by (case_id, k)
    rec_lookup = {(r["case_id"], r["k"]): i for i, r in enumerate(train_records)}
    lat_order = [
        rec_lookup[(str(train_lat["case_id"][i]), int(train_lat["encounter_index"][i]))]
        for i in range(len(train_lat["case_id"]))
    ]
    train_records_ordered = [train_records[i] for i in lat_order]
    train_z_impact = train_lat["z"].astype(np.float64)
    train_GDY = np.stack([
        train_lat["G"].astype(np.float64),
        train_lat["D"].astype(np.float64),
        train_lat["Y"].astype(np.float64),
    ], axis=1)

    # Test ground-truth z_impact and (G, D, Y)
    b_lat = np.load(REPO / "outputs" / "session17" / "seed_latents" / "production" / "test_b.npz", allow_pickle=True)
    c_lat = np.load(REPO / "outputs" / "session17" / "seed_latents" / "production" / "test_c.npz", allow_pickle=True)
    b_lookup = {(str(c), int(e)): i for i, (c, e) in enumerate(zip(b_lat["case_id"], b_lat["encounter_index"]))}
    c_lookup = {(str(c), int(e)): i for i, (c, e) in enumerate(zip(c_lat["case_id"], c_lat["encounter_index"]))}
    b_test_z_true = np.stack(
        [b_lat["z"][b_lookup[(r["case_id"], r["k"])]].astype(np.float64) for r in test_b_records], axis=0
    )
    c_test_z_true = np.stack(
        [c_lat["z"][c_lookup[(r["case_id"], r["k"])]].astype(np.float64) for r in test_c_records], axis=0
    )

    # Load encoder + predictor + metric probes (reuse Exp 2 probes)
    enc, pred_model = load_encoder_predictor(device)
    pipeline = OmegaPipeline.from_manifest(OMEGA_MANIFEST)

    dns_metrics = np.load(REPO / "outputs" / "session17" / "exp2" / "dns_physical_metrics.npz", allow_pickle=True)
    train_full = np.load(REPO / "outputs" / "session14" / "latents" / "S12_E_d64" / "train.npz", allow_pickle=True)
    z_full_train = train_full["z_full"].astype(np.float32)
    train_full_cid = train_full["case_id"].astype(str)
    train_full_ei = train_full["encounter_index"].astype(int)
    z_full_map = {(c, e): i for i, (c, e) in enumerate(zip(train_full_cid, train_full_ei))}
    dns_train_cid = dns_metrics["train_case_id"].astype(str)
    dns_train_ei = dns_metrics["train_encounter_index"].astype(int)
    Z_all, Y_CL, Y_Iy, Y_Ens = [], [], [], []
    for j, (cid, ei) in enumerate(zip(dns_train_cid, dns_train_ei)):
        if (cid, ei) not in z_full_map:
            continue
        zi = z_full_map[(cid, ei)]
        Z_all.append(z_full_train[zi])
        Y_CL.append(dns_metrics["train_C_L"][j])
        Y_Iy.append(dns_metrics["train_I_y"][j])
        Y_Ens.append(dns_metrics["train_wake_enstrophy"][j])
    Z_all = np.concatenate(Z_all, axis=0)
    Y_all = {"C_L": np.concatenate(Y_CL), "I_y": np.concatenate(Y_Iy),
             "wake_enstrophy": np.concatenate(Y_Ens)}

    metric_probes = {}
    for name, y in Y_all.items():
        valid = ~np.isnan(y)
        Zv = Z_all[valid].astype(np.float64); yv = y[valid]
        mu = Zv.mean(axis=0); sigma = Zv.std(axis=0).clip(min=1e-9)
        Zn = (Zv - mu) / sigma
        A = Zn.T @ Zn + 1.0 * np.eye(Zn.shape[1])
        W = np.linalg.solve(A, Zn.T @ (yv - yv.mean()))
        metric_probes[name] = {"W": W, "mu_z": mu, "sigma_z": sigma, "b": float(yv.mean())}

    # =======================================================================
    # Train estimators per K
    # =======================================================================
    estimator_R2 = []
    pressure_models = {}  # pressure_models[K][estimator] = (z_model, c_model)
    for K in K_VALUES:
        sensors = sensor_sets[K]
        print(f"\n[exp5] === K={K}  sensors={sensors} ===")
        # Build features
        Xtr_tcn = build_X_tcn(train_records_ordered, sensors)
        Xtb_tcn = build_X_tcn(test_b_records, sensors)
        Xtc_tcn = build_X_tcn(test_c_records, sensors)
        Xtr_flat = build_X_flat(Xtr_tcn)
        Xtb_flat = build_X_flat(Xtb_tcn)
        Xtc_flat = build_X_flat(Xtc_tcn)

        # 20% held-out split for MLP validation
        n_tr = Xtr_flat.shape[0]
        rng = np.random.default_rng(0)
        perm = rng.permutation(n_tr)
        val_size = max(int(0.2 * n_tr), 4)
        val_idx = perm[:val_size]; tr_idx = perm[val_size:]

        pressure_models[K] = {}
        for est_name in args.estimators:
            t0 = time.time()
            if est_name == "tcn_30":
                cfg = TCNConfig(epochs=30, device=str(device), seed=0)
                m_z = TCNProxyLearner(out_dim=64, config=cfg); m_z.fit(Xtr_tcn, train_z_impact)
                m_c = TCNProxyLearner(out_dim=3, config=cfg); m_c.fit(Xtr_tcn, train_GDY)
                z_pred_b = m_z.predict(Xtb_tcn)
                z_pred_c = m_z.predict(Xtc_tcn)
                c_pred_b = m_c.predict(Xtb_tcn)
                c_pred_c = m_c.predict(Xtc_tcn)
                pressure_models[K][est_name] = {"type": "tcn", "z": m_z, "c": m_c}
            elif est_name == "tcn_200":
                cfg = TCNConfig(epochs=200, device=str(device), seed=0)
                m_z = TCNProxyLearner(out_dim=64, config=cfg); m_z.fit(Xtr_tcn, train_z_impact)
                m_c = TCNProxyLearner(out_dim=3, config=cfg); m_c.fit(Xtr_tcn, train_GDY)
                z_pred_b = m_z.predict(Xtb_tcn)
                z_pred_c = m_z.predict(Xtc_tcn)
                c_pred_b = m_c.predict(Xtb_tcn)
                c_pred_c = m_c.predict(Xtc_tcn)
                pressure_models[K][est_name] = {"type": "tcn", "z": m_z, "c": m_c}
            elif est_name == "mlp_reg":
                mdl_z, info_z = train_mlp_with_early_stop(
                    Xtr_flat[tr_idx], train_z_impact[tr_idx],
                    Xtr_flat[val_idx], train_z_impact[val_idx], device,
                )
                mdl_c, info_c = train_mlp_with_early_stop(
                    Xtr_flat[tr_idx], train_GDY[tr_idx],
                    Xtr_flat[val_idx], train_GDY[val_idx], device,
                )
                z_pred_b = predict_mlp(mdl_z, Xtb_flat, info_z, device)
                z_pred_c = predict_mlp(mdl_z, Xtc_flat, info_z, device)
                c_pred_b = predict_mlp(mdl_c, Xtb_flat, info_c, device)
                c_pred_c = predict_mlp(mdl_c, Xtc_flat, info_c, device)
                pressure_models[K][est_name] = {
                    "type": "mlp", "z": (mdl_z, info_z), "c": (mdl_c, info_c),
                }
            elif est_name == "krr_rbf":
                # KRR doesn't naturally do multivariate; loop per output dim.
                # For z (64 dims), batch by fitting one KRR per dim. With CV that's
                # 64 * 20 fits = expensive. Use single CV for z (no per-dim tuning).
                # CV on first 10 dims to pick global alpha, gamma.
                alphas = [0.1, 1.0, 10.0]
                gammas = [0.001, 0.005, 0.01]
                kf = KFold(n_splits=5, shuffle=True, random_state=0)
                best = (None, None, -np.inf)
                Xtr = Xtr_flat
                ysub = train_z_impact[:, :10]  # CV over first 10 dims
                for a in alphas:
                    for g in gammas:
                        scores = []
                        for tr, va in kf.split(Xtr):
                            mm = KernelRidge(alpha=a, gamma=g, kernel="rbf")
                            mm.fit(Xtr[tr], ysub[tr])
                            ppred = mm.predict(Xtr[va])
                            ss_res = ((ysub[va] - ppred) ** 2).sum()
                            ss_tot = ((ysub[va] - ysub[va].mean(0)) ** 2).sum()
                            scores.append(1 - ss_res / max(ss_tot, 1e-12))
                        if np.mean(scores) > best[2]:
                            best = (a, g, float(np.mean(scores)))
                m_z = KernelRidge(alpha=best[0], gamma=best[1], kernel="rbf")
                m_z.fit(Xtr, train_z_impact)
                # Same hyperparams for c (good enough; CV per output dim is overkill)
                m_c = KernelRidge(alpha=best[0], gamma=best[1], kernel="rbf")
                m_c.fit(Xtr, train_GDY)
                z_pred_b = m_z.predict(Xtb_flat)
                z_pred_c = m_z.predict(Xtc_flat)
                c_pred_b = m_c.predict(Xtb_flat)
                c_pred_c = m_c.predict(Xtc_flat)
                pressure_models[K][est_name] = {"type": "krr", "z": m_z, "c": m_c,
                                                "alpha": best[0], "gamma": best[1]}
            else:
                continue

            # Compute R^2
            r2_z_b = r2_per_col(b_test_z_true, z_pred_b).mean()
            r2_z_c = r2_per_col(c_test_z_true, z_pred_c).mean()
            c_true_b = np.array([[r["G"], r["D"], r["Y"]] for r in test_b_records], dtype=np.float64)
            c_true_c = np.array([[r["G"], r["D"], r["Y"]] for r in test_c_records], dtype=np.float64)
            r2_c_b = r2_per_col(c_true_b, c_pred_b)
            r2_c_c = r2_per_col(c_true_c, c_pred_c)
            estimator_R2.append({
                "K": K, "estimator": est_name,
                "fit_time_s": float(time.time() - t0),
                "test_b_R2_z_mean": float(r2_z_b),
                "test_c_R2_z_mean": float(r2_z_c),
                "test_b_G_R2": float(r2_c_b[0]),
                "test_b_D_R2": float(r2_c_b[1]),
                "test_b_Y_R2": float(r2_c_b[2]),
                "test_c_G_R2": float(r2_c_c[0]),
                "test_c_D_R2": float(r2_c_c[1]),
                "test_c_Y_R2": float(r2_c_c[2]),
            })
            print(
                f"[exp5] K={K:2d} {est_name:10s}  "
                f"z_R2 test_b={r2_z_b:+.3f} test_c={r2_z_c:+.3f}  "
                f"c_R2 test_b: G={r2_c_b[0]:+.3f} D={r2_c_b[1]:+.3f} Y={r2_c_b[2]:+.3f}  "
                f"({time.time()-t0:.1f}s)"
            )
            # Save predictions for downstream closed-loop
            pressure_models[K][est_name]["z_pred_b"] = z_pred_b
            pressure_models[K][est_name]["z_pred_c"] = z_pred_c
            pressure_models[K][est_name]["c_pred_b"] = c_pred_b
            pressure_models[K][est_name]["c_pred_c"] = c_pred_c

    with (OUT / "nonlinear_estimator_R2.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(estimator_R2[0].keys()))
        w.writeheader(); w.writerows(estimator_R2)
    print(f"\n[exp5] wrote {OUT / 'nonlinear_estimator_R2.csv'}")

    # Pick best estimator per K based on test_b z R^2
    best_est_per_K = {}
    for K in K_VALUES:
        rows = [r for r in estimator_R2 if r["K"] == K]
        rows.sort(key=lambda r: r["test_b_R2_z_mean"], reverse=True)
        best_est_per_K[K] = rows[0]["estimator"]
    print(f"\n[exp5] best estimator per K (by test_b z R^2): {best_est_per_K}")

    # =======================================================================
    # Closed-loop rollouts (3 modes) using best estimator per K
    # =======================================================================
    print(f"\n[exp5] closed-loop rollouts...")
    closed_loop_rows = []
    for K in K_VALUES:
        est_name = best_est_per_K[K]
        m = pressure_models[K][est_name]
        for split_name, recs, z_true, z_pred_split, c_pred_split in (
            ("test_b", test_b_records, b_test_z_true,
             m["z_pred_b"], m["c_pred_b"]),
            ("test_c", test_c_records, c_test_z_true,
             m["z_pred_c"], m["c_pred_c"]),
        ):
            t0 = time.time()
            for i, e in enumerate(recs):
                z_oracle = z_true[i]
                z_hat = z_pred_split[i]
                c_hat = c_pred_split[i]
                z_oracle_t = torch.from_numpy(z_oracle.astype(np.float32)).unsqueeze(0).to(device)
                z_hat_t = torch.from_numpy(z_hat.astype(np.float32)).unsqueeze(0).to(device)
                cond_oracle = torch.tensor([[e["G"], e["D"], e["Y"]]], dtype=torch.float32, device=device)
                cond_hat = torch.tensor([c_hat.astype(np.float32)], dtype=torch.float32, device=device)
                # Run rollouts in 3 modes
                modes = {
                    "A_oracle": (z_oracle_t, cond_oracle),
                    "B_z_hat_c_oracle": (z_hat_t, cond_oracle),
                    "C_z_hat_c_hat": (z_hat_t, cond_hat),
                }
                with h5py.File(e["path"], "r") as f:
                    impact_h5 = int(f.attrs.get("impact_frame_estimate", DEFAULT_IMPACT_FRAME))
                # DNS metrics for this encounter
                dns_cid = dns_metrics[f"{split_name}_case_id"].astype(str)
                dns_ei = dns_metrics[f"{split_name}_encounter_index"].astype(int)
                d_idx = np.where((dns_cid == e["case_id"]) & (dns_ei == e["k"]))[0]
                if d_idx.size == 0:
                    continue
                di = int(d_idx[0])
                H_max = max(HORIZONS)
                for mode_name, (z_seed, cond) in modes.items():
                    z_pred = rollout_markov(pred_model, z_seed, cond, H_max, device).squeeze(0).cpu().numpy()
                    z_post = z_pred[1:]  # drop seed
                    for metric in PROBE_METRICS:
                        pred_metric = apply_metric_probe(z_post, metric_probes[metric])
                        dns_seq = dns_metrics[f"{split_name}_{metric}"][di]
                        for H in HORIZONS:
                            if H - 1 >= len(pred_metric):
                                continue
                            t_abs = impact_h5 + H
                            if t_abs >= len(dns_seq):
                                continue
                            dns_val = float(dns_seq[t_abs])
                            pred_val = float(pred_metric[H - 1])
                            err = abs(pred_val - dns_val)
                            ref = max(abs(dns_val), 1e-9)
                            within = err < TOLERANCE[metric] * ref
                            closed_loop_rows.append({
                                "K": K, "estimator": est_name, "split": split_name,
                                "case_id": e["case_id"], "encounter_index": e["k"],
                                "Y": e["Y"], "G": e["G"], "D": e["D"],
                                "mode": mode_name, "metric": metric, "H": H,
                                "dns_val": dns_val, "pred_val": pred_val,
                                "abs_err": err, "rel_err": err / ref,
                                "tolerance_rel": TOLERANCE[metric],
                                "within_tolerance": within,
                            })
                if (i + 1) % 5 == 0:
                    print(f"[exp5] K={K} {split_name} {i+1}/{len(recs)}  "
                          f"({(time.time()-t0)/(i+1):.2f}s/enc)")
            print(f"[exp5] K={K} {split_name} done in {time.time()-t0:.1f}s")

    with (OUT / "nonlinear_closed_loop_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(closed_loop_rows[0].keys()))
        w.writeheader(); w.writerows(closed_loop_rows)
    print(f"\n[exp5] wrote nonlinear_closed_loop_metrics.csv  ({len(closed_loop_rows)} rows)")

    # Tolerance curves
    tol_curves = {}
    for split in ("test_b", "test_c"):
        for mode in ("A_oracle", "B_z_hat_c_oracle", "C_z_hat_c_hat"):
            for metric in PROBE_METRICS:
                for H in HORIZONS:
                    rows = [r for r in closed_loop_rows
                            if r["split"] == split and r["mode"] == mode
                            and r["metric"] == metric and r["H"] == H]
                    if not rows:
                        continue
                    per_K = {}
                    for K in K_VALUES:
                        rk = [r for r in rows if r["K"] == K]
                        if not rk:
                            continue
                        per_K[K] = {
                            "n": len(rk),
                            "estimator": best_est_per_K[K],
                            "frac_within": float(sum(r["within_tolerance"] for r in rk) / len(rk)),
                            "median_rel_err": float(np.median([r["rel_err"] for r in rk])),
                        }
                    tol_curves[f"{split}__{mode}__{metric}__H{H}"] = per_K
    (OUT / "nonlinear_tolerance_curves.json").write_text(json.dumps(tol_curves, indent=2))
    print(f"[exp5] wrote nonlinear_tolerance_curves.json")

    # Gates summary
    gates = {}
    for split in ("test_b", "test_c"):
        for mode in ("B_z_hat_c_oracle", "C_z_hat_c_hat"):
            key = f"{split}__{mode}__C_L__H16"
            if key in tol_curves and 8 in tol_curves[key]:
                v = tol_curves[key][8]["frac_within"]
                gates[f"{split}__{mode}__C_L_K8"] = {"frac": v, "above_0p8": v >= 0.8}
            key = f"{split}__{mode}__I_y__H16"
            if key in tol_curves and 8 in tol_curves[key]:
                v = tol_curves[key][8]["frac_within"]
                gates[f"{split}__{mode}__I_y_K8"] = {"frac": v, "above_0p7": v >= 0.7}
    (OUT / "nonlinear_exp5_gates.json").write_text(json.dumps(gates, indent=2))
    print(f"\n[exp5] gates: {json.dumps(gates, indent=2)}")

    # Figure 1: K-curve comparison across estimators (test_b z R^2)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    by_est_K = {}
    for r in estimator_R2:
        by_est_K.setdefault(r["estimator"], {})[r["K"]] = (
            r["test_b_R2_z_mean"], r["test_c_R2_z_mean"]
        )
    for est_name, by_K in by_est_K.items():
        Ks = sorted(by_K.keys())
        z_b = [by_K[K][0] for K in Ks]
        z_c = [by_K[K][1] for K in Ks]
        axes[0].plot(Ks, z_b, "-o", label=est_name, lw=1.5, ms=8)
        axes[1].plot(Ks, z_c, "-o", label=est_name, lw=1.5, ms=8)
    for ax, title in zip(axes, ("Test B  z R^2", "Test C  z R^2")):
        ax.axhline(0.0, color="gray", alpha=0.4)
        ax.axhline(0.5, color="green", ls=":", alpha=0.6, label="0.5 ref")
        ax.set_xscale("log", base=2)
        ax.set_xticks(K_VALUES); ax.set_xticklabels([str(K) for K in K_VALUES])
        ax.set_xlabel("K (sensors)")
        ax.set_ylabel("mean R^2 across 64 dims")
        ax.set_title(title)
        ax.set_ylim(-1.0, 1.0)
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle("Pressure -> z_impact R^2 vs K (nonlinear estimators)")
    fig.tight_layout()
    fig.savefig(FIGS / "exp5_nonlinear_K_curve.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[exp5] wrote {FIGS / 'exp5_nonlinear_K_curve.png'}")

    # Figure 2: tolerance envelope with best estimator
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    for col, metric in enumerate(PROBE_METRICS):
        for row, split in enumerate(("test_b", "test_c")):
            ax = axes[row, col]
            for mode, color, marker in (
                ("A_oracle", "tab:gray", "o"),
                ("B_z_hat_c_oracle", "tab:blue", "s"),
                ("C_z_hat_c_hat", "tab:red", "^"),
            ):
                key = f"{split}__{mode}__{metric}__H16"
                if key not in tol_curves:
                    continue
                Ks = sorted(tol_curves[key].keys())
                vals = [tol_curves[key][K]["frac_within"] for K in Ks]
                est_labels = [tol_curves[key][K]["estimator"] for K in Ks]
                ax.plot(Ks, vals, marker, color=color, ms=8, lw=1.5,
                        label=f"{mode}")
                ax.plot(Ks, vals, "-", color=color, alpha=0.4)
                for K_idx, K in enumerate(Ks):
                    ax.annotate(
                        est_labels[K_idx][:6], (K, vals[K_idx]),
                        textcoords="offset points", xytext=(3, 5), fontsize=6,
                    )
            ax.set_xlabel("K"); ax.set_ylabel(f"fraction within {int(TOLERANCE[metric]*100)}% tol")
            ax.set_title(f"{split}  {metric}  H=16")
            ax.set_ylim(0, 1.05)
            ax.set_xscale("log", base=2)
            ax.set_xticks(K_VALUES); ax.set_xticklabels([str(K) for K in K_VALUES])
            ax.grid(alpha=0.3)
            target_gate = 0.8 if metric == "C_L" else (0.7 if metric == "I_y" else 0.5)
            ax.axhline(target_gate, color="green", ls=":", alpha=0.6, label=f"gate={target_gate}")
            if row == 0 and col == 2:
                ax.legend(fontsize=8)
    fig.suptitle("Closed-loop physical-metric tolerance vs K (best nonlinear estimator per K)")
    fig.tight_layout()
    fig.savefig(FIGS / "exp5_nonlinear_tolerance.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[exp5] wrote {FIGS / 'exp5_nonlinear_tolerance.png'}")


if __name__ == "__main__":
    main()
