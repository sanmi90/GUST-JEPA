"""Session 17, Experiment 5: closed-loop sparse pressure observability.

Streamlined recipe (no TCN, uses linear ridge for the pressure->z and
pressure->(G,D,Y) maps):

  1. For K in {2, 4, 8, 16}, use the TCSI-selected sensor indices from
     Session 14 (outputs/session14/tcsi_pilot/methods_portfolio.json).
  2. From the 30-frame pre-impact p_wall window at K sensors, fit a ridge
     regressor to predict z_impact (64-D) and a separate ridge to predict
     (G, D, Y).
  3. For each Test B / Test C encounter, run 3 closed-loop Markov rollouts:
     Mode A: oracle z_impact + oracle c
     Mode B: pressure-inferred z_hat + oracle c
     Mode C: pressure-inferred z_hat + pressure-inferred c_hat
  4. Apply the z->{C_L, I_y, wake_enstrophy} probes from Exp 2 to the
     predicted z trajectories.
  5. Tolerance curves: fraction of encounters within metric tolerance vs K.

Tolerances (SESSION17_PLAN.md): 10% of |C_L|, 15% of |I_y|, 25% of wake_enstrophy.

Outputs:
    outputs/session17/exp5/pressure_to_z_R2.csv
    outputs/session17/exp5/pressure_to_c_R2.csv
    outputs/session17/exp5/closed_loop_physical_metrics.csv
    outputs/session17/exp5/tolerance_curves.npz
    outputs/session17/figures/{exp5_K_curve_physical_metrics, exp5_tolerance_envelope}.png
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
from torch import Tensor
from torch.nn import functional as F


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.models.predictor import (  # noqa: E402
    AutoregressivePredictor,
    CausalSelfAttentionWithRoPE,
)
from src.models.rope import apply_rope  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402


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
HALF_WINDOW = 15  # 30-frame pre-impact window
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


def load_pressure_window(rec: dict, impact: int = DEFAULT_IMPACT_FRAME) -> np.ndarray:
    """Return pressure window of shape (2*HALF_WINDOW, 192) centred at impact-1.

    Per CLAUDE.md, p_wall is already the spanwise-mean. Shape (120, 192)."""
    with h5py.File(rec["path"], "r") as f:
        p_wall = np.asarray(f["p_wall"], dtype=np.float32)  # (120, 192)
        rec_impact = int(f.attrs.get("impact_frame_estimate", DEFAULT_IMPACT_FRAME))
    t_start = max(0, rec_impact - 2 * HALF_WINDOW)
    t_end = rec_impact  # pre-impact window only
    window = p_wall[t_start:t_end]
    # Pad to fixed length 2*HALF_WINDOW
    if window.shape[0] < 2 * HALF_WINDOW:
        pad = np.zeros((2 * HALF_WINDOW - window.shape[0], 192), dtype=np.float32)
        window = np.concatenate([pad, window], axis=0)
    return window, rec_impact


def build_feature_matrix(records: list[dict], sensor_idx: list[int]) -> tuple[np.ndarray, list[int]]:
    """Stack the K-sensor 30-frame pressure window per record into a flat feature vector.

    Returns (X: (n, K*30), impact_frames: list[int])."""
    K = len(sensor_idx)
    X = np.zeros((len(records), K * 2 * HALF_WINDOW), dtype=np.float64)
    impacts = []
    for i, rec in enumerate(records):
        w, impact = load_pressure_window(rec)
        X[i] = w[:, sensor_idx].reshape(-1)
        impacts.append(impact)
    return X, impacts


def fit_ridge_multi(X: np.ndarray, Y: np.ndarray, alpha: float = 1.0) -> dict:
    """Standardise X; fit one ridge per output column."""
    mu_x = X.mean(axis=0)
    sigma_x = X.std(axis=0).clip(min=1e-9)
    Xn = (X - mu_x) / sigma_x
    mu_y = Y.mean(axis=0)
    Yc = Y - mu_y
    A = Xn.T @ Xn + alpha * np.eye(Xn.shape[1])
    W = np.linalg.solve(A, Xn.T @ Yc)  # (d_x, d_y)
    return {"W": W, "mu_x": mu_x, "sigma_x": sigma_x, "mu_y": mu_y}


def apply_ridge_multi(X: np.ndarray, model: dict) -> np.ndarray:
    Xn = (X - model["mu_x"]) / model["sigma_x"]
    return Xn @ model["W"] + model["mu_y"]


def r2_per_col(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    if y_true.ndim == 1:
        ss_res = float(((y_true - y_pred) ** 2).sum())
        ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
        return np.array([1.0 - ss_res / max(ss_tot, 1e-12)])
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.clip(ss_tot, 1e-12, None)


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = require_rtx6000(gpu_index=args.gpu)
    print(f"[exp5] device={device}")

    # Load TCSI sensor selections
    portfolio = json.load(open(TCSI_PORTFOLIO))
    sensor_sets = {K: portfolio["TCSI_picks"][str(K)] for K in K_VALUES}
    print(f"[exp5] sensor sets:")
    for K, idx in sensor_sets.items():
        print(f"  K={K}: {idx}")

    enc, pred_model = load_encoder_predictor(device)
    pipeline = OmegaPipeline.from_manifest(OMEGA_MANIFEST)
    train_lat = np.load(
        REPO / "outputs" / "session17" / "seed_latents" / "production" / "train.npz",
        allow_pickle=True,
    )
    train_z_impact = train_lat["z"].astype(np.float64)  # (180, 64)
    train_GDY = np.stack([
        train_lat["G"].astype(np.float64),
        train_lat["D"].astype(np.float64),
        train_lat["Y"].astype(np.float64),
    ], axis=1)

    # Build the training feature matrices per K
    train_records = gather_split("train")
    test_b_records = gather_split("test_b")
    test_c_records = gather_split("test_c")
    print(f"[exp5] train: {len(train_records)} test_b: {len(test_b_records)} test_c: {len(test_c_records)}")

    # Match train records to train_lat order by (case_id, k)
    rec_lookup = {(r["case_id"], r["k"]): i for i, r in enumerate(train_records)}
    lat_order = [
        rec_lookup[(train_lat["case_id"][i], int(train_lat["encounter_index"][i]))]
        for i in range(len(train_lat["case_id"]))
    ]
    # train_records_ordered[i] corresponds to train_z_impact[i]
    train_records_ordered = [train_records[i] for i in lat_order]

    # Load z->metric probes (from Exp 2)
    dns_metrics = np.load(REPO / "outputs" / "session17" / "exp2" / "dns_physical_metrics.npz", allow_pickle=True)
    train_full = np.load(REPO / "outputs" / "session14" / "latents" / "S12_E_d64" / "train.npz", allow_pickle=True)
    # Fit probes z -> {C_L, I_y, wake_enstrophy} on train per-frame
    z_full_train = train_full["z_full"].astype(np.float32)
    # Align by (case_id, k)
    dns_cid = dns_metrics["train_case_id"].astype(str)
    dns_ei = dns_metrics["train_encounter_index"].astype(int)
    train_full_cid = train_full["case_id"].astype(str)
    train_full_ei = train_full["encounter_index"].astype(int)
    z_full_map = {(c, e): i for i, (c, e) in enumerate(zip(train_full_cid, train_full_ei))}
    Z_all = []; Y_CL, Y_Iy, Y_Ens = [], [], []
    for j, (cid, ei) in enumerate(zip(dns_cid, dns_ei)):
        if (cid, ei) not in z_full_map:
            continue
        zi = z_full_map[(cid, ei)]
        zf = z_full_train[zi]
        Z_all.append(zf)
        Y_CL.append(dns_metrics["train_C_L"][j])
        Y_Iy.append(dns_metrics["train_I_y"][j])
        Y_Ens.append(dns_metrics["train_wake_enstrophy"][j])
    Z_all = np.concatenate(Z_all, axis=0)
    Y_all = {"C_L": np.concatenate(Y_CL), "I_y": np.concatenate(Y_Iy),
             "wake_enstrophy": np.concatenate(Y_Ens)}

    metric_probes = {}
    for name, y in Y_all.items():
        valid = ~np.isnan(y)
        Zv = Z_all[valid].astype(np.float64)
        yv = y[valid]
        mu = Zv.mean(axis=0); sigma = Zv.std(axis=0).clip(min=1e-9)
        Zn = (Zv - mu) / sigma
        A = Zn.T @ Zn + 1.0 * np.eye(Zn.shape[1])
        W = np.linalg.solve(A, Zn.T @ (yv - yv.mean()))
        metric_probes[name] = {"W": W, "mu_z": mu, "sigma_z": sigma, "b": float(yv.mean())}
        pred = (Zn @ W) + yv.mean()
        r2 = 1 - ((yv - pred) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
        print(f"[exp5] z->{name:20s} train R^2 = {r2:.3f}")

    # ---------- Step (a) + (b): build pressure -> z and pressure -> (G,D,Y) per K
    p_to_z_R2 = []
    p_to_c_R2 = []
    pressure_models = {}
    for K in K_VALUES:
        sensors = sensor_sets[K]
        X_train, _ = build_feature_matrix(train_records_ordered, sensors)
        X_test_b, _ = build_feature_matrix(test_b_records, sensors)
        X_test_c, _ = build_feature_matrix(test_c_records, sensors)
        # Pressure -> z_impact
        model_z = fit_ridge_multi(X_train, train_z_impact, alpha=1.0)
        z_pred_train = apply_ridge_multi(X_train, model_z)
        z_pred_b = apply_ridge_multi(X_test_b, model_z)
        z_pred_c = apply_ridge_multi(X_test_c, model_z)
        # Pressure -> (G,D,Y)
        model_c = fit_ridge_multi(X_train, train_GDY, alpha=1.0)
        c_pred_b = apply_ridge_multi(X_test_b, model_c)
        c_pred_c = apply_ridge_multi(X_test_c, model_c)
        pressure_models[K] = {
            "z": model_z, "c": model_c,
            "X_train": X_train, "X_test_b": X_test_b, "X_test_c": X_test_c,
            "z_pred_train": z_pred_train, "z_pred_b": z_pred_b, "z_pred_c": z_pred_c,
            "c_pred_b": c_pred_b, "c_pred_c": c_pred_c,
        }
        # Compute R^2
        # z R^2: per-dimension average
        r2_z_b = float(r2_per_col(train_z_impact, z_pred_train).mean())  # train as sanity
        # Get test_b truth z_impact
        b_lat = np.load(REPO / "outputs" / "session17" / "seed_latents" / "production" / "test_b.npz", allow_pickle=True)
        c_lat = np.load(REPO / "outputs" / "session17" / "seed_latents" / "production" / "test_c.npz", allow_pickle=True)
        # Match test_b records to b_lat ordering by (case_id, k)
        b_lookup = {(str(c), int(e)): i for i, (c, e) in enumerate(zip(b_lat["case_id"], b_lat["encounter_index"]))}
        c_lookup = {(str(c), int(e)): i for i, (c, e) in enumerate(zip(c_lat["case_id"], c_lat["encounter_index"]))}
        # Note: test_b records (v1, 28) ordering must match b_lat ordering (v1p5, 56) by (case, k)
        # Find matched indices
        b_test_z_true = []
        for r in test_b_records:
            b_test_z_true.append(b_lat["z"][b_lookup[(r["case_id"], r["k"])]].astype(np.float64))
        c_test_z_true = []
        for r in test_c_records:
            c_test_z_true.append(c_lat["z"][c_lookup[(r["case_id"], r["k"])]].astype(np.float64))
        b_test_z_true = np.stack(b_test_z_true, axis=0)
        c_test_z_true = np.stack(c_test_z_true, axis=0)
        r2_z_b_test = float(r2_per_col(b_test_z_true, z_pred_b).mean())
        r2_z_c_test = float(r2_per_col(c_test_z_true, z_pred_c).mean())
        c_true_b = np.array([[r["G"], r["D"], r["Y"]] for r in test_b_records], dtype=np.float64)
        c_true_c = np.array([[r["G"], r["D"], r["Y"]] for r in test_c_records], dtype=np.float64)
        r2_c_b = r2_per_col(c_true_b, c_pred_b)
        r2_c_c = r2_per_col(c_true_c, c_pred_c)
        p_to_z_R2.append({
            "K": K,
            "train_R2_z_mean": r2_z_b,
            "test_b_R2_z_mean": r2_z_b_test,
            "test_c_R2_z_mean": r2_z_c_test,
        })
        p_to_c_R2.append({
            "K": K,
            "G_test_b": float(r2_c_b[0]), "D_test_b": float(r2_c_b[1]), "Y_test_b": float(r2_c_b[2]),
            "G_test_c": float(r2_c_c[0]), "D_test_c": float(r2_c_c[1]), "Y_test_c": float(r2_c_c[2]),
        })
        pressure_models[K]["b_test_z_true"] = b_test_z_true
        pressure_models[K]["c_test_z_true"] = c_test_z_true
        print(
            f"[exp5] K={K:2d}: z R^2 train={r2_z_b:.3f} test_b={r2_z_b_test:.3f} test_c={r2_z_c_test:.3f}  "
            f"c R^2 test_b: G={r2_c_b[0]:.3f} D={r2_c_b[1]:.3f} Y={r2_c_b[2]:.3f}"
        )

    # Save Z and C R^2
    with (OUT / "pressure_to_z_R2.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(p_to_z_R2[0].keys())); w.writeheader(); w.writerows(p_to_z_R2)
    with (OUT / "pressure_to_c_R2.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(p_to_c_R2[0].keys())); w.writeheader(); w.writerows(p_to_c_R2)
    print(f"\n[exp5] wrote pressure_to_z_R2.csv and pressure_to_c_R2.csv")

    # ---------- Step (c): closed-loop rollouts in 3 modes
    print(f"\n[exp5] closed-loop rollouts...")
    closed_loop_rows = []
    for K in K_VALUES:
        sensors = sensor_sets[K]
        model_z = pressure_models[K]["z"]
        model_c = pressure_models[K]["c"]
        for split_name, recs in (("test_b", test_b_records), ("test_c", test_c_records)):
            t0 = time.time()
            n = len(recs)
            for i, e in enumerate(recs):
                # Oracle z_impact (from production encoded latents)
                key = "b" if split_name == "test_b" else "c"
                if split_name == "test_b":
                    z_oracle = pressure_models[K]["b_test_z_true"][i]
                else:
                    z_oracle = pressure_models[K]["c_test_z_true"][i]
                z_oracle_t = torch.from_numpy(z_oracle.astype(np.float32)).unsqueeze(0).to(device)
                # Pressure features
                w_pres, _ = load_pressure_window(e)
                feat = w_pres[:, sensors].reshape(1, -1).astype(np.float64)
                z_hat = apply_ridge_multi(feat, model_z)[0]  # (64,)
                c_hat = apply_ridge_multi(feat, model_c)[0]  # (3,)
                z_hat_t = torch.from_numpy(z_hat.astype(np.float32)).unsqueeze(0).to(device)
                cond_oracle = torch.tensor([[e["G"], e["D"], e["Y"]]], dtype=torch.float32, device=device)
                cond_hat = torch.tensor([c_hat.astype(np.float32)], dtype=torch.float32, device=device)

                # Run 3 closed-loop modes
                modes = {
                    "A_oracle": (z_oracle_t, cond_oracle),
                    "B_z_hat_c_oracle": (z_hat_t, cond_oracle),
                    "C_z_hat_c_hat": (z_hat_t, cond_hat),
                }
                # Need DNS truth too for the comparison
                with h5py.File(e["path"], "r") as f:
                    omega_raw = np.asarray(f["omega_z"], dtype=np.float32)
                    impact_h5 = int(f.attrs.get("impact_frame_estimate", DEFAULT_IMPACT_FRAME))
                omega_clean = pipeline.preprocess_raw(omega_raw, e["case_id"], e["k"])
                omega_norm = pipeline.normalize(omega_clean).astype(np.float32)
                # DNS z trajectory for reference
                z_dns = encode_full(enc, omega_norm, device).cpu().numpy()
                # Run rollouts and apply metric probes
                H_max = max(HORIZONS)
                for mode_name, (z_seed, cond) in modes.items():
                    z_pred = rollout_markov(pred_model, z_seed, cond, H_max, device).squeeze(0).cpu().numpy()
                    z_post = z_pred[1:]  # drop seed
                    # Apply metric probes
                    for metric in PROBE_METRICS:
                        pred_metric = apply_metric_probe(z_post, metric_probes[metric])
                        # DNS metric from cached
                        dns_arr = dns_metrics[f"{split_name}_{metric}"]  # (n_enc, T)
                        d_cid = dns_metrics[f"{split_name}_case_id"].astype(str)
                        d_ei = dns_metrics[f"{split_name}_encounter_index"].astype(int)
                        d_idx = np.where((d_cid == e["case_id"]) & (d_ei == e["k"]))[0]
                        if d_idx.size == 0:
                            continue
                        dns_seq = dns_arr[d_idx[0]]
                        for H in HORIZONS:
                            if H - 1 >= len(pred_metric):
                                continue
                            t_abs = impact_h5 + H  # frame index in DNS
                            if t_abs >= len(dns_seq):
                                continue
                            dns_val = float(dns_seq[t_abs])
                            pred_val = float(pred_metric[H - 1])
                            err = abs(pred_val - dns_val)
                            ref = max(abs(dns_val), 1e-9)
                            within = err < TOLERANCE[metric] * ref
                            closed_loop_rows.append({
                                "K": K,
                                "split": split_name,
                                "case_id": e["case_id"],
                                "encounter_index": e["k"],
                                "Y": e["Y"], "G": e["G"], "D": e["D"],
                                "mode": mode_name,
                                "metric": metric,
                                "H": H,
                                "dns_val": dns_val,
                                "pred_val": pred_val,
                                "abs_err": err,
                                "rel_err": err / ref,
                                "tolerance_rel": TOLERANCE[metric],
                                "within_tolerance": within,
                            })
                if (i + 1) % 5 == 0:
                    print(f"[exp5] K={K} {split_name} {i+1}/{n}  ({(time.time()-t0)/(i+1):.2f}s/enc)")
            print(f"[exp5] K={K} {split_name} done in {time.time()-t0:.1f}s")

    with (OUT / "closed_loop_physical_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(closed_loop_rows[0].keys()))
        w.writeheader(); w.writerows(closed_loop_rows)
    print(f"\n[exp5] wrote closed_loop_physical_metrics.csv  ({len(closed_loop_rows)} rows)")

    # ---------- Step (d): tolerance curves
    print(f"\n[exp5] tolerance curves...")
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
                            "frac_within": float(sum(r["within_tolerance"] for r in rk) / len(rk)),
                            "median_rel_err": float(np.median([r["rel_err"] for r in rk])),
                        }
                    tol_curves[f"{split}__{mode}__{metric}__H{H}"] = per_K

    np.savez_compressed(
        OUT / "tolerance_curves.npz",
        **{k: np.array([(v.get(K, {}).get("frac_within", np.nan)) for K in K_VALUES])
           for k, v in tol_curves.items()},
    )

    with (OUT / "tolerance_curves.json").open("w") as f:
        json.dump(tol_curves, f, indent=2)
    print(f"[exp5] wrote tolerance_curves.json")

    # Acceptance gates
    gates = {}
    for split in ("test_b", "test_c"):
        if f"{split}__C_z_hat_c_hat__C_L__H16" in tol_curves:
            cl_at_K8 = tol_curves[f"{split}__C_z_hat_c_hat__C_L__H16"].get(8, {}).get("frac_within", 0.0)
            gates[f"{split}__C_L_K8_above_0p8"] = cl_at_K8 >= 0.8
            gates[f"{split}__C_L_K8_frac"] = cl_at_K8
        if f"{split}__C_z_hat_c_hat__I_y__H16" in tol_curves:
            iy_at_K8 = tol_curves[f"{split}__C_z_hat_c_hat__I_y__H16"].get(8, {}).get("frac_within", 0.0)
            gates[f"{split}__I_y_K8_above_0p7"] = iy_at_K8 >= 0.7
            gates[f"{split}__I_y_K8_frac"] = iy_at_K8
    print(f"\n[exp5] gates: {gates}")
    (OUT / "exp5_gates.json").write_text(json.dumps(gates, indent=2))

    # Figure 1: K-curve of physical metric errors (mode A, B, C) at H=16
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
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
                ks = sorted(tol_curves[key].keys())
                vals = [tol_curves[key][K]["median_rel_err"] for K in ks]
                ax.plot(ks, vals, marker, color=color, ms=8, lw=1.5, label=mode)
                ax.plot(ks, vals, "-", color=color, alpha=0.4)
            ax.axhline(TOLERANCE[metric], color="green", ls="--", alpha=0.6,
                       label=f"tol={int(TOLERANCE[metric]*100)}%")
            ax.set_xlabel("K (sensors)")
            ax.set_ylabel(f"{metric} median rel err")
            ax.set_title(f"{split}  {metric}  H=16")
            ax.set_xscale("log", base=2)
            ax.set_xticks(K_VALUES); ax.set_xticklabels([str(K) for K in K_VALUES])
            ax.grid(alpha=0.3)
            if row == 0 and col == 2:
                ax.legend(fontsize=8)
    fig.suptitle("Closed-loop physical-metric errors vs K (median across encounters)")
    fig.tight_layout()
    fig.savefig(FIGS / "exp5_K_curve_physical_metrics.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[exp5] wrote {FIGS / 'exp5_K_curve_physical_metrics.png'}")

    # Figure 2: tolerance envelope
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
                ks = sorted(tol_curves[key].keys())
                vals = [tol_curves[key][K]["frac_within"] for K in ks]
                ax.plot(ks, vals, marker, color=color, ms=8, lw=1.5, label=mode)
                ax.plot(ks, vals, "-", color=color, alpha=0.4)
            ax.set_xlabel("K")
            ax.set_ylabel(f"fraction within {int(TOLERANCE[metric]*100)}% tol")
            ax.set_title(f"{split}  {metric}  H=16")
            ax.set_ylim(0, 1.05)
            ax.set_xscale("log", base=2)
            ax.set_xticks(K_VALUES); ax.set_xticklabels([str(K) for K in K_VALUES])
            ax.grid(alpha=0.3)
            ax.axhline(0.8 if metric == "C_L" else (0.7 if metric == "I_y" else 0.5),
                       color="green", ls=":", alpha=0.6)
            if row == 0 and col == 2:
                ax.legend(fontsize=8)
    fig.suptitle("Fraction of encounters within tolerance vs K (H=16, 3 modes)")
    fig.tight_layout()
    fig.savefig(FIGS / "exp5_tolerance_envelope.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[exp5] wrote {FIGS / 'exp5_tolerance_envelope.png'}")


if __name__ == "__main__":
    main()
