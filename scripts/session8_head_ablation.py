"""Session 8 Step 2: auxiliary-head ablation on R3 iter-20000.

Compares three CL_future prediction methods on Test B:
1. Fresh linear probe on z, fit on Test A latents, evaluated on Test B
   (the Session 7 method; the baseline that gave the +0.14 Test B delta).
2. The trained R3 observable_head applied directly to Test B latents.
3. Fresh probe for a different observable (drag C_D, leading-edge p_LE),
   to test whether R3's latent encodes general flow state or CL-specific
   structure.

Method 3 uses the cache's per-frame C_D and the leading-edge pressure
sensor from p_wall (the surface point closest to the LE). Both observables
are predicted with the same delta=(8, 16, 24) frame offsets so the head
analysis is the same shape as the CL pipeline.

Output: ``outputs/runs/session8/head_ablation.csv`` plus an info string
on which method maps to which interpretation row in the plan's matrix.

Runs on cuda:1 (RTX 6000 #2) to avoid Step 3 R3-seed=42 on cuda:0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.models.observable_head import ObservableHead  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402


SESSION7_ROOT = REPO / "outputs" / "runs" / "session7"
R3_DIR = SESSION7_ROOT / "run_r3_sigreg_obs_bn"
R3_CKPT_PATH = R3_DIR / "checkpoint_iter020000.pt"
DELTAS = (8, 16, 24)
D = 32

PREVENT = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", PREVENT / "data" / "processed" / "vortex-jepa"))


def gather_encounters(split_key: str) -> list[dict]:
    """Load encounters with omega, C_L, C_D, and the LE-station p_wall trace."""
    with open(REPO / "configs" / "splits" / "split_v2.json") as f:
        manifest = json.load(f)
    encs = []
    for cid, case in manifest["cases"].items():
        if split_key == "test_a" and case["split"] == "train":
            ks = (case.get("val_encounter_indices") or case["test_a_encounter_indices"])
        elif split_key == "test_b" and case["split"] == "test_b":
            ks = list(range(case["n_encounters_full"]))
        else:
            continue
        for k in ks:
            path = CACHE / "v1" / cid / f"encounter_{k:02d}.h5"
            with h5py.File(path, "r") as g:
                p_wall = np.asarray(g["p_wall"], dtype=np.float32)  # (T, 192)
                # Leading-edge index 0 is the point closest to LE for this airfoil.
                p_le = p_wall[:, 0]
                encs.append({
                    "case_id": cid,
                    "k": int(k),
                    "omega_z": np.asarray(g["omega_z"], dtype=np.float32),
                    "C_L": np.asarray(g["C_L"], dtype=np.float32),
                    "C_D": np.asarray(g["C_D"], dtype=np.float32),
                    "p_LE": p_le,
                    "G": float(g.attrs["G"]),
                    "D": float(g.attrs["D"]),
                    "Y": float(g.attrs["Y"]),
                })
    return encs


def build_future(encs: list[dict], series_key: str) -> np.ndarray:
    N = len(encs)
    T = encs[0]["omega_z"].shape[0]
    out = np.empty((N, T, len(DELTAS)), dtype=np.float32)
    for i, e in enumerate(encs):
        s = e[series_key]
        for j, d in enumerate(DELTAS):
            for t in range(T):
                src = t + d
                out[i, t, j] = s[src] if src < s.shape[0] else s[-1]
    return out


def case_of(encs: list[dict]) -> np.ndarray:
    return np.array([e["case_id"] for e in encs])


def c_of(encs: list[dict]) -> np.ndarray:
    return np.stack([[e["G"], e["D"], e["Y"]] for e in encs], axis=0).astype(np.float32)


def encode_split(enc: HybridCNNViTEncoder, encs: list[dict], device: torch.device) -> np.ndarray:
    N = len(encs)
    T = encs[0]["omega_z"].shape[0]
    out = np.empty((N, T, D), dtype=np.float32)
    with torch.no_grad():
        for i, e in enumerate(encs):
            x = torch.from_numpy(e["omega_z"]).unsqueeze(1).to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                z = enc(x.unsqueeze(0))
            out[i] = z.squeeze(0).float().cpu().numpy()
    return out


def fit_mlp_oos(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray,
                seed: int = 0, hidden: int = 64, epochs: int = 400, lr: float = 1e-2) -> tuple[float, float]:
    """Returns (r2_overall, r2_eval_independent_baseline)."""
    torch.manual_seed(seed)
    in_dim = X_train.shape[1]
    out_dim = y_train.shape[1]
    X_tr = torch.from_numpy(X_train.astype(np.float32))
    y_tr = torch.from_numpy(y_train.astype(np.float32))
    X_ev = torch.from_numpy(X_eval.astype(np.float32))
    y_ev = torch.from_numpy(y_eval.astype(np.float32))
    model = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        p = model(X_tr)
        loss = F.mse_loss(p, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        p_ev = model(X_ev)
    ss_res = ((y_ev - p_ev) ** 2).sum(dim=0)
    ss_tot = ((y_ev - y_ev.mean(dim=0)) ** 2).sum(dim=0).clamp_min(1e-8)
    return float((1.0 - ss_res / ss_tot).mean().item()), 0.0  # baseline unused here


def fit_mlp(X: np.ndarray, y: np.ndarray, seed: int = 0, hidden: int = 64,
            epochs: int = 400, lr: float = 1e-2) -> float:
    torch.manual_seed(seed)
    n, in_dim = X.shape
    out_dim = y.shape[1]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    fit, ev = perm[: int(0.75 * n)], perm[int(0.75 * n):]
    X_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32))
    model = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        p = model(X_t[fit])
        loss = F.mse_loss(p, y_t[fit])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        p_ev = model(X_t[ev])
    ss_res = ((y_t[ev] - p_ev) ** 2).sum(dim=0)
    ss_tot = ((y_t[ev] - y_t[ev].mean(dim=0)) ** 2).sum(dim=0).clamp_min(1e-8)
    return float((1.0 - ss_res / ss_tot).mean().item())


def trained_head_r2(z_eval: np.ndarray, y_eval: np.ndarray,
                    head: ObservableHead, device: torch.device) -> float:
    """Evaluate the trained R3 observable head directly on Test B latents.

    The head outputs CL(t + delta) for delta in DELTAS; reduction is the
    averaged per-delta r2 over the evaluation set, matching the MLP fits
    above so the numbers are comparable.
    """
    head = head.eval().to(device)
    with torch.no_grad():
        z_t = torch.from_numpy(z_eval.astype(np.float32)).to(device)
        pred = head(z_t).cpu().numpy()
    p_ev = torch.from_numpy(pred)
    y_ev = torch.from_numpy(y_eval.astype(np.float32))
    ss_res = ((y_ev - p_ev) ** 2).sum(dim=0)
    ss_tot = ((y_ev - y_ev.mean(dim=0)) ** 2).sum(dim=0).clamp_min(1e-8)
    return float((1.0 - ss_res / ss_tot).mean().item())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Session 8 Step 2: auxiliary-head ablation")
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument(
        "--output",
        type=str,
        default="outputs/runs/session8/head_ablation.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)
    print(f"[head-ablation] device={device}", flush=True)

    print("[head-ablation] loading R3 iter-20000 checkpoint", flush=True)
    blob = torch.load(R3_CKPT_PATH, map_location="cpu", weights_only=False)
    enc = HybridCNNViTEncoder(
        latent_dim=int(blob["args"]["d"]),
        projection_norm=blob["args"].get("projection_norm", "batchnorm"),
    )
    enc_state = {
        k.removeprefix("encoder."): v
        for k, v in blob["jepa_state_dict"].items()
        if k.startswith("encoder.")
    }
    enc.load_state_dict(enc_state, strict=False)
    enc.eval().to(device)

    head = ObservableHead(
        latent_dim=int(blob["args"]["d"]),
        n_deltas=len(DELTAS),
    )
    head_state = {
        k.removeprefix("observable_head."): v
        for k, v in blob["jepa_state_dict"].items()
        if k.startswith("observable_head.")
    }
    head.load_state_dict(head_state, strict=True)
    head.eval().to(device)
    print(f"[head-ablation] loaded encoder ({sum(p.numel() for p in enc.parameters())/1e6:.1f}M params) "
          f"and observable head ({sum(p.numel() for p in head.parameters())/1e3:.0f}K params)",
          flush=True)

    print("[head-ablation] gathering Test A and Test B encounters", flush=True)
    SPLITS = {s: gather_encounters(s) for s in ("test_a", "test_b")}
    for s, encs in SPLITS.items():
        print(f"  {s}: {len(encs)} encounters", flush=True)

    print("[head-ablation] encoding Test A and Test B", flush=True)
    z = {s: encode_split(enc, SPLITS[s], device) for s in SPLITS}
    T = z["test_a"].shape[1]

    case_a = case_of(SPLITS["test_a"])
    case_b = case_of(SPLITS["test_b"])
    c_a = c_of(SPLITS["test_a"])
    c_b = c_of(SPLITS["test_b"])

    rows = []
    for series in ("C_L", "C_D", "p_LE"):
        print(f"[head-ablation] target={series}", flush=True)
        Y_RAW = {s: build_future(SPLITS[s], series) for s in SPLITS}
        MASK = {s: np.isfinite(Y_RAW[s]).reshape(Y_RAW[s].shape[0], -1).all(axis=1)
                for s in SPLITS}
        for s in SPLITS:
            n_drop = int((~MASK[s]).sum())
            if n_drop:
                print(f"  {series} {s}: {n_drop} encounters dropped for non-finite values",
                      flush=True)
        Y = {s: Y_RAW[s][MASK[s]] for s in SPLITS}

        z_a_cl = z["test_a"][MASK["test_a"]]
        z_b_cl = z["test_b"][MASK["test_b"]]
        c_a_cl = c_a[MASK["test_a"]]
        c_b_cl = c_b[MASK["test_b"]]
        y_a = Y["test_a"].reshape(-1, 3)
        y_b = Y["test_b"].reshape(-1, 3)
        z_a_flat = z_a_cl.reshape(-1, D)
        z_b_flat = z_b_cl.reshape(-1, D)
        ct_a_flat = np.hstack([
            np.repeat(c_a_cl, T, axis=0),
            np.tile(np.arange(T)[:, None], (z_a_cl.shape[0], 1)).astype(np.float32),
        ])
        ct_b_flat = np.hstack([
            np.repeat(c_b_cl, T, axis=0),
            np.tile(np.arange(T)[:, None], (z_b_cl.shape[0], 1)).astype(np.float32),
        ])

        # Method 1: fresh probe on z, fit on Test A, eval on Test B (Session 7 method)
        r2_fresh_b, _ = fit_mlp_oos(z_a_flat, y_a, z_b_flat, y_b)
        # (c, t) baseline for the same OOS task
        r2_ct_b, _ = fit_mlp_oos(ct_a_flat, y_a, ct_b_flat, y_b)
        delta_fresh = r2_fresh_b - r2_ct_b

        # Test A in-sample (sanity)
        r2_fresh_a = fit_mlp(z_a_flat, y_a)
        r2_ct_a = fit_mlp(ct_a_flat, y_a)
        delta_fresh_a = r2_fresh_a - r2_ct_a

        # Method 2: trained R3 head, only meaningful for series=C_L (the trained target)
        if series == "C_L":
            r2_trained_b = trained_head_r2(z_b_flat, y_b, head, device)
            delta_trained_b = r2_trained_b - r2_ct_b
            r2_trained_a = trained_head_r2(z_a_flat, y_a, head, device)
            delta_trained_a = r2_trained_a - r2_ct_a
        else:
            r2_trained_b = float("nan")
            delta_trained_b = float("nan")
            r2_trained_a = float("nan")
            delta_trained_a = float("nan")

        rows.append({
            "target": series,
            "test_a_fresh_r2": r2_fresh_a,
            "test_a_ct_r2": r2_ct_a,
            "test_a_delta_fresh": delta_fresh_a,
            "test_a_trained_r2": r2_trained_a,
            "test_a_delta_trained": delta_trained_a,
            "test_b_fresh_r2": r2_fresh_b,
            "test_b_ct_r2": r2_ct_b,
            "test_b_delta_fresh": delta_fresh,
            "test_b_trained_r2": r2_trained_b,
            "test_b_delta_trained": delta_trained_b,
        })

    out_path = REPO / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[head-ablation] wrote {len(df)} rows to {out_path}", flush=True)
    print(df.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
