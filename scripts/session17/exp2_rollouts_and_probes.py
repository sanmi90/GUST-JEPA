"""Session 17, Experiment 2 (light version): rollouts + linear probes on z.

Avoids the SL decoder bottleneck by using linear z->observable probes for the
physical metrics. Train probes for C_L, I_y, I_x, wake_enstrophy on the
production train pool (where DNS metrics are computed exactly), then apply
to the predicted z trajectories.

Outputs:
    outputs/session17/exp2/rollout_metrics_per_encounter.npz
    outputs/session17/exp2/probe_train_quality.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import h5py
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
SPLIT_MANIFEST = REPO / "configs" / "splits" / "split_v2.json"
PARTITION = "v1"
DEFAULT_IMPACT_FRAME = 40
OUT = REPO / "outputs" / "session17" / "exp2"
OUT.mkdir(parents=True, exist_ok=True)
CACHE_ROOT = Path(
    os.environ.get(
        "VORTEX_JEPA_CACHE",
        str(Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
            / "data" / "processed" / "vortex-jepa"),
    )
)


def gather_split_encounters(split: str) -> list[dict]:
    with open(SPLIT_MANIFEST) as f:
        manifest = json.load(f)
    out = []
    for cid, case in manifest["cases"].items():
        if split == "test_b" and case["split"] == "test_b":
            ks = list(range(int(case["n_encounters_full"])))
        elif split == "test_c" and case["split"] == "test_c":
            ks = list(range(int(case["n_encounters_full"])))
        else:
            continue
        for k in ks:
            path = CACHE_ROOT / PARTITION / cid / f"encounter_{int(k):02d}.h5"
            if not path.exists():
                continue
            out.append({
                "case_id": cid, "k": int(k), "path": path,
                "G": float(case.get("G", 0.0)),
                "D": float(case.get("D", 0.0)),
                "Y": float(case.get("Y", 0.0)),
            })
    return out


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
    enc.eval().to(device)
    pred.eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    for p in pred.parameters():
        p.requires_grad_(False)
    return enc, pred


def make_markov_attention_forward():
    def markov_forward(self: CausalSelfAttentionWithRoPE, x: Tensor) -> Tensor:
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
def markov_attention(model: AutoregressivePredictor) -> Iterator[None]:
    new_forward = make_markov_attention_forward()
    originals = []
    for module in model.modules():
        if isinstance(module, CausalSelfAttentionWithRoPE):
            originals.append((module, module.forward))
            module.forward = new_forward.__get__(module, type(module))
    try:
        yield
    finally:
        for module, original in originals:
            module.forward = original


@torch.no_grad()
def encode_full(enc, omega_norm, device):
    x = torch.from_numpy(omega_norm).to(device).unsqueeze(0).unsqueeze(2)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        z = enc(x)
    return z.float().squeeze(0)


@torch.no_grad()
def rollout_markov_only(pred, z_impact, cond, steps, device):
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
def rollout_autoregressive(pred, z_seed, cond, steps, device):
    max_seq = int(pred.max_seq_len)
    if z_seed.dim() == 2:
        z_seed = z_seed.unsqueeze(0)
    z_full = z_seed.clone()
    for _ in range(steps):
        ctx = z_full[:, -max_seq:, :]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            z_hat = pred(ctx, cond)
        z_full = torch.cat([z_full, z_hat[:, -1:, :].float()], dim=1)
    return z_full


def fit_linear_probe(Z: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> dict:
    """Ridge regression: returns {'W', 'mu_z', 'sigma_z', 'b'} for inference."""
    Z = Z.astype(np.float64)
    y = y.astype(np.float64)
    mu = Z.mean(axis=0)
    sigma = Z.std(axis=0).clip(min=1e-9)
    Zn = (Z - mu) / sigma
    A = Zn.T @ Zn + alpha * np.eye(Zn.shape[1])
    yc = y - y.mean()
    W = np.linalg.solve(A, Zn.T @ yc)
    return {"W": W, "mu_z": mu, "sigma_z": sigma, "b": float(y.mean())}


def apply_probe(z: np.ndarray, probe: dict) -> np.ndarray:
    Zn = (z.astype(np.float64) - probe["mu_z"]) / probe["sigma_z"]
    return Zn @ probe["W"] + probe["b"]


def train_probes_for_all_metrics() -> dict:
    """Fit z -> {C_L, I_y, I_x, enstrophy, circ_pos, circ_neg} probes on production
    train per-frame data."""
    # Load DNS metrics (per_frame for train)
    dns = np.load(OUT / "dns_physical_metrics.npz", allow_pickle=True)
    # Load production train z_full
    train_lat = np.load(
        REPO / "outputs" / "session17" / "seed_latents" / "production" / "train.npz",
        allow_pickle=True,
    )
    # Match (case_id, encounter_index) between train z and DNS metrics
    z_cid = train_lat["case_id"].astype(str)
    z_ei = train_lat["encounter_index"].astype(int)
    d_cid = dns["train_case_id"].astype(str)
    d_ei = dns["train_encounter_index"].astype(int)
    # Build matched-index mapping
    z_index = {(c, e): i for i, (c, e) in enumerate(zip(z_cid, z_ei))}
    matched_z = []
    matched_metrics = {k: [] for k in ("C_L", "I_y", "I_x", "wake_enstrophy",
                                       "circulation_pos", "circulation_neg")}
    impact_train = dns["train_impact_frame"]
    # WARNING: production train.npz here has only impact-frame z (no z_full).
    # Use session14 train.npz for z_full instead.
    train_full = np.load(
        REPO / "outputs" / "session14" / "latents" / "S12_E_d64" / "train.npz",
        allow_pickle=True,
    )
    z_full_train = train_full["z_full"].astype(np.float32)
    z_full_cid = train_full["case_id"].astype(str)
    z_full_ei = train_full["encounter_index"].astype(int)
    z_full_map = {(c, e): i for i, (c, e) in enumerate(zip(z_full_cid, z_full_ei))}

    Z_all = []
    Y_all = {k: [] for k in matched_metrics}
    for j, (cid, ei) in enumerate(zip(d_cid, d_ei)):
        if (cid, ei) not in z_full_map:
            continue
        zi = z_full_map[(cid, ei)]
        zf = z_full_train[zi]  # (120, 64)
        Z_all.append(zf)
        for k in matched_metrics:
            Y_all[k].append(dns[f"train_{k}"][j])

    Z_all = np.concatenate(Z_all, axis=0)  # (n_enc * 120, 64)
    Y_stacked = {k: np.concatenate(v) for k, v in Y_all.items()}
    print(
        f"[exp2-light] probe training: n_samples={Z_all.shape[0]}  d={Z_all.shape[1]}"
    )

    probes = {}
    train_r2 = {}
    for metric, y in Y_stacked.items():
        # Drop NaN frames if any
        valid = ~np.isnan(y)
        Z_v = Z_all[valid]
        y_v = y[valid]
        p = fit_linear_probe(Z_v, y_v, alpha=1.0)
        probes[metric] = p
        # Train R^2
        pred = apply_probe(Z_v, p)
        ss_res = float(((y_v - pred) ** 2).sum())
        ss_tot = float(((y_v - y_v.mean()) ** 2).sum())
        train_r2[metric] = 1.0 - ss_res / max(ss_tot, 1e-12)
        print(f"[exp2-light] probe {metric:20s}: train R^2 = {train_r2[metric]:+.3f}")
    return {"probes": probes, "train_r2": train_r2}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--splits", nargs="+", default=["test_b", "test_c"])
    args = p.parse_args()

    device = require_rtx6000(gpu_index=args.gpu)
    print(f"[exp2-light] device={device}")

    probe_bundle = train_probes_for_all_metrics()
    probes = probe_bundle["probes"]
    (OUT / "probe_train_quality.json").write_text(
        json.dumps(probe_bundle["train_r2"], indent=2)
    )

    enc, pred_model = load_encoder_predictor(device)
    pipeline = OmegaPipeline.from_manifest(OMEGA_MANIFEST)

    dns = np.load(OUT / "dns_physical_metrics.npz", allow_pickle=True)

    save: dict = {}
    for split in args.splits:
        encs = gather_split_encounters(split)
        print(f"\n[exp2-light] split={split}: {len(encs)} encounters")
        t0 = time.time()
        # Pre-extract DNS metrics aligned with encounters by (case_id, k)
        dns_cid = dns[f"{split}_case_id"].astype(str)
        dns_ei = dns[f"{split}_encounter_index"].astype(int)
        dns_map = {(c, e): i for i, (c, e) in enumerate(zip(dns_cid, dns_ei))}

        per = {
            "case_id": [], "encounter_index": [],
            "G": [], "D": [], "Y": [],
            "impact_frame": [], "H_max": [],
        }
        for metric in ("CL", "I_y", "I_x", "enstrophy", "circ_pos", "circ_neg"):
            for path in ("dns", "markov", "ar", "full"):
                per[f"{path}_{metric}"] = []  # per_encounter lists of arrays

        # Metric name conversion (dns metrics use full names)
        dns_metric_keys = {
            "CL": "C_L", "I_y": "I_y", "I_x": "I_x",
            "enstrophy": "wake_enstrophy",
            "circ_pos": "circulation_pos", "circ_neg": "circulation_neg",
        }
        probe_metric_keys = {
            "CL": "C_L", "I_y": "I_y", "I_x": "I_x",
            "enstrophy": "wake_enstrophy",
            "circ_pos": "circulation_pos", "circ_neg": "circulation_neg",
        }

        for i, e in enumerate(encs):
            with h5py.File(e["path"], "r") as f:
                omega_raw = np.asarray(f["omega_z"], dtype=np.float32)
                impact = int(f.attrs.get("impact_frame_estimate", DEFAULT_IMPACT_FRAME))
            omega_clean = pipeline.preprocess_raw(omega_raw, e["case_id"], e["k"])
            omega_norm = pipeline.normalize(omega_clean).astype(np.float32)
            T_full = omega_norm.shape[0]
            if impact >= T_full - 1:
                continue
            z_dns_full = encode_full(enc, omega_norm, device)  # (T, 64)
            cond = torch.tensor([[e["G"], e["D"], e["Y"]]], dtype=torch.float32, device=device)
            H_max = T_full - impact - 1

            z_imp = z_dns_full[impact:impact + 1]
            z_markov = rollout_markov_only(pred_model, z_imp, cond, H_max, device).squeeze(0)
            z_ar = rollout_autoregressive(pred_model, z_imp, cond, H_max, device).squeeze(0)
            max_seq = int(pred_model.max_seq_len)
            seed_start = max(0, impact - max_seq + 1)
            z_seed = z_dns_full[seed_start:impact + 1]
            z_full_pred = rollout_autoregressive(pred_model, z_seed, cond, H_max, device).squeeze(0)

            # Slice post-impact z trajectories
            z_dns_post = z_dns_full[impact + 1: impact + 1 + H_max].cpu().numpy()
            z_markov_post = z_markov[1:].cpu().numpy()
            z_ar_post = z_ar[1:].cpu().numpy()
            z_full_post = z_full_pred[z_seed.shape[0]:].cpu().numpy()

            # Apply probes to each (predicted) trajectory
            per["case_id"].append(e["case_id"])
            per["encounter_index"].append(e["k"])
            per["G"].append(e["G"])
            per["D"].append(e["D"])
            per["Y"].append(e["Y"])
            per["impact_frame"].append(impact)
            per["H_max"].append(H_max)

            # DNS truth from cached DNS metrics
            di = dns_map[(e["case_id"], e["k"])]
            for metric, dns_key in dns_metric_keys.items():
                dns_arr = dns[f"{split}_{dns_key}"][di]  # (120,)
                # post-impact slice
                per[f"dns_{metric}"].append(dns_arr[impact + 1: impact + 1 + H_max])

            # Predictions: probe(z_post) for each mode
            for path, z_post in (
                ("markov", z_markov_post),
                ("ar", z_ar_post),
                ("full", z_full_post),
            ):
                for metric, probe_key in probe_metric_keys.items():
                    pred = apply_probe(z_post, probes[probe_key])
                    per[f"{path}_{metric}"].append(pred)

            if (i + 1) % 10 == 0:
                print(f"[exp2-light] {split} {i+1}/{len(encs)}  ({(time.time()-t0)/(i+1):.2f}s/enc)")
        print(f"[exp2-light] {split} done in {time.time() - t0:.1f}s")

        # Pad to T_pad=80 with NaN and save
        T_pad = 80
        save[f"{split}_case_id"] = np.asarray(per["case_id"])
        save[f"{split}_encounter_index"] = np.asarray(per["encounter_index"])
        save[f"{split}_G"] = np.asarray(per["G"], dtype=np.float64)
        save[f"{split}_D"] = np.asarray(per["D"], dtype=np.float64)
        save[f"{split}_Y"] = np.asarray(per["Y"], dtype=np.float64)
        save[f"{split}_impact_frame"] = np.asarray(per["impact_frame"], dtype=np.int32)
        save[f"{split}_H_max"] = np.asarray(per["H_max"], dtype=np.int32)
        for key, val in per.items():
            if key in ("case_id", "encounter_index", "G", "D", "Y", "impact_frame", "H_max"):
                continue
            arr = np.full((len(val), T_pad), np.nan, dtype=np.float64)
            for j, row in enumerate(val):
                L = min(len(row), T_pad)
                arr[j, :L] = row[:L]
            save[f"{split}_{key}"] = arr

    np.savez_compressed(OUT / "rollout_metrics_per_encounter.npz", **save)
    print(f"\n[exp2-light] wrote {OUT / 'rollout_metrics_per_encounter.npz'}")


if __name__ == "__main__":
    main()
