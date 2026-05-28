"""Session 17, Experiment 2: physical Markov closure.

For each Test B / Test C encounter (v1 split):
  1. Encode DNS omega -> z_dns
  2. Run 3 rollouts at cond=true: Markov-only, AR-from-impact, Full-context
  3. Decode each rollout's z_full through the SL decoder -> omega_pred
  4. Compute per-frame physical metrics on each decoded path
     (C_L from probe on z; I_y, I_x, enstrophy, circulation from omega)
  5. Compare to DNS reference (cached omega + HDF5 C_L)

The headline metrics at horizon H:
  - C_L_rollout vs C_L_DNS
  - I_y_rollout vs I_y_DNS (where I_y = integral x * omega dA)
  - wake enstrophy
Per-encounter bootstrap (95% CI) of (Markov - Full) deltas tells us at which
horizon Markov-only diverges from Full-context.

Output:
    outputs/session17/exp2/physical_metrics_per_encounter.npz
    outputs/session17/exp2/exp2_horizon_summary.json
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
from src.models.lap_film_decoder import LapFiLMDecoder  # noqa: E402
from src.models.rope import apply_rope  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402


ENCODER_CKPT = (
    REPO / "outputs" / "runs" / "session12" / "S12_E_d64" / "encoder"
    / "checkpoint_iter020000.pt"
)
DECODER_CKPT = (
    REPO / "outputs" / "runs" / "session12" / "S12_E_d64" / "encoder"
    / "decoder_specloss_recipe" / "decoder_iter012000.pt"
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
# Physical extent: x in [-1.5, +4.5] (H=192), y in [-1.5, +1.5] (W=96).
DX = 6.0 / 192
DY = 3.0 / 96
X_GRID = np.linspace(-1.5, 4.5, 192).astype(np.float32)
Y_GRID = np.linspace(-1.5, 1.5, 96).astype(np.float32)
# Wake-region mask: x in [0.5, 4], |y| < 1
WAKE_X_MIN, WAKE_X_MAX = 0.5, 4.0
WAKE_Y_MAX = 1.0
HORIZONS = (1, 4, 8, 16, 24, 32, 48, 64, 79)


def gather_split_encounters(split: str) -> list[dict]:
    with open(SPLIT_MANIFEST) as f:
        manifest = json.load(f)
    out: list[dict] = []
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


def load_encoder_predictor(device: torch.device):
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


def load_decoder(device: torch.device, enc_args: dict) -> LapFiLMDecoder:
    blob = torch.load(DECODER_CKPT, map_location="cpu", weights_only=False)
    dargs = blob["args"]
    bc = int(dargs.get("decoder_base_ch", 64))
    channels = (bc, bc, int(bc * 0.75), int(bc * 0.5), int(bc * 0.375))
    dec = LapFiLMDecoder(
        latent_dim=int(enc_args["d"]),
        channels=channels,
        resblocks_per_level=int(dargs.get("decoder_resblocks_per_level", 2)),
        upsample=dargs.get("decoder_upsample", "pixelshuffle"),
        fourier_bands=int(dargs.get("decoder_fourier_bands") or 4),
        use_film=bool(dargs.get("decoder_use_film", True)),
        airfoil_mask_path=dargs.get("airfoil_mask_path"),
    )
    dec.load_state_dict(blob["decoder_state_dict"])
    dec.eval().to(device)
    for p in dec.parameters():
        p.requires_grad_(False)
    return dec


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


@torch.no_grad()
def decode_z_sequence(z: Tensor, dec: LapFiLMDecoder, pipeline: OmegaPipeline,
                     device, batch_size: int = 16) -> np.ndarray:
    """z: (T, d). Decode to (T, 192, 96) raw-scale omega."""
    T = z.shape[0]
    out = np.zeros((T, 192, 96), dtype=np.float32)
    for b in range(0, T, batch_size):
        zb = z[b : b + batch_size].unsqueeze(0).to(device)  # (1, B, d)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = dec(zb)
        if isinstance(pred, dict):
            pred = pred["pred"]
        pred = pred.float().squeeze(0)  # (B, ..., H, W)
        while pred.dim() > 3:
            pred = pred.squeeze(0)
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
        pred_np = pred.cpu().numpy()
        if pred_np.shape != (zb.shape[1], 192, 96):
            # handle the alternate (B, 1, H, W) shape
            pred_np = pred_np.reshape(-1, 192, 96)
        for i in range(pred_np.shape[0]):
            out[b + i] = pipeline.unnormalize(pred_np[i])
    return out


def compute_physical_metrics(omega: np.ndarray) -> dict:
    """omega: (T, H=192, W=96) raw scale. Return per-frame metrics."""
    T = omega.shape[0]
    # Build x grid (H direction) and y grid (W direction).
    xx = X_GRID[:, None]  # (192, 1)
    yy = Y_GRID[None, :]  # (1, 96)
    # Vorticity impulses
    I_x = np.zeros(T)
    I_y = np.zeros(T)
    # Wake enstrophy and circulation
    wake_mask = (xx >= WAKE_X_MIN) & (xx <= WAKE_X_MAX) & (np.abs(yy) <= WAKE_Y_MAX)
    wake_enstrophy = np.zeros(T)
    circulation_pos = np.zeros(T)
    circulation_neg = np.zeros(T)
    for t in range(T):
        w = omega[t]
        # I_x = -integral y * omega dA  (sign convention from plan)
        I_x[t] = -np.sum(yy * w * DX * DY)
        # I_y = integral x * omega dA
        I_y[t] = np.sum(xx * w * DX * DY)
        ww = w * wake_mask
        wake_enstrophy[t] = float((ww**2).sum() * DX * DY)
        pos = ww[ww > 1.0]  # threshold to avoid noise; circulation of positive vortices
        neg = ww[ww < -1.0]
        circulation_pos[t] = float(pos.sum() * DX * DY)
        circulation_neg[t] = float(neg.sum() * DX * DY)
    return {
        "I_x": I_x,
        "I_y": I_y,
        "wake_enstrophy": wake_enstrophy,
        "circulation_pos": circulation_pos,
        "circulation_neg": circulation_neg,
    }


def train_z_to_CL_probe(train_z_full: np.ndarray, train_CL: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a linear z -> C_L on per-frame data; return (W, b) such that C_L = z @ W + b."""
    Z = train_z_full.reshape(-1, train_z_full.shape[-1]).astype(np.float64)
    y = train_CL.reshape(-1).astype(np.float64)
    # Ridge with alpha=1.0 on standardised features.
    mu = Z.mean(axis=0)
    sigma = Z.std(axis=0).clip(min=1e-9)
    Zn = (Z - mu) / sigma
    A = Zn.T @ Zn + 1.0 * np.eye(Zn.shape[1])
    W = np.linalg.solve(A, Zn.T @ (y - y.mean()))
    b = float(y.mean())
    return W, mu, sigma, b


def apply_z_to_CL_probe(z_full: np.ndarray, W: np.ndarray, mu, sigma, b) -> np.ndarray:
    """z_full: (T, d). Return (T,) predicted C_L."""
    Zn = (z_full.astype(np.float64) - mu) / sigma
    return Zn @ W + b


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--splits", nargs="+", default=["test_b", "test_c"])
    p.add_argument("--limit-encounters", type=int, default=None)
    args = p.parse_args()

    device = require_rtx6000(gpu_index=args.gpu)
    print(f"[exp2] device={device}")

    enc, pred_model = load_encoder_predictor(device)
    enc_blob = torch.load(ENCODER_CKPT, map_location="cpu", weights_only=False)
    dec = load_decoder(device, enc_blob["args"])
    pipeline = OmegaPipeline.from_manifest(OMEGA_MANIFEST)
    print(
        f"[exp2] loaded enc/pred/dec; pred max_seq_len={pred_model.max_seq_len}"
    )

    # Train z -> C_L probe on production train pool.
    train_per_frame = np.load(
        REPO / "outputs" / "session16" / "exp2" / "per_frame_targets" / "train.npz",
        allow_pickle=True,
    )
    W, mu, sigma, b = train_z_to_CL_probe(
        train_per_frame["z_full"], train_per_frame["C_L"]
    )
    # Sanity: train R^2
    pred_train = apply_z_to_CL_probe(
        train_per_frame["z_full"].reshape(-1, train_per_frame["z_full"].shape[-1]),
        W, mu, sigma, b,
    )
    true_train = train_per_frame["C_L"].reshape(-1)
    ss_res = ((true_train - pred_train) ** 2).sum()
    ss_tot = ((true_train - true_train.mean()) ** 2).sum()
    print(f"[exp2] linear z->C_L probe train R^2 = {1 - ss_res / ss_tot:.3f}")

    # Per-split processing
    all_records = {}
    for split in args.splits:
        encs = gather_split_encounters(split)
        if args.limit_encounters is not None:
            encs = encs[: args.limit_encounters]
        print(f"\n[exp2] split={split}: {len(encs)} encounters")

        per_enc = {
            "case_id": [],
            "encounter_index": [],
            "G": [], "D": [], "Y": [],
            "impact_frame": [],
            "n_post": [],
            "dns_CL": [],         # (n_post,) per encounter
            "dns_I_y": [],
            "dns_I_x": [],
            "dns_enstrophy": [],
            "dns_circ_pos": [],
            "dns_circ_neg": [],
            # 3 rollout modes, each with the same metrics
            "markov_CL": [], "markov_I_y": [], "markov_I_x": [],
            "markov_enstrophy": [], "markov_circ_pos": [], "markov_circ_neg": [],
            "ar_CL": [], "ar_I_y": [], "ar_I_x": [],
            "ar_enstrophy": [], "ar_circ_pos": [], "ar_circ_neg": [],
            "full_CL": [], "full_I_y": [], "full_I_x": [],
            "full_enstrophy": [], "full_circ_pos": [], "full_circ_neg": [],
        }
        t0 = time.time()
        for i, e in enumerate(encs):
            ti = time.time()
            with h5py.File(e["path"], "r") as f:
                omega_raw = np.asarray(f["omega_z"], dtype=np.float32)
                impact = int(f.attrs.get("impact_frame_estimate", DEFAULT_IMPACT_FRAME))
                CL_dns = np.asarray(f["C_L"], dtype=np.float32)
            print(f"[exp2] enc {i} {e['case_id']} k={e['k']}: loaded h5 in {time.time()-ti:.2f}s", flush=True)
            tx = time.time()
            omega_clean = pipeline.preprocess_raw(omega_raw, e["case_id"], e["k"])
            omega_norm = pipeline.normalize(omega_clean).astype(np.float32)
            T_full = omega_norm.shape[0]
            print(f"[exp2]   preproc in {time.time()-tx:.2f}s, T={T_full}", flush=True)
            if impact >= T_full - 1:
                continue
            tx = time.time()
            z_dns = encode_full(enc, omega_norm, device)  # (T, 64)
            print(f"[exp2]   encode in {time.time()-tx:.2f}s", flush=True)
            cond = torch.tensor([[e["G"], e["D"], e["Y"]]], dtype=torch.float32, device=device)
            H_max = T_full - impact - 1

            # Markov
            tx = time.time()
            z_imp = z_dns[impact:impact + 1]
            z_markov = rollout_markov_only(pred_model, z_imp, cond, H_max, device).squeeze(0)
            print(f"[exp2]   markov rollout in {time.time()-tx:.2f}s (H_max={H_max})", flush=True)
            # AR
            tx = time.time()
            z_ar = rollout_autoregressive(pred_model, z_imp, cond, H_max, device).squeeze(0)
            print(f"[exp2]   AR rollout in {time.time()-tx:.2f}s", flush=True)
            # Full
            tx = time.time()
            max_seq = int(pred_model.max_seq_len)
            seed_start = max(0, impact - max_seq + 1)
            z_seed = z_dns[seed_start:impact + 1]
            z_full = rollout_autoregressive(pred_model, z_seed, cond, H_max, device).squeeze(0)
            print(f"[exp2]   full rollout in {time.time()-tx:.2f}s", flush=True)

            # Decode the post-impact frames for each path (impact + 1 onward).
            z_markov_post = z_markov[1:]  # (H_max, 64)
            z_ar_post = z_ar[1:]
            z_full_post = z_full[z_seed.shape[0]:]  # drop the seed
            z_dns_post = z_dns[impact + 1: impact + 1 + H_max]  # (H_max, 64)

            # Decode each
            om_dns_post = omega_raw[impact + 1: impact + 1 + H_max]  # DNS cached raw omega
            tx = time.time()
            om_markov_post = decode_z_sequence(z_markov_post, dec, pipeline, device)
            print(f"[exp2]   decode markov in {time.time()-tx:.2f}s shape={om_markov_post.shape}", flush=True)
            tx = time.time()
            om_ar_post = decode_z_sequence(z_ar_post, dec, pipeline, device)
            print(f"[exp2]   decode AR in {time.time()-tx:.2f}s", flush=True)
            tx = time.time()
            om_full_post = decode_z_sequence(z_full_post, dec, pipeline, device)
            print(f"[exp2]   decode full in {time.time()-tx:.2f}s", flush=True)

            # DNS metrics from cached raw omega
            dns_metrics = compute_physical_metrics(om_dns_post)
            mk_metrics = compute_physical_metrics(om_markov_post)
            ar_metrics = compute_physical_metrics(om_ar_post)
            fc_metrics = compute_physical_metrics(om_full_post)

            # C_L from probe on each path's z; DNS from HDF5
            CL_dns_post = CL_dns[impact + 1: impact + 1 + H_max].astype(np.float64)
            CL_markov = apply_z_to_CL_probe(z_markov_post.cpu().numpy(), W, mu, sigma, b)
            CL_ar = apply_z_to_CL_probe(z_ar_post.cpu().numpy(), W, mu, sigma, b)
            CL_full = apply_z_to_CL_probe(z_full_post.cpu().numpy(), W, mu, sigma, b)

            per_enc["case_id"].append(e["case_id"])
            per_enc["encounter_index"].append(e["k"])
            per_enc["G"].append(e["G"])
            per_enc["D"].append(e["D"])
            per_enc["Y"].append(e["Y"])
            per_enc["impact_frame"].append(impact)
            per_enc["n_post"].append(H_max)

            per_enc["dns_CL"].append(CL_dns_post)
            per_enc["dns_I_y"].append(dns_metrics["I_y"])
            per_enc["dns_I_x"].append(dns_metrics["I_x"])
            per_enc["dns_enstrophy"].append(dns_metrics["wake_enstrophy"])
            per_enc["dns_circ_pos"].append(dns_metrics["circulation_pos"])
            per_enc["dns_circ_neg"].append(dns_metrics["circulation_neg"])
            per_enc["markov_CL"].append(CL_markov)
            per_enc["markov_I_y"].append(mk_metrics["I_y"])
            per_enc["markov_I_x"].append(mk_metrics["I_x"])
            per_enc["markov_enstrophy"].append(mk_metrics["wake_enstrophy"])
            per_enc["markov_circ_pos"].append(mk_metrics["circulation_pos"])
            per_enc["markov_circ_neg"].append(mk_metrics["circulation_neg"])
            per_enc["ar_CL"].append(CL_ar)
            per_enc["ar_I_y"].append(ar_metrics["I_y"])
            per_enc["ar_I_x"].append(ar_metrics["I_x"])
            per_enc["ar_enstrophy"].append(ar_metrics["wake_enstrophy"])
            per_enc["ar_circ_pos"].append(ar_metrics["circulation_pos"])
            per_enc["ar_circ_neg"].append(ar_metrics["circulation_neg"])
            per_enc["full_CL"].append(CL_full)
            per_enc["full_I_y"].append(fc_metrics["I_y"])
            per_enc["full_I_x"].append(fc_metrics["I_x"])
            per_enc["full_enstrophy"].append(fc_metrics["wake_enstrophy"])
            per_enc["full_circ_pos"].append(fc_metrics["circulation_pos"])
            per_enc["full_circ_neg"].append(fc_metrics["circulation_neg"])

            if (i + 1) % 5 == 0:
                print(
                    f"[exp2] {split} {i+1}/{len(encs)}  "
                    f"({(time.time() - t0)/(i+1):.1f}s/enc)"
                )
        all_records[split] = per_enc
        print(f"[exp2] {split} done in {time.time() - t0:.1f}s")

    # Save to NPZ.
    save: dict = {}
    for split, per in all_records.items():
        for k, v in per.items():
            if k in ("case_id", "encounter_index", "G", "D", "Y", "impact_frame", "n_post"):
                save[f"{split}_{k}"] = np.asarray(v)
            else:
                # list of arrays per encounter (different lengths)
                # Pad to T=80 with NaN.
                T_pad = 80
                arr = np.full((len(v), T_pad), np.nan, dtype=np.float64)
                for ii, row in enumerate(v):
                    L = min(len(row), T_pad)
                    arr[ii, :L] = row[:L]
                save[f"{split}_{k}"] = arr
    np.savez_compressed(OUT / "physical_metrics_per_encounter.npz", **save)
    print(f"[exp2] wrote {OUT / 'physical_metrics_per_encounter.npz'}")


if __name__ == "__main__":
    main()
