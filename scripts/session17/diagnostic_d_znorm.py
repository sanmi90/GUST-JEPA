"""Session 17, Diagnostic D: long-horizon conditioning paradox.

Per Session 16 D119-bis: at H>=64 the cond=zero rollout sometimes BEATS the
cond=true rollout in latent RMSE. Hypothesis: explicit AdaLN-Zero conditioning
amplifies systematic prediction errors over many autoregressive steps, while
cond=zero rollouts relax to a stable latent basin.

We compute the histograms of ||z_pred(t)|| at H = 32, 64, 79 for cond=true and
cond=zero Markov-only rollouts on Test B. If cond=true drifts outward (mean
norm growing with H) while cond=zero contracts (mean norm stable or
decreasing), the mechanism is confirmed.

Outputs:
    outputs/session17/diagnostic_d/drift_summary.json
    outputs/session17/diagnostic_d/z_norm_histograms.png
"""

from __future__ import annotations

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
SPLIT_MANIFEST = REPO / "configs" / "splits" / "split_v2.json"
PARTITION = "v1"
DEFAULT_IMPACT_FRAME = 40
OUT = REPO / "outputs" / "session17" / "diagnostic_d"
OUT.mkdir(parents=True, exist_ok=True)
FIGS = REPO / "outputs" / "session17" / "diagnostic_d"
CACHE_ROOT = Path(
    os.environ.get(
        "VORTEX_JEPA_CACHE",
        str(Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
            / "data" / "processed" / "vortex-jepa"),
    )
)
HORIZONS = (32, 64, 79)


def gather_test_b():
    with open(SPLIT_MANIFEST) as f:
        manifest = json.load(f)
    out = []
    for cid, case in manifest["cases"].items():
        if case["split"] != "test_b":
            continue
        for k in range(int(case["n_encounters_full"])):
            path = CACHE_ROOT / PARTITION / cid / f"encounter_{int(k):02d}.h5"
            if path.exists():
                out.append({
                    "case_id": cid, "k": int(k), "path": path,
                    "G": float(case["G"]), "D": float(case["D"]), "Y": float(case["Y"]),
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
def encode_full(enc, omega_norm, device):
    x = torch.from_numpy(omega_norm).to(device).unsqueeze(0).unsqueeze(2)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        z = enc(x)
    return z.float().squeeze(0)


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


def main() -> None:
    device = require_rtx6000(gpu_index=0)
    print(f"[diag-D] device={device}")
    enc, pred_model = load_encoder_predictor(device)
    pipeline = OmegaPipeline.from_manifest(OMEGA_MANIFEST)
    encs = gather_test_b()
    print(f"[diag-D] test_b: {len(encs)} encounters")

    # Per-encounter ||z_pred(t)|| for cond=true and cond=zero Markov rollouts.
    n = len(encs)
    T_MAX = 79
    norms_true = np.full((n, T_MAX + 1), np.nan)  # +1 to include z_impact (norm at t=0)
    norms_zero = np.full((n, T_MAX + 1), np.nan)
    norms_dns = np.full((n, T_MAX + 1), np.nan)
    t0 = time.time()
    for i, e in enumerate(encs):
        with h5py.File(e["path"], "r") as f:
            omega_raw = np.asarray(f["omega_z"], dtype=np.float32)
            impact = int(f.attrs.get("impact_frame_estimate", DEFAULT_IMPACT_FRAME))
        omega_clean = pipeline.preprocess_raw(omega_raw, e["case_id"], e["k"])
        omega_norm = pipeline.normalize(omega_clean).astype(np.float32)
        T_full = omega_norm.shape[0]
        if impact >= T_full - 1:
            continue
        z_dns = encode_full(enc, omega_norm, device)
        cond_true = torch.tensor([[e["G"], e["D"], e["Y"]]], dtype=torch.float32, device=device)
        cond_zero = torch.zeros((1, 3), dtype=torch.float32, device=device)
        z_imp = z_dns[impact:impact + 1]
        H_max = min(T_full - impact - 1, T_MAX)
        # cond=true rollout
        z_pred_true = rollout_markov(pred_model, z_imp, cond_true, H_max, device).squeeze(0)
        # cond=zero rollout
        z_pred_zero = rollout_markov(pred_model, z_imp, cond_zero, H_max, device).squeeze(0)
        # store norms (z_pred_true[0] is z_impact, [1:] are predictions)
        for tt in range(H_max + 1):
            norms_true[i, tt] = float(torch.linalg.norm(z_pred_true[tt]).cpu())
            norms_zero[i, tt] = float(torch.linalg.norm(z_pred_zero[tt]).cpu())
            if impact + tt < T_full:
                norms_dns[i, tt] = float(torch.linalg.norm(z_dns[impact + tt]).cpu())
        if (i + 1) % 5 == 0:
            print(f"[diag-D] {i+1}/{n}  ({(time.time()-t0)/(i+1):.2f}s/enc)")

    print(f"[diag-D] done in {time.time()-t0:.1f}s")

    # Compute mean+std vs horizon
    horizons_full = np.arange(T_MAX + 1)
    summary = {
        "horizon_grid": horizons_full.tolist(),
        "norms_true_mean": np.nanmean(norms_true, axis=0).tolist(),
        "norms_true_std": np.nanstd(norms_true, axis=0).tolist(),
        "norms_zero_mean": np.nanmean(norms_zero, axis=0).tolist(),
        "norms_zero_std": np.nanstd(norms_zero, axis=0).tolist(),
        "norms_dns_mean": np.nanmean(norms_dns, axis=0).tolist(),
        "norms_dns_std": np.nanstd(norms_dns, axis=0).tolist(),
        "horizons_for_histograms": list(HORIZONS),
    }
    # Histograms at H=32, 64, 79
    hist_data = {}
    for H in HORIZONS:
        hist_data[str(H)] = {
            "cond_true": [float(v) for v in norms_true[:, H] if not np.isnan(v)],
            "cond_zero": [float(v) for v in norms_zero[:, H] if not np.isnan(v)],
            "dns": [float(v) for v in norms_dns[:, H] if not np.isnan(v)],
        }
    summary["histogram_data"] = hist_data

    # Verdict: at H=32, 64, 79 compare mean norms
    print("\n[diag-D] mean ||z|| at horizons:")
    print(f"  {'H':>4} {'cond=true':>12} {'cond=zero':>12} {'DNS':>10}")
    for H in HORIZONS:
        mt = np.nanmean(norms_true[:, H])
        mz = np.nanmean(norms_zero[:, H])
        md = np.nanmean(norms_dns[:, H])
        print(f"  {H:>4} {mt:>12.3f} {mz:>12.3f} {md:>10.3f}")

    (OUT / "drift_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[diag-D] wrote {OUT / 'drift_summary.json'}")

    # Figure: 3 histograms (H=32, 64, 79) + curve plot
    fig = plt.figure(figsize=(15, 8))
    # Top row: norms vs H (with IQR bands)
    ax_curve = fig.add_subplot(2, 1, 1)
    h_grid = horizons_full
    for label, arr, color in [
        ("cond=true (Markov)", norms_true, "tab:blue"),
        ("cond=zero (Markov)", norms_zero, "tab:orange"),
        ("DNS", norms_dns, "k"),
    ]:
        mean = np.nanmean(arr, axis=0)
        q25 = np.nanpercentile(arr, 25, axis=0)
        q75 = np.nanpercentile(arr, 75, axis=0)
        ax_curve.fill_between(h_grid, q25, q75, alpha=0.2, color=color)
        ax_curve.plot(h_grid, mean, "-", color=color, lw=1.5, label=label)
    for H in HORIZONS:
        ax_curve.axvline(H, color="gray", lw=0.6, ls="--", alpha=0.6)
    ax_curve.set_xlabel("horizon H (frames post-impact)")
    ax_curve.set_ylabel(r"$\|z(H)\|$")
    ax_curve.set_title("Latent norm vs horizon (mean, IQR), Test B")
    ax_curve.legend()
    ax_curve.grid(alpha=0.3)

    # Bottom row: histograms at H=32, 64, 79
    for col, H in enumerate(HORIZONS):
        ax = fig.add_subplot(2, 3, 3 + col + 1)
        for label, arr, color in [
            ("cond=true", norms_true[:, H], "tab:blue"),
            ("cond=zero", norms_zero[:, H], "tab:orange"),
            ("DNS", norms_dns[:, H], "k"),
        ]:
            valid = arr[~np.isnan(arr)]
            if valid.size == 0:
                continue
            ax.hist(valid, bins=12, alpha=0.5, color=color, label=label, edgecolor="white")
        ax.set_title(f"H = {H}")
        ax.set_xlabel(r"$\|z(H)\|$")
        ax.set_ylabel("count")
        if col == 0:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Diagnostic D: cond=true drifts outward, cond=zero stays close to DNS at long horizons?")
    fig.tight_layout()
    fig.savefig(FIGS / "z_norm_histograms.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[diag-D] wrote {FIGS / 'z_norm_histograms.png'}")


if __name__ == "__main__":
    main()
