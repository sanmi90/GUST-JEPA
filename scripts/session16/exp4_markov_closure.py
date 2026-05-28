"""Session 16, Experiment 4: Markov closure test on the production predictor.

Question: given z_impact (the encoder latent at the impact frame), is the
post-impact latent trajectory fully determined by the predictor as a function
of (z_impact, elapsed steps), or does the predictor rely on temporal context
beyond z_impact?

Three rollout modes per encounter:

    A. MARKOV-ONLY: at every step the predictor's attention is MASKED so that
       every query position can only attend to position 0 (z_impact) and to
       itself. The attention is the only path by which a query position learns
       anything about its history; cutting all other positions out forces the
       predictor to use only (z_impact, RoPE_position) plus the residual
       value at the current position.

       Masking choice (documented for the paper): mask[i, 0] = 0; mask[i, i]
       = 0; mask[i, j] = -inf for j not in (0, i). j=i (self-attention) is
       kept open so the value-projection signal at the current query position
       can still propagate through the residual. Without it, the attention
       output would be exactly v_0 at every query position and the entire
       sequence would collapse to a constant attention output.

    B. AUTOREGRESSIVE FROM Z_IMPACT (1-frame seed, sliding window): standard
       open-loop rollout starting from a 1-frame context z[impact]. The
       predictor sees its own prior predictions as the context grows out
       to max_seq_len = 32.

    C. FULL-CONTEXT (32-frame seed ending at impact): standard rollout with
       the full pre-impact history of length min(32, impact+1) as the seed.

Per-step latent RMSE z_pred[t] vs z_dns[t] is the headline metric. SSIM after
decoding through the production SL decoder is the visual sanity check.

Verification: the no-gust baseline encounter (G=0, periodic shedding) should
admit Markov-only closure because the dynamics is autonomous and the impact
frame is just a representative phase.

Output:
    outputs/session16/exp4/markov_closure.json
    outputs/session16/exp4/markov_closure_per_encounter.npz
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
OUT = REPO / "outputs" / "session16" / "exp4"
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
    out: list[dict] = []
    for cid, case in manifest["cases"].items():
        if split == "test_a" and case["split"] == "train":
            ks = (case.get("val_encounter_indices") or case["test_a_encounter_indices"])
        elif split == "test_b" and case["split"] == "test_b":
            ks = list(range(int(case["n_encounters_full"])))
        elif split == "test_c" and case["split"] == "test_c":
            ks = list(range(int(case["n_encounters_full"])))
        elif split == "baseline":
            if cid != "Baseline":
                continue
            ks = list(range(int(case["n_encounters_full"])))
        else:
            continue
        for k in ks:
            path = CACHE_ROOT / PARTITION / cid / f"encounter_{int(k):02d}.h5"
            if not path.exists():
                continue
            out.append({
                "case_id": cid,
                "k": int(k),
                "path": path,
                "G": float(case.get("G", 0.0)),
                "D": float(case.get("D", 0.0)),
                "Y": float(case.get("Y", 0.0)),
            })
    return out


def load_encoder_predictor(
    device: torch.device,
) -> tuple[HybridCNNViTEncoder, AutoregressivePredictor]:
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
    enc_state = {
        k.removeprefix("encoder."): v
        for k, v in full_state.items() if k.startswith("encoder.")
    }
    pred_state = {
        k.removeprefix("predictor."): v
        for k, v in full_state.items() if k.startswith("predictor.")
    }
    enc.load_state_dict(enc_state, strict=False)
    pred.load_state_dict(pred_state, strict=False)
    enc.eval().to(device)
    pred.eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    for p in pred.parameters():
        p.requires_grad_(False)
    return enc, pred


def make_markov_attention_forward():
    """Return a forward replacement for CausalSelfAttentionWithRoPE.

    The replacement uses an explicit attention mask that exposes only
    position 0 (and self) to every query position. is_causal is disabled
    because we are passing an explicit mask.
    """

    def markov_forward(self: CausalSelfAttentionWithRoPE, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = apply_rope(q, self.rope_cos[:T], self.rope_sin[:T])
        k = apply_rope(k, self.rope_cos[:T], self.rope_sin[:T])

        # Markov-only mask: each query position can attend to position 0
        # (z_impact) and to itself. All other positions are blocked.
        # Including the diagonal keeps the value-projection at the query
        # alive, which prevents the attention output from collapsing to
        # the constant v_0 across all positions.
        mask = torch.full((T, T), float("-inf"), device=x.device, dtype=q.dtype)
        mask[:, 0] = 0.0
        mask.fill_diagonal_(0.0)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
        )
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.proj(out)

    return markov_forward


@contextmanager
def markov_attention(model: AutoregressivePredictor) -> Iterator[None]:
    """Temporarily replace every CausalSelfAttentionWithRoPE forward with
    the Markov-only mask version.
    """
    new_forward = make_markov_attention_forward()
    originals: list[tuple[CausalSelfAttentionWithRoPE, callable]] = []
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
def encode_full(
    enc: HybridCNNViTEncoder, omega_norm: np.ndarray, device: torch.device
) -> Tensor:
    x = torch.from_numpy(omega_norm).to(device).unsqueeze(0).unsqueeze(2)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        z = enc(x)
    return z.float().squeeze(0)


@torch.no_grad()
def rollout_markov_only(
    pred: AutoregressivePredictor,
    z_impact: Tensor,
    cond: Tensor,
    steps: int,
    device: torch.device,
) -> Tensor:
    """Markov-only rollout starting at z_impact (T_init=1)."""
    max_seq = int(pred.max_seq_len)
    z_full = z_impact.clone().unsqueeze(0) if z_impact.dim() == 2 else z_impact.clone()
    if z_full.dim() != 3:
        raise ValueError(f"z_impact shape unexpected: {tuple(z_full.shape)}")
    with markov_attention(pred):
        for _ in range(steps):
            ctx = z_full[:, -max_seq:, :]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                z_hat = pred(ctx, cond)
            z_full = torch.cat([z_full, z_hat[:, -1:, :].float()], dim=1)
    return z_full


@torch.no_grad()
def rollout_autoregressive(
    pred: AutoregressivePredictor,
    z_seed: Tensor,
    cond: Tensor,
    steps: int,
    device: torch.device,
) -> Tensor:
    max_seq = int(pred.max_seq_len)
    if z_seed.dim() == 2:
        z_seed = z_seed.unsqueeze(0)
    z_full = z_seed.clone()
    for _ in range(steps):
        ctx = z_full[:, -max_seq:, :]
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            z_hat = pred(ctx, cond)
        z_full = torch.cat([z_full, z_hat[:, -1:, :].float()], dim=1)
    return z_full


def per_step_latent_rmse(z_pred: Tensor, z_dns: Tensor) -> np.ndarray:
    """L2 norm per time step. Both are (T, d)."""
    diff = z_pred - z_dns
    return torch.sqrt((diff ** 2).mean(dim=-1)).cpu().numpy()


def evaluate_encounter(
    encounter: dict,
    enc: HybridCNNViTEncoder,
    pred: AutoregressivePredictor,
    pipeline: OmegaPipeline,
    device: torch.device,
) -> dict | None:
    case_id = encounter["case_id"]
    k = int(encounter["k"])
    G, D, Y = encounter["G"], encounter["D"], encounter["Y"]

    with h5py.File(encounter["path"], "r") as f:
        omega_raw = np.asarray(f["omega_z"], dtype=np.float32)
        impact_frame = int(f.attrs.get("impact_frame_estimate", DEFAULT_IMPACT_FRAME))

    omega_clean = pipeline.preprocess_raw(omega_raw, case_id, k)
    omega_norm = pipeline.normalize(omega_clean).astype(np.float32)
    T_full = int(omega_norm.shape[0])
    if impact_frame >= T_full - 1:
        return None

    z_dns = encode_full(enc, omega_norm, device)  # (T, d)
    cond = torch.tensor([[G, D, Y]], dtype=torch.float32, device=device)

    H_max = T_full - impact_frame - 1  # number of post-impact frames to predict

    # Markov-only rollout
    z_impact = z_dns[impact_frame:impact_frame + 1]  # (1, d)
    z_markov = rollout_markov_only(pred, z_impact, cond, steps=H_max, device=device)
    z_markov_post = z_markov.squeeze(0)[1:, :]  # drop the seed, keep H_max predictions
    rmse_markov = per_step_latent_rmse(z_markov_post, z_dns[impact_frame + 1:])

    # Autoregressive rollout from z_impact (1-frame seed, sliding window context)
    z_ar = rollout_autoregressive(pred, z_impact, cond, steps=H_max, device=device)
    z_ar_post = z_ar.squeeze(0)[1:, :]
    rmse_ar = per_step_latent_rmse(z_ar_post, z_dns[impact_frame + 1:])

    # Full-context rollout (32-frame seed ending at the impact frame)
    seed_len = min(32, impact_frame + 1)
    z_seed = z_dns[impact_frame + 1 - seed_len:impact_frame + 1]  # (seed_len, d)
    z_fc = rollout_autoregressive(pred, z_seed, cond, steps=H_max, device=device)
    z_fc_post = z_fc.squeeze(0)[seed_len:, :]  # post-impact predictions
    rmse_fc = per_step_latent_rmse(z_fc_post, z_dns[impact_frame + 1:])

    return {
        "case_id": case_id,
        "encounter_index": k,
        "G": G, "D": D, "Y": Y,
        "impact_frame": impact_frame,
        "n_post_impact": H_max,
        "rmse_markov": rmse_markov.tolist(),
        "rmse_ar_from_impact": rmse_ar.tolist(),
        "rmse_full_context": rmse_fc.tolist(),
    }


def horizon_summary(per_enc: list[dict], horizons: list[int]) -> dict:
    summary: dict = {"horizons": horizons}
    for mode, key in (
        ("markov", "rmse_markov"),
        ("ar_from_impact", "rmse_ar_from_impact"),
        ("full_context", "rmse_full_context"),
    ):
        per_h_mean = []
        per_h_median = []
        per_h_count = []
        for H in horizons:
            vals = []
            for rec in per_enc:
                rmse_arr = rec[key]
                if H <= len(rmse_arr):
                    vals.append(rmse_arr[H - 1])
            per_h_mean.append(float(np.mean(vals)) if vals else float("nan"))
            per_h_median.append(float(np.median(vals)) if vals else float("nan"))
            per_h_count.append(int(len(vals)))
        summary[mode] = {
            "mean_by_horizon": per_h_mean,
            "median_by_horizon": per_h_median,
            "count_by_horizon": per_h_count,
        }
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--splits", nargs="+", default=["baseline", "test_a", "test_b", "test_c"])
    args = p.parse_args()

    device = require_rtx6000(gpu_index=args.gpu)
    print(f"[exp4] device={device} ({torch.cuda.get_device_name(device.index)})")

    enc, pred = load_encoder_predictor(device)
    pipeline = OmegaPipeline.from_manifest(OMEGA_MANIFEST)
    print(f"[exp4] encoder + predictor loaded; predictor max_seq_len={pred.max_seq_len}")

    HORIZONS = [1, 2, 4, 8, 16, 32, 64, 79]
    per_encounter_records: dict[str, list[dict]] = {}
    summaries: dict[str, dict] = {}
    t0 = time.time()
    for split in args.splits:
        encs = gather_split_encounters(split)
        print(f"\n[exp4] split={split}: {len(encs)} encounters")
        records = []
        for i, e in enumerate(encs, 1):
            rec = evaluate_encounter(e, enc, pred, pipeline, device)
            if rec is None:
                continue
            records.append(rec)
            if i % 10 == 0:
                print(f"  ...{i}/{len(encs)} done")
        per_encounter_records[split] = records
        summary = horizon_summary(records, HORIZONS)
        summaries[split] = summary
        print(f"[exp4] {split} horizon summary (mean latent RMSE per horizon H):")
        print(f"  {'H':>4s}  {'markov':>10s}  {'ar_from_imp':>12s}  {'full_ctx':>10s}")
        for h_idx, H in enumerate(HORIZONS):
            print(
                f"  {H:>4d}  "
                f"{summary['markov']['mean_by_horizon'][h_idx]:>10.3f}  "
                f"{summary['ar_from_impact']['mean_by_horizon'][h_idx]:>12.3f}  "
                f"{summary['full_context']['mean_by_horizon'][h_idx]:>10.3f}"
            )

    elapsed = time.time() - t0
    print(f"\n[exp4] total wall time: {elapsed:.1f}s")

    final = {
        "encoder_ckpt": str(ENCODER_CKPT.relative_to(REPO)),
        "horizons": HORIZONS,
        "masking_choice": (
            "MARKOV mode: mask[i, 0] = 0; mask[i, i] = 0; mask[i, j] = -inf for "
            "j not in (0, i). The j=i self-attention is kept open to keep the "
            "value-projection signal at the current query alive; without it, the "
            "attention output would be exactly v_0 at every position and the "
            "sequence would collapse to a constant attention output. Diagonal "
            "keeps each position's own value contribution while attention "
            "information is sourced only from position 0 (z_impact)."
        ),
        "per_split_summary": summaries,
    }
    save_js = OUT / "markov_closure.json"
    save_js.write_text(json.dumps(final, indent=2))
    print(f"[exp4] wrote {save_js.relative_to(REPO)}")

    save_npz_args: dict = {}
    for split, recs in per_encounter_records.items():
        if not recs:
            continue
        case_ids = np.array([r["case_id"] for r in recs], dtype=object)
        encounter_idxs = np.array([r["encounter_index"] for r in recs], dtype=np.int32)
        impact_frames = np.array([r["impact_frame"] for r in recs], dtype=np.int32)
        Gs = np.array([r["G"] for r in recs], dtype=np.float32)
        Ds = np.array([r["D"] for r in recs], dtype=np.float32)
        Ys = np.array([r["Y"] for r in recs], dtype=np.float32)
        # Pad rmse arrays to the maximum length
        T_max = max(len(r["rmse_markov"]) for r in recs)
        def _pad(records: list[dict], key: str) -> np.ndarray:
            arr = np.full((len(records), T_max), np.nan, dtype=np.float32)
            for i, r in enumerate(records):
                vals = r[key]
                arr[i, : len(vals)] = vals
            return arr
        save_npz_args[f"{split}_case_id"] = case_ids
        save_npz_args[f"{split}_encounter_index"] = encounter_idxs
        save_npz_args[f"{split}_impact_frame"] = impact_frames
        save_npz_args[f"{split}_G"] = Gs
        save_npz_args[f"{split}_D"] = Ds
        save_npz_args[f"{split}_Y"] = Ys
        save_npz_args[f"{split}_rmse_markov"] = _pad(recs, "rmse_markov")
        save_npz_args[f"{split}_rmse_ar_from_impact"] = _pad(recs, "rmse_ar_from_impact")
        save_npz_args[f"{split}_rmse_full_context"] = _pad(recs, "rmse_full_context")
    save_npz = OUT / "markov_closure_per_encounter.npz"
    np.savez(save_npz, **save_npz_args)
    print(f"[exp4] wrote {save_npz.relative_to(REPO)}")


if __name__ == "__main__":
    main()
