"""Session 16, Experiment 3 extension: pixel-level gradient-SHAP for Y axis.

Uses a regularized MLP probe (weight_decay 1e-2, early stopping on test_a)
trained from impact-frame z to predict Y. Since the MLP is end-to-end
differentiable, gradient-SHAP through encoder+probe is straightforward.

Bootstrap stability and intervention validation mirror exp3_shap.py /
exp3_bootstrap.py / exp3_intervention.py.

Output:
    outputs/session16/exp3/shap_Y_attribution.npz
    outputs/session16/exp3/shap_Y_bootstrap.json
    outputs/session16/exp3/shap_Y_intervention.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from scripts.session16.exp3_shap import (  # noqa: E402
    ENCODER_CKPT, OMEGA_MANIFEST, DEFAULT_IMPACT_FRAME,
    gather_split_encounters, load_encoder, load_omega_normalized,
    compute_phase_matched_baseline, integrated_gradients,
)
from src.utils.device import require_rtx6000  # noqa: E402

TARGETS_DIR = REPO / "outputs" / "session16" / "exp2" / "per_frame_targets"
OUT = REPO / "outputs" / "session16" / "exp3"


class MLPProbe(nn.Module):
    def __init__(self, in_dim=64, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_y_probe(train_imp: dict, test_a_imp: dict, device, *,
                  weight_decay=1e-2, lr=3e-4, batch=64, max_iters=4000, patience=400) -> tuple[MLPProbe, float, float]:
    """IMPACT-FRAME-only Y probe (matches D118-bis setup).

    Per-frame training failed for Y (Exp 2 redo confirmed test_b R^2 = -0.21).
    The IMPACT-frame z encodes Y nonlinearly (D118-bis: KRR Y test_b 0.73).
    We train on impact-frame z only so the probe targets that encoding.
    """
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    z_train = train_imp["z"].astype(np.float32)
    y_train = train_imp["Y"].astype(np.float32)
    z_val = test_a_imp["z"].astype(np.float32)
    y_val = test_a_imp["Y"].astype(np.float32)
    d = z_train.shape[1]
    mean = float(y_train.mean())
    std = float(y_train.std()) or 1.0
    y_norm = (y_train - mean) / std
    yv_norm = (y_val - mean) / std

    probe = MLPProbe(in_dim=d).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    z_val_t = torch.from_numpy(z_val).to(device)
    y_val_t = torch.from_numpy(yv_norm).to(device)

    best_val = float("inf")
    best_state = None
    best_iter = 0
    for it in range(max_iters):
        idx = rng.choice(z_train.shape[0], size=min(batch, z_train.shape[0]), replace=False)
        z_batch = z_train[idx]
        y_batch = y_norm[idx]
        probe.train()
        z_t = torch.from_numpy(z_batch).to(device)
        y_t = torch.from_numpy(y_batch).to(device)
        opt.zero_grad()
        loss = ((probe(z_t) - y_t) ** 2).mean()
        loss.backward()
        opt.step()
        if (it + 1) % 50 == 0:
            probe.eval()
            with torch.no_grad():
                vl = ((probe(z_val_t) - y_val_t) ** 2).mean().item()
            if vl < best_val:
                best_val = vl
                best_state = {k: v.clone() for k, v in probe.state_dict().items()}
                best_iter = it + 1
            elif it + 1 - best_iter > patience:
                break
    if best_state is not None:
        probe.load_state_dict(best_state)
    probe.eval()
    for p in probe.parameters():
        p.requires_grad_(False)
    print(f"[exp3-Y] impact-frame MLP trained: best_iter={best_iter} mean={mean:.4f} std={std:.4f}")
    return probe, mean, std


IMPACT_LATENTS_DIR = REPO / "outputs" / "session14" / "latents" / "S12_E_d64"


def load_impact_split(name: str) -> dict:
    d = np.load(IMPACT_LATENTS_DIR / f"{name}.npz", allow_pickle=True)
    return {
        "z": d["z"].astype(np.float64),
        "Y": d["Y"].astype(np.float64),
    }


def load_target_split(name: str) -> dict:
    d = np.load(TARGETS_DIR / f"{name}.npz", allow_pickle=True)
    return {
        "z_full": d["z_full"].astype(np.float32),
        "Y": d["Y"].astype(np.float32),
    }


def pairwise_pearson(maps: np.ndarray) -> np.ndarray:
    n = maps.shape[0]
    flat = maps.reshape(n, -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(flat, axis=1, keepdims=True)
    norm = np.where(norm > 0, norm, 1.0)
    flat_n = flat / norm
    return flat_n @ flat_n.T


def gaussian_inpaint(omega: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    blurred = gaussian_filter(omega.astype(np.float32), sigma=sigma)
    out = omega.copy()
    out[mask] = blurred[mask]
    return out


def topk_pixels(attr: np.ndarray, K: int) -> np.ndarray:
    abs_attr = np.abs(attr)
    flat = abs_attr.ravel()
    idx = np.argpartition(flat, -K)[-K:]
    mask = np.zeros_like(flat, dtype=bool)
    mask[idx] = True
    return mask.reshape(attr.shape)


@torch.no_grad()
def predict_y(enc, probe, omega_norm: np.ndarray, mean: float, std: float, device) -> float:
    o_t = torch.from_numpy(omega_norm.astype(np.float32)).to(device)
    x_in = o_t.view(1, 1, 1, *omega_norm.shape)
    z = enc(x_in)
    pred = probe(z[:, 0, :])
    return float(pred.cpu().item()) * std + mean


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--splits", nargs="+", default=["test_b", "test_c"])
    p.add_argument("--n-steps", type=int, default=32)
    p.add_argument("--K", type=int, default=400)
    p.add_argument("--sigma", type=float, default=3.0)
    p.add_argument("--n-random-control", type=int, default=5)
    p.add_argument("--stability-threshold", type=float, default=0.7)
    args = p.parse_args()

    device = require_rtx6000(gpu_index=args.gpu)
    print(f"[exp3-Y] device={device}")

    enc = load_encoder(device)
    pipeline = OmegaPipeline.from_manifest(OMEGA_MANIFEST)
    # IMPACT-FRAME-only probe training (per D118-bis success regime)
    train_imp = load_impact_split("train")
    test_a_imp = load_impact_split("test_a")
    probe, mean, std = train_y_probe(train_imp, test_a_imp, device)

    # Eval impact-frame Y probe on test_b / test_c impact frames
    for sp_name in ("test_b", "test_c"):
        d = load_impact_split(sp_name)
        with torch.no_grad():
            yp = probe(torch.from_numpy(d["z"].astype(np.float32)).to(device)).cpu().numpy() * std + mean
        from sklearn.metrics import r2_score
        r2 = float(r2_score(d["Y"], yp))
        print(f"[exp3-Y] impact-frame MLP Y R^2 on {sp_name}: {r2:+.3f}")

    # Baseline phase-matched
    baseline_encs = gather_split_encounters("baseline_pool")
    baseline = compute_phase_matched_baseline(baseline_encs, pipeline, device)
    baseline_individual_omegas = baseline["individual"]
    print(f"[exp3-Y] baseline pool: {len(baseline_encs)} encounters")

    out_npz: dict = {
        "baseline_mean_per_frame": baseline["mean_per_frame"],
        "n_integration_steps": np.array(args.n_steps),
    }
    bootstrap_records: dict = {}
    intervention_records: dict = {}
    t0 = time.time()
    for split in args.splits:
        encs = gather_split_encounters(split)
        print(f"\n[exp3-Y] split={split}: {len(encs)} encounters")
        n = len(encs)
        attribution = np.zeros((n, 192, 96), dtype=np.float32)
        pred_actual = np.zeros(n, dtype=np.float32)
        pred_baseline = np.zeros(n, dtype=np.float32)
        bootstrap_pairwise = np.zeros((n, len(baseline_encs), len(baseline_encs)), dtype=np.float32)
        bootstrap_mean = np.zeros(n, dtype=np.float32)
        bootstrap_stable = np.zeros(n, dtype=bool)
        intervention_per_enc = []
        case_ids = []
        encounter_indices = []
        Gs, Ds, Ys, impact_frames = [], [], [], []
        rng = np.random.default_rng(0)

        for i, e in enumerate(encs):
            omega_norm, impact = load_omega_normalized(e, pipeline)
            T_actual = omega_norm.shape[0]
            if impact >= T_actual:
                impact = T_actual - 1
            omega_input = torch.from_numpy(omega_norm[impact]).to(device)
            base_frame_idx = min(impact, baseline["mean_per_frame"].shape[0] - 1)
            omega_baseline = torch.from_numpy(baseline["mean_per_frame"][base_frame_idx]).to(device)

            # Main SHAP attribution (phase-matched mean baseline)
            attr = integrated_gradients(enc, probe, omega_baseline, omega_input,
                                        n_steps=args.n_steps, device=device)
            attribution[i] = attr.cpu().numpy()

            pred_actual[i] = predict_y(enc, probe, omega_norm[impact], mean, std, device)
            with torch.no_grad():
                x_b = omega_baseline.view(1, 1, 1, 192, 96)
                z_b = enc(x_b)
                pred_baseline[i] = float(probe(z_b[:, 0, :]).cpu().item()) * std + mean

            # Bootstrap (4 individual baselines)
            boot_maps = np.zeros((len(baseline_encs), 192, 96), dtype=np.float32)
            for b in range(len(baseline_encs)):
                base_frame_idx_b = min(impact, baseline_individual_omegas[b].shape[0] - 1)
                base_t = torch.from_numpy(baseline_individual_omegas[b][base_frame_idx_b]).to(device)
                attr_b = integrated_gradients(enc, probe, base_t, omega_input,
                                              n_steps=args.n_steps, device=device)
                boot_maps[b] = attr_b.cpu().numpy()
            R = pairwise_pearson(boot_maps)
            bootstrap_pairwise[i] = R
            off = R[np.triu_indices(len(baseline_encs), k=1)]
            bootstrap_mean[i] = float(off.mean())
            bootstrap_stable[i] = bool(bootstrap_mean[i] >= args.stability_threshold)

            # Intervention (only for stable encounters)
            if bootstrap_stable[i]:
                K = min(args.K, attr.shape[0] * attr.shape[1])
                shap_mask = topk_pixels(attribution[i], K)
                omega_shap = gaussian_inpaint(omega_norm[impact], shap_mask, sigma=args.sigma)
                pred_shap = predict_y(enc, probe, omega_shap, mean, std, device)
                deltas_rand = []
                for _ in range(args.n_random_control):
                    flat = np.zeros(attr.shape[0] * attr.shape[1], dtype=bool)
                    idx = rng.choice(flat.size, size=K, replace=False)
                    flat[idx] = True
                    rmask = flat.reshape(attr.shape)
                    omega_rand = gaussian_inpaint(omega_norm[impact], rmask, sigma=args.sigma)
                    pred_rand = predict_y(enc, probe, omega_rand, mean, std, device)
                    deltas_rand.append(pred_rand - pred_actual[i])
                deltas_rand = np.array(deltas_rand)
                delta_shap = pred_shap - pred_actual[i]
                intervention_per_enc.append({
                    "case_id": e["case_id"], "encounter_index": int(e["k"]),
                    "stable": True,
                    "pred_unmodified": float(pred_actual[i]),
                    "pred_shap_intervened": float(pred_shap),
                    "delta_shap": float(delta_shap),
                    "delta_random_mean": float(deltas_rand.mean()),
                    "abs_ratio": float(abs(delta_shap) / (abs(deltas_rand.mean()) + 1e-6)),
                })
            else:
                intervention_per_enc.append({
                    "case_id": e["case_id"], "encounter_index": int(e["k"]),
                    "stable": False,
                })

            case_ids.append(e["case_id"])
            encounter_indices.append(e["k"])
            Gs.append(e["G"]); Ds.append(e["D"]); Ys.append(e["Y"])
            impact_frames.append(int(impact))
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{len(encs)}  ({time.time()-t0:.1f}s)")

        out_npz[f"{split}_Y_attr"] = attribution
        out_npz[f"{split}_Y_pred"] = pred_actual
        out_npz[f"{split}_Y_pred_baseline"] = pred_baseline
        out_npz[f"{split}_Y_bootstrap_pairwise_r"] = bootstrap_pairwise
        out_npz[f"{split}_Y_bootstrap_mean_r"] = bootstrap_mean
        out_npz[f"{split}_Y_bootstrap_stable"] = bootstrap_stable
        out_npz[f"{split}_case_id"] = np.array(case_ids, dtype=object)
        out_npz[f"{split}_encounter_index"] = np.array(encounter_indices, dtype=np.int32)
        out_npz[f"{split}_G"] = np.array(Gs, dtype=np.float32)
        out_npz[f"{split}_D"] = np.array(Ds, dtype=np.float32)
        out_npz[f"{split}_Y"] = np.array(Ys, dtype=np.float32)
        out_npz[f"{split}_impact_frame"] = np.array(impact_frames, dtype=np.int32)

        bootstrap_records[split] = {
            "n_total": int(n),
            "n_stable": int(bootstrap_stable.sum()),
            "stability_rate": float(bootstrap_stable.mean()),
            "mean_pairwise_r_distribution": {
                "min": float(bootstrap_mean.min()),
                "median": float(np.median(bootstrap_mean)),
                "max": float(bootstrap_mean.max()),
            },
        }
        intervention_records[split] = {
            "per_encounter": intervention_per_enc,
        }
        kept = [r for r in intervention_per_enc if r.get("stable", False)]
        if kept:
            shap_d = np.array([r["delta_shap"] for r in kept])
            rand_d = np.array([r["delta_random_mean"] for r in kept])
            intervention_records[split]["summary"] = {
                "n_kept": len(kept),
                "mean_abs_delta_shap": float(np.mean(np.abs(shap_d))),
                "mean_abs_delta_random": float(np.mean(np.abs(rand_d))),
                "median_abs_ratio_shap_over_random": float(np.median(np.abs(shap_d) / (np.abs(rand_d) + 1e-6))),
                "n_shap_dominates_random": int(np.sum(np.abs(shap_d) > np.abs(rand_d))),
            }
            print(f"\n[exp3-Y] {split}: stable={int(bootstrap_stable.sum())}/{n} "
                  f"({100*bootstrap_stable.mean():.0f}%) "
                  f"|delta_shap|={intervention_records[split]['summary']['mean_abs_delta_shap']:.4f} "
                  f"|delta_rand|={intervention_records[split]['summary']['mean_abs_delta_random']:.4f} "
                  f"ratio={intervention_records[split]['summary']['median_abs_ratio_shap_over_random']:.2f}x "
                  f"shap>rand {intervention_records[split]['summary']['n_shap_dominates_random']}/{intervention_records[split]['summary']['n_kept']}")

    save_npz = OUT / "shap_Y_attribution.npz"
    np.savez_compressed(save_npz, **out_npz)
    print(f"\n[exp3-Y] wrote {save_npz.relative_to(REPO)}")

    save_boot = OUT / "shap_Y_bootstrap.json"
    save_boot.write_text(json.dumps({
        "stability_threshold": args.stability_threshold,
        "n_integration_steps": args.n_steps,
        "drops_per_split": bootstrap_records,
    }, indent=2))
    save_intv = OUT / "shap_Y_intervention.json"
    save_intv.write_text(json.dumps({
        "K_pixels": args.K, "sigma_grid_cells": args.sigma,
        "results": intervention_records,
    }, indent=2))
    print(f"[exp3-Y] wrote {save_boot.relative_to(REPO)} + {save_intv.relative_to(REPO)}")
    print(f"[exp3-Y] total wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
