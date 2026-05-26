"""Session 16, Experiment 3: SHAP intervention validation.

For each (target, encounter) pair that PASSED the bootstrap stability gate,
compare two interventions on the impact-frame omega field:

    (A) SHAP-driven inpaint: zero-mask the top-K SHAP-attribution pixels and
        fill via Gaussian blur (sigma = 3 grid cells) of the surrounding
        field. Per the spec: NOT phase-mean replacement (which would carry
        no-gust baseline geometry into the field and confound the
        attribution).

    (B) Random-K inpaint: same Gaussian-blur fill but on a random K-pixel
        subset (uniform draw, with the same K).

After each intervention, re-encode the modified omega field and re-predict
the target via the trained probe. Compare delta_target between A and B:
strong attribution should mean A's delta is much larger than B's.

Output:
    outputs/session16/exp3/shap_intervention.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.session16.exp3_shap import (  # noqa: E402
    PROBE_TARGETS, N_INTEGRATION_STEPS,
    gather_split_encounters, load_encoder, load_split_data, load_omega_normalized,
    train_probes, integrated_gradients,
)
from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402

OMEGA_MANIFEST = REPO / "outputs" / "data_pipeline" / "v1" / "manifest.json"
OUT = REPO / "outputs" / "session16" / "exp3"


def gaussian_inpaint(omega: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Replace omega values inside `mask` with values from the Gaussian-blurred
    full field. Outside `mask` is preserved exactly.
    """
    blurred = gaussian_filter(omega.astype(np.float32), sigma=sigma)
    out = omega.copy()
    out[mask] = blurred[mask]
    return out


def topk_pixels(attribution: np.ndarray, K: int) -> np.ndarray:
    """Return boolean mask of the top-K |attribution| pixels."""
    abs_attr = np.abs(attribution)
    flat = abs_attr.ravel()
    threshold_idx = np.argpartition(flat, -K)[-K:]
    mask = np.zeros_like(flat, dtype=bool)
    mask[threshold_idx] = True
    return mask.reshape(attribution.shape)


def random_k_mask(K: int, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    total = shape[0] * shape[1]
    idx = rng.choice(total, size=K, replace=False)
    mask = np.zeros(total, dtype=bool)
    mask[idx] = True
    return mask.reshape(shape)


@torch.no_grad()
def predict_target(
    enc, probe, omega_norm: np.ndarray, mean: float, std: float, device: torch.device
) -> float:
    o_t = torch.from_numpy(omega_norm.astype(np.float32)).to(device)
    x_in = o_t.view(1, 1, 1, *omega_norm.shape)
    z = enc(x_in)
    pred_norm = probe(z[:, 0, :])
    return float(pred_norm.cpu().item()) * std + mean


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--splits", nargs="+", default=["test_b", "test_c"])
    p.add_argument("--K", type=int, default=400, help="number of top-attribution pixels to intervene")
    p.add_argument("--sigma", type=float, default=3.0, help="Gaussian-blur sigma in grid cells")
    p.add_argument("--n-random-control", type=int, default=5, help="number of random control draws per encounter")
    args = p.parse_args()

    device = require_rtx6000(gpu_index=args.gpu)
    print(f"[exp3-int] device={device}")

    enc = load_encoder(device)
    pipeline = OmegaPipeline.from_manifest(OMEGA_MANIFEST)

    train_data = load_split_data("train")
    probes = train_probes(PROBE_TARGETS, train_data, device)

    shap_blob = np.load(OUT / "shap_attribution.npz", allow_pickle=True)
    boot_blob = np.load(OUT / "shap_bootstrap.npz", allow_pickle=True)

    results: dict = {}
    rng = np.random.default_rng(0)

    t0 = time.time()
    for split in args.splits:
        encs = gather_split_encounters(split)
        case_ids_shap = shap_blob[f"{split}_case_id"]
        encounter_idx_shap = shap_blob[f"{split}_encounter_index"]
        results[split] = {}
        for target_name in PROBE_TARGETS:
            probe, t_mean, t_std = probes[target_name]
            attrs = shap_blob[f"{split}_{target_name}_attr"]  # (n, 192, 96)
            preds_actual = shap_blob[f"{split}_{target_name}_pred"]
            stable_mask = boot_blob[f"{split}_{target_name}_stable"]
            n = len(encs)
            assert attrs.shape[0] == n

            per_enc_records = []
            kept = 0
            for i, e in enumerate(encs):
                if not bool(stable_mask[i]):
                    per_enc_records.append({
                        "case_id": e["case_id"], "encounter_index": int(e["k"]),
                        "dropped_for_instability": True,
                    })
                    continue
                omega_norm, impact = load_omega_normalized(e, pipeline)
                T_actual = omega_norm.shape[0]
                if impact >= T_actual:
                    impact = T_actual - 1
                base_omega = omega_norm[impact].copy()
                attr = attrs[i]
                K = min(args.K, attr.size)

                pred_unmod = predict_target(enc, probe, base_omega, t_mean, t_std, device)

                shap_mask = topk_pixels(attr, K)
                omega_shap = gaussian_inpaint(base_omega, shap_mask, sigma=args.sigma)
                pred_shap = predict_target(enc, probe, omega_shap, t_mean, t_std, device)
                delta_shap = pred_shap - pred_unmod

                deltas_random = []
                for _ in range(args.n_random_control):
                    rmask = random_k_mask(K, attr.shape, rng)
                    omega_rand = gaussian_inpaint(base_omega, rmask, sigma=args.sigma)
                    pred_rand = predict_target(enc, probe, omega_rand, t_mean, t_std, device)
                    deltas_random.append(pred_rand - pred_unmod)
                deltas_random = np.array(deltas_random)

                per_enc_records.append({
                    "case_id": e["case_id"], "encounter_index": int(e["k"]),
                    "dropped_for_instability": False,
                    "pred_unmodified": float(pred_unmod),
                    "pred_shap_intervened": float(pred_shap),
                    "delta_shap": float(delta_shap),
                    "delta_random_mean": float(deltas_random.mean()),
                    "delta_random_std": float(deltas_random.std(ddof=1)) if len(deltas_random) > 1 else 0.0,
                    "abs_ratio_shap_over_random": float(abs(delta_shap) / (abs(deltas_random.mean()) + 1e-6)),
                    "K_pixels_changed": int(K),
                })
                kept += 1
            results[split][target_name] = {
                "n_total": n,
                "n_kept_after_stability_filter": kept,
                "n_dropped": int((~stable_mask).sum()),
                "per_encounter": per_enc_records,
            }
            kept_recs = [r for r in per_enc_records if not r.get("dropped_for_instability", False)]
            if kept_recs:
                shap_d = np.array([r["delta_shap"] for r in kept_recs])
                rand_d = np.array([r["delta_random_mean"] for r in kept_recs])
                results[split][target_name]["summary"] = {
                    "mean_abs_delta_shap": float(np.mean(np.abs(shap_d))),
                    "mean_abs_delta_random": float(np.mean(np.abs(rand_d))),
                    "median_abs_ratio_shap_over_random": float(
                        np.median(np.abs(shap_d) / (np.abs(rand_d) + 1e-6))
                    ),
                    "n_shap_dominates_random": int(np.sum(np.abs(shap_d) > np.abs(rand_d))),
                    "n_kept": len(kept_recs),
                }
                s = results[split][target_name]["summary"]
                print(
                    f"[exp3-int] {split:<8s} {target_name:<20s} "
                    f"|delta_shap| = {s['mean_abs_delta_shap']:.4f}  "
                    f"|delta_random| = {s['mean_abs_delta_random']:.4f}  "
                    f"ratio = {s['median_abs_ratio_shap_over_random']:.2f}x  "
                    f"shap > random in {s['n_shap_dominates_random']}/{s['n_kept']}"
                )

    save = OUT / "shap_intervention.json"
    save.write_text(json.dumps({
        "K_pixels": args.K,
        "sigma_grid_cells": args.sigma,
        "n_random_control_draws_per_encounter": args.n_random_control,
        "inpaint_method": "Gaussian blur of the FULL field (sigma in grid cells), with replacement only at masked pixels.",
        "results": results,
    }, indent=2))
    print(f"\n[exp3-int] wrote {save.relative_to(REPO)}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
