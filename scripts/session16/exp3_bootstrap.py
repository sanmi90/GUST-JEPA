"""Session 16, Experiment 3: bootstrap diagnostic for SHAP stability.

For each (target, encounter) pair, recompute integrated gradients using each
of the 4 baseline encounters individually (rather than the phase-matched
mean over all 4). Then measure pairwise Pearson correlation across the 4
attribution maps. Mean pairwise r > 0.7 = stable; otherwise flag for drop.

Per session priority 3: any (target, encounter) pair whose bootstrap
convergence fails is dropped from downstream structure extraction.

Output:
    outputs/session16/exp3/shap_bootstrap.npz / .json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Reuse the SHAP plumbing
from scripts.session16.exp3_shap import (  # noqa: E402
    PROBE_TARGETS, N_INTEGRATION_STEPS,
    gather_split_encounters, load_encoder, load_split_data, load_omega_normalized,
    train_probes, integrated_gradients,
)
from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402

OMEGA_MANIFEST = REPO / "outputs" / "data_pipeline" / "v1" / "manifest.json"
OUT = REPO / "outputs" / "session16" / "exp3"


def pairwise_pearson(maps: np.ndarray) -> np.ndarray:
    """maps: (n_bootstrap, H, W). Returns (n_bootstrap, n_bootstrap) Pearson r."""
    n = maps.shape[0]
    flat = maps.reshape(n, -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(flat, axis=1, keepdims=True)
    norm = np.where(norm > 0, norm, 1.0)
    flat_n = flat / norm
    return flat_n @ flat_n.T


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--splits", nargs="+", default=["test_b", "test_c"])
    p.add_argument("--n-steps", type=int, default=N_INTEGRATION_STEPS)
    p.add_argument("--stability-threshold", type=float, default=0.7)
    args = p.parse_args()

    device = require_rtx6000(gpu_index=args.gpu)
    print(f"[exp3-boot] device={device}")

    enc = load_encoder(device)
    pipeline = OmegaPipeline.from_manifest(OMEGA_MANIFEST)

    train_data = load_split_data("train")
    probes = train_probes(PROBE_TARGETS, train_data, device)

    # Get the 4 baseline encounters individually
    baseline_encs = gather_split_encounters("baseline_pool")
    baseline_omegas: list[np.ndarray] = []
    baseline_impacts: list[int] = []
    for e in baseline_encs:
        o, imp = load_omega_normalized(e, pipeline)
        baseline_omegas.append(o)
        baseline_impacts.append(imp)
    n_boot = len(baseline_omegas)
    print(f"[exp3-boot] {n_boot} baseline encounters available as individual backgrounds")

    out_artefacts: dict = {}
    t0 = time.time()
    flag_records: dict[str, list[dict]] = {}

    for split in args.splits:
        encs = gather_split_encounters(split)
        print(f"\n[exp3-boot] split={split}: {len(encs)} encounters")
        flag_records[split] = []
        pairwise_r_per_target = {
            tg: np.zeros((len(encs), n_boot, n_boot), dtype=np.float32)
            for tg in PROBE_TARGETS
        }
        mean_r_per_target = {tg: np.zeros(len(encs), dtype=np.float32) for tg in PROBE_TARGETS}
        stable_per_target = {tg: np.zeros(len(encs), dtype=bool) for tg in PROBE_TARGETS}
        case_ids = []
        encounter_indices = []
        impact_frames = []
        for i, e in enumerate(encs):
            omega_norm, impact = load_omega_normalized(e, pipeline)
            T_actual = omega_norm.shape[0]
            if impact >= T_actual:
                impact = T_actual - 1
            omega_input = torch.from_numpy(omega_norm[impact]).to(device)

            for target_name in PROBE_TARGETS:
                probe, _, _ = probes[target_name]
                maps = np.zeros((n_boot, 192, 96), dtype=np.float32)
                for b in range(n_boot):
                    base_frame_idx = min(impact, baseline_omegas[b].shape[0] - 1)
                    base_t = torch.from_numpy(baseline_omegas[b][base_frame_idx]).to(device)
                    attr = integrated_gradients(
                        enc, probe, base_t, omega_input, n_steps=args.n_steps, device=device
                    )
                    maps[b] = attr.cpu().numpy()
                R = pairwise_pearson(maps)
                pairwise_r_per_target[target_name][i] = R
                off = R[np.triu_indices(n_boot, k=1)]
                mean_r = float(off.mean())
                mean_r_per_target[target_name][i] = mean_r
                stable_per_target[target_name][i] = bool(mean_r >= args.stability_threshold)
                if mean_r < args.stability_threshold:
                    flag_records[split].append({
                        "encounter_index": int(e["k"]),
                        "case_id": e["case_id"],
                        "target": target_name,
                        "mean_pairwise_r": mean_r,
                    })
            case_ids.append(e["case_id"])
            encounter_indices.append(int(e["k"]))
            impact_frames.append(int(impact))
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{len(encs)} ({time.time() - t0:.1f}s)")

        for target_name in PROBE_TARGETS:
            out_artefacts[f"{split}_{target_name}_pairwise_r"] = pairwise_r_per_target[target_name]
            out_artefacts[f"{split}_{target_name}_mean_pairwise_r"] = mean_r_per_target[target_name]
            out_artefacts[f"{split}_{target_name}_stable"] = stable_per_target[target_name]
        out_artefacts[f"{split}_case_id"] = np.array(case_ids, dtype=object)
        out_artefacts[f"{split}_encounter_index"] = np.array(encounter_indices, dtype=np.int32)
        out_artefacts[f"{split}_impact_frame"] = np.array(impact_frames, dtype=np.int32)

        print(f"[exp3-boot] {split} stability rate per target:")
        for tg in PROBE_TARGETS:
            stab = stable_per_target[tg]
            print(
                f"  {tg:<20s} stable: {int(stab.sum())}/{len(encs)} "
                f"({100*stab.mean():.1f}%) mean_r min/median/max="
                f"{mean_r_per_target[tg].min():.3f}/"
                f"{np.median(mean_r_per_target[tg]):.3f}/"
                f"{mean_r_per_target[tg].max():.3f}"
            )

    save_npz = OUT / "shap_bootstrap.npz"
    np.savez_compressed(save_npz, **out_artefacts)
    print(f"\n[exp3-boot] wrote {save_npz.relative_to(REPO)}")

    save_js = OUT / "shap_bootstrap.json"
    save_js.write_text(json.dumps({
        "n_baseline_backgrounds": n_boot,
        "stability_threshold": args.stability_threshold,
        "n_integration_steps": args.n_steps,
        "drops_per_split": flag_records,
        "drops_total_count": {k: len(v) for k, v in flag_records.items()},
    }, indent=2))
    print(f"[exp3-boot] wrote {save_js.relative_to(REPO)}")
    print(f"[exp3-boot] total wall: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
