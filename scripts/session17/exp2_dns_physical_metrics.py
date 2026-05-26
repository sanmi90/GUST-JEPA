"""Session 17, Experiment 2 helper: compute DNS physical metrics from cached omega.

For each encounter in {train, test_b, test_c} (split_v1), compute per-frame:
  - C_L (from cache /C_L)
  - C_D (from cache /C_D)
  - I_y = integral x * omega dA          (vorticity impulse, y-component)
  - I_x = -integral y * omega dA         (vorticity impulse, x-component)
  - wake_enstrophy = sum(omega^2) over wake region [x in [0.5, 4], |y| < 1]
  - circulation_pos / circulation_neg over wake (threshold |omega| > 1)
  - radial_spectrum_l2_norm at each frame (mean across 16 bins)

These are saved as a single big NPZ that downstream Exp 2 work consumes.
Computed in raw (un-normalised) omega scale.

Output:
    outputs/session17/exp2/dns_physical_metrics.npz
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402


OMEGA_MANIFEST = REPO / "outputs" / "data_pipeline" / "v1" / "manifest.json"
SPLIT_MANIFEST = REPO / "configs" / "splits" / "split_v1.json"
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

DX = 6.0 / 192
DY = 3.0 / 96
X_GRID = np.linspace(-1.5, 4.5, 192).astype(np.float32)
Y_GRID = np.linspace(-1.5, 1.5, 96).astype(np.float32)
WAKE_X_MIN, WAKE_X_MAX = 0.5, 4.0
WAKE_Y_MAX = 1.0


def gather_split_encounters(split: str) -> list[dict]:
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
            out.append({
                "case_id": cid, "k": int(k), "path": path,
                "G": float(case.get("G", 0.0)),
                "D": float(case.get("D", 0.0)),
                "Y": float(case.get("Y", 0.0)),
            })
    return out


def compute_metrics_for_omega(omega_raw: np.ndarray) -> dict:
    """omega_raw: (T, 192, 96). Returns per-frame metric arrays of shape (T,)."""
    T = omega_raw.shape[0]
    xx = X_GRID[:, None]
    yy = Y_GRID[None, :]
    wake_mask = (xx >= WAKE_X_MIN) & (xx <= WAKE_X_MAX) & (np.abs(yy) <= WAKE_Y_MAX)
    # vectorize across T
    omega_wake = omega_raw * wake_mask[None, :, :]  # (T, 192, 96) masked
    I_x = -np.einsum("tij,j->t", omega_raw, Y_GRID) * DX * DY
    I_y = np.einsum("tij,i->t", omega_raw, X_GRID) * DX * DY
    wake_enstrophy = (omega_wake**2).sum(axis=(1, 2)) * DX * DY
    pos_mask = omega_wake > 1.0
    neg_mask = omega_wake < -1.0
    circulation_pos = (omega_wake * pos_mask).sum(axis=(1, 2)) * DX * DY
    circulation_neg = (omega_wake * neg_mask).sum(axis=(1, 2)) * DX * DY
    return {
        "I_x": I_x.astype(np.float64),
        "I_y": I_y.astype(np.float64),
        "wake_enstrophy": wake_enstrophy.astype(np.float64),
        "circulation_pos": circulation_pos.astype(np.float64),
        "circulation_neg": circulation_neg.astype(np.float64),
    }


def main() -> None:
    pipeline = OmegaPipeline.from_manifest(OMEGA_MANIFEST)

    all_per_split = {}
    for split in ("train", "test_a", "test_b", "test_c"):
        encs = gather_split_encounters(split)
        print(f"[exp2-dns] {split}: {len(encs)} encounters")
        per = {
            "case_id": [], "encounter_index": [], "impact_frame": [],
            "G": [], "D": [], "Y": [],
            "C_L": [], "C_D": [],
            "I_y": [], "I_x": [],
            "wake_enstrophy": [], "circulation_pos": [], "circulation_neg": [],
        }
        t0 = time.time()
        for i, e in enumerate(encs):
            with h5py.File(e["path"], "r") as f:
                omega_raw = np.asarray(f["omega_z"], dtype=np.float32)
                impact = int(f.attrs.get("impact_frame_estimate", DEFAULT_IMPACT_FRAME))
                CL = np.asarray(f["C_L"], dtype=np.float64)
                CD = np.asarray(f["C_D"], dtype=np.float64)
            # Apply the same omega preprocessing as the pipeline (mask + clip) so the
            # I_y / enstrophy live on the same field the encoder sees. Then UN-normalize
            # back to raw scale for the integration (linear).
            omega_clean = pipeline.preprocess_raw(omega_raw, e["case_id"], e["k"])
            m = compute_metrics_for_omega(omega_clean.astype(np.float32))
            per["case_id"].append(e["case_id"])
            per["encounter_index"].append(e["k"])
            per["impact_frame"].append(impact)
            per["G"].append(e["G"])
            per["D"].append(e["D"])
            per["Y"].append(e["Y"])
            per["C_L"].append(CL)
            per["C_D"].append(CD)
            for k in ("I_y", "I_x", "wake_enstrophy", "circulation_pos", "circulation_neg"):
                per[k].append(m[k])
            if (i + 1) % 50 == 0 or i == len(encs) - 1:
                print(f"[exp2-dns] {split} {i+1}/{len(encs)}  ({(time.time()-t0)/(i+1):.2f}s/enc)")
        all_per_split[split] = per
        print(f"[exp2-dns] {split} done in {time.time()-t0:.1f}s")

    # Save NPZ.
    save = {}
    for split, per in all_per_split.items():
        save[f"{split}_case_id"] = np.asarray(per["case_id"])
        save[f"{split}_encounter_index"] = np.asarray(per["encounter_index"])
        save[f"{split}_impact_frame"] = np.asarray(per["impact_frame"])
        save[f"{split}_G"] = np.asarray(per["G"], dtype=np.float64)
        save[f"{split}_D"] = np.asarray(per["D"], dtype=np.float64)
        save[f"{split}_Y"] = np.asarray(per["Y"], dtype=np.float64)
        # Stack 120-frame arrays (all encounters are 120 frames per CLAUDE.md)
        for k in ("C_L", "C_D", "I_y", "I_x", "wake_enstrophy",
                  "circulation_pos", "circulation_neg"):
            arrs = per[k]
            # Each is a per-frame array; pad if shorter
            T_max = max(a.shape[0] for a in arrs) if arrs else 0
            stacked = np.full((len(arrs), T_max), np.nan, dtype=np.float64)
            for j, a in enumerate(arrs):
                stacked[j, :a.shape[0]] = a
            save[f"{split}_{k}"] = stacked
    np.savez_compressed(OUT / "dns_physical_metrics.npz", **save)
    print(f"[exp2-dns] wrote {OUT / 'dns_physical_metrics.npz'}")


if __name__ == "__main__":
    main()
