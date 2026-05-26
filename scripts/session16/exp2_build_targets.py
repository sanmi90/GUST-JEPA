"""Session 16, Experiment 2: build per-frame probe targets for every cached
encounter in train/test_a/test_b/test_c.

Targets per frame (each is a single scalar per (encounter, frame)):
    G, D, Y                          : conditioning (constant in t, broadcast)
    C_L, C_D                         : lift / drag coefficients
    peak_pos_omega, peak_neg_omega   : max / min vorticity in the field
    centroid_x, centroid_y           : |omega|-weighted centroid (chord coords)
    circulation_pos, circulation_neg : signed area integral
    wake_length, wake_thickness      : downstream extent (|omega| > 1)
    wake_enstrophy                   : (omega^2) integrated over wake region

Wake region: x in [0.5, 4.0], y in [-1.0, 1.0]. Threshold for length /
thickness: |omega| > 1.0 raw units.

Output:
    outputs/session16/exp2/per_frame_targets/{split}.npz
        per-target arrays of shape (n_encounters, 120)
        z_full (n_encounters, 120, 64) -- mirrored from latents NPZ for
            single-file probe training
        case_id, encounter_index per encounter
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


SPLIT_MANIFEST = REPO / "configs" / "splits" / "split_v1.json"
LATENTS_DIR = REPO / "outputs" / "session14" / "latents" / "S12_E_d64"
OUT = REPO / "outputs" / "session16" / "exp2" / "per_frame_targets"
OUT.mkdir(parents=True, exist_ok=True)
PARTITION = "v1"

CACHE_ROOT = Path(
    os.environ.get(
        "VORTEX_JEPA_CACHE",
        str(Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
            / "data" / "processed" / "vortex-jepa"),
    )
)

X_EXTENT = (-1.5, 4.5)
Y_EXTENT = (-1.5, 1.5)
NX, NY = 192, 96
DX = (X_EXTENT[1] - X_EXTENT[0]) / NX
DY = (Y_EXTENT[1] - Y_EXTENT[0]) / NY

WAKE_X = (0.5, 4.0)
WAKE_Y = (-1.0, 1.0)
WAKE_THRESHOLD = 1.0


def wake_mask() -> np.ndarray:
    x = np.linspace(X_EXTENT[0] + DX / 2, X_EXTENT[1] - DX / 2, NX)
    y = np.linspace(Y_EXTENT[0] + DY / 2, Y_EXTENT[1] - DY / 2, NY)
    in_x = (x >= WAKE_X[0]) & (x <= WAKE_X[1])
    in_y = (y >= WAKE_Y[0]) & (y <= WAKE_Y[1])
    return (in_x[:, None] & in_y[None, :]).astype(bool)


def per_frame_descriptors(omega_raw: np.ndarray) -> dict:
    """omega_raw: (T, 192, 96). Returns dict of (T,) arrays."""
    T = omega_raw.shape[0]
    x = np.linspace(X_EXTENT[0] + DX / 2, X_EXTENT[1] - DX / 2, NX)
    y = np.linspace(Y_EXTENT[0] + DY / 2, Y_EXTENT[1] - DY / 2, NY)
    mask = wake_mask()
    abs_o = np.abs(omega_raw)
    pos_o = np.clip(omega_raw, 0.0, None)
    neg_o = np.clip(omega_raw, None, 0.0)

    out: dict = {
        "peak_pos_omega": omega_raw.reshape(T, -1).max(axis=1).astype(np.float32),
        "peak_neg_omega": omega_raw.reshape(T, -1).min(axis=1).astype(np.float32),
        "circulation_pos": (pos_o.sum(axis=(1, 2)) * DX * DY).astype(np.float32),
        "circulation_neg": (neg_o.sum(axis=(1, 2)) * DX * DY).astype(np.float32),
        "wake_enstrophy": ((abs_o * mask[None]) ** 2).sum(axis=(1, 2)).astype(np.float32) * DX * DY,
    }

    total_w = abs_o.sum(axis=(1, 2))
    cx = np.where(
        total_w > 0,
        (x[None, :, None] * abs_o).sum(axis=(1, 2)) / np.maximum(total_w, 1e-12),
        np.nan,
    ).astype(np.float32)
    cy = np.where(
        total_w > 0,
        (y[None, None, :] * abs_o).sum(axis=(1, 2)) / np.maximum(total_w, 1e-12),
        np.nan,
    ).astype(np.float32)
    out["centroid_x"] = cx
    out["centroid_y"] = cy

    wake_field = abs_o * mask[None]
    active = wake_field > WAKE_THRESHOLD
    wlen = np.zeros(T, dtype=np.float32)
    wthk = np.zeros(T, dtype=np.float32)
    for t in range(T):
        a = active[t]
        if a.any():
            xs_active = np.where(a.any(axis=1))[0]
            ys_active = np.where(a.any(axis=0))[0]
            wlen[t] = (xs_active.max() - xs_active.min()) * DX
            wthk[t] = (ys_active.max() - ys_active.min()) * DY
    out["wake_length"] = wlen
    out["wake_thickness"] = wthk
    return out


def gather_encounters_for_split(split: str) -> list[dict]:
    with open(SPLIT_MANIFEST) as f:
        manifest = json.load(f)
    out: list[dict] = []
    for cid, case in manifest["cases"].items():
        if split == "train" and case["split"] == "train":
            ks = case["train_encounter_indices"]
        elif split == "test_a" and case["split"] == "train":
            ks = case["test_a_encounter_indices"]
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
                "case_id": cid,
                "k": int(k),
                "G": float(case.get("G", 0.0)),
                "D": float(case.get("D", 0.0)),
                "Y": float(case.get("Y", 0.0)),
                "path": path,
            })
    return out


def main() -> None:
    for split in ("train", "test_a", "test_b", "test_c"):
        encs = gather_encounters_for_split(split)
        if not encs:
            continue
        latents_path = LATENTS_DIR / f"{split}.npz"
        latents = np.load(latents_path, allow_pickle=True)
        # Build a lookup case_id+encounter_index -> latent row index
        lat_case = latents["case_id"]
        lat_idx = latents["encounter_index"]
        lookup = {(str(lat_case[i]), int(lat_idx[i])): i for i in range(len(lat_case))}

        n = len(encs)
        T = 120

        targets = {
            "G": np.zeros((n, T), dtype=np.float32),
            "D": np.zeros((n, T), dtype=np.float32),
            "Y": np.zeros((n, T), dtype=np.float32),
            "C_L": np.zeros((n, T), dtype=np.float32),
            "C_D": np.zeros((n, T), dtype=np.float32),
            "peak_pos_omega": np.zeros((n, T), dtype=np.float32),
            "peak_neg_omega": np.zeros((n, T), dtype=np.float32),
            "centroid_x": np.zeros((n, T), dtype=np.float32),
            "centroid_y": np.zeros((n, T), dtype=np.float32),
            "circulation_pos": np.zeros((n, T), dtype=np.float32),
            "circulation_neg": np.zeros((n, T), dtype=np.float32),
            "wake_length": np.zeros((n, T), dtype=np.float32),
            "wake_thickness": np.zeros((n, T), dtype=np.float32),
            "wake_enstrophy": np.zeros((n, T), dtype=np.float32),
        }
        z_full = np.zeros((n, T, 64), dtype=np.float32)
        case_ids = np.array([e["case_id"] for e in encs], dtype=object)
        encounter_indices = np.array([e["k"] for e in encs], dtype=np.int32)
        impact_frames = np.zeros(n, dtype=np.int32)

        t0 = time.time()
        for i, rec in enumerate(encs):
            row = lookup.get((rec["case_id"], rec["k"]))
            if row is None:
                continue
            z_full[i] = latents["z_full"][row]
            impact_frames[i] = int(latents["impact_frame"][row])

            with h5py.File(rec["path"], "r") as f:
                omega_raw = np.asarray(f["omega_z"], dtype=np.float32)
                C_L = np.asarray(f["C_L"], dtype=np.float32)
                C_D = np.asarray(f["C_D"], dtype=np.float32)
            T_actual = omega_raw.shape[0]
            T_use = min(T, T_actual)
            targets["G"][i, :T_use] = rec["G"]
            targets["D"][i, :T_use] = rec["D"]
            targets["Y"][i, :T_use] = rec["Y"]
            targets["C_L"][i, :T_use] = C_L[:T_use]
            targets["C_D"][i, :T_use] = C_D[:T_use]
            desc = per_frame_descriptors(omega_raw[:T_use])
            for key in (
                "peak_pos_omega", "peak_neg_omega",
                "centroid_x", "centroid_y",
                "circulation_pos", "circulation_neg",
                "wake_length", "wake_thickness", "wake_enstrophy",
            ):
                targets[key][i, :T_use] = desc[key]
            if (i + 1) % 20 == 0:
                print(f"[exp2-targets] {split}: {i+1}/{n} done ({time.time()-t0:.1f}s)")

        out_path = OUT / f"{split}.npz"
        np.savez_compressed(
            out_path,
            z_full=z_full,
            case_id=case_ids,
            encounter_index=encounter_indices,
            impact_frame=impact_frames,
            **targets,
        )
        print(f"[exp2-targets] wrote {out_path.relative_to(REPO)} ({n} encounters)")


if __name__ == "__main__":
    main()
