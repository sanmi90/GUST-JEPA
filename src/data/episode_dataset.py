"""Per-encounter, sub-trajectory dataset for JEPA training.

Reads the partition v1 cache (omega_z, p_wall, C_L, C_D per encounter) and
yields fixed-length sub-trajectories. The start frame is drawn from an
impact-aware mixture (default 70% impact-aware in [8, 40], 30% uniform in
[0, n_frames - L]) so the model sees the gust event in most batches.

Split semantics (consume `split_v{partition}.json`):
    train   -> for each case with split == 'train', enumerate train_encounter_indices
    test_a  -> for each case with split == 'train', enumerate test_a_encounter_indices
    test_b  -> for each case with split == 'test_b',  enumerate all encounters
    test_c  -> for each case with split == 'test_c',  enumerate all encounters

Sample dict (returned from __getitem__):
    omega_z         torch.float32 (L, nx, ny)
    p_wall          torch.float32 (L, n_surface_points)    if return_pressure
    C_L, C_D        torch.float32 (L,)                     if return_forces
    case_id         str
    encounter_index int
    frame_start     int
    G, D, Y         float
    source_group    str
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml


_VALID_SPLITS = ("train", "test_a", "test_b", "test_c")


class EpisodeDataset(torch.utils.data.Dataset):
    """One sample = one randomly-started sub-trajectory of an encounter."""

    def __init__(
        self,
        partition: str = "v1",
        split: str = "train",
        prevent_root: str | Path | None = None,
        cache_root: str | Path | None = None,
        subtraj_len: int = 32,
        impact_aware_fraction: float = 0.7,
        impact_aware_start_range: tuple[int, int] | None = None,
        uniform_start_range: tuple[int, int] | None = None,
        return_pressure: bool = True,
        return_forces: bool = True,
        seed: int | None = None,
    ) -> None:
        if split not in _VALID_SPLITS:
            raise ValueError(f"split must be one of {_VALID_SPLITS}, got {split!r}")

        repo = Path(__file__).resolve().parents[2]
        with open(repo / "configs" / "preprocessing.yaml") as f:
            config = yaml.safe_load(f)
        with open(repo / f"split_{partition}.json") as f:
            split_manifest = json.load(f)

        prevent_root = Path(prevent_root or os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
        if cache_root is None:
            cache_root = os.environ.get("VORTEX_JEPA_CACHE")
        if cache_root is None:
            cache_root = prevent_root / config["cache"]["root_default"]
        cache_root = Path(cache_root)

        n_frames = int(config["encounter"]["frames_per_encounter"])
        if impact_aware_start_range is None:
            impact_aware_start_range = tuple(split_manifest["subtrajectory_sampling"]["impact_aware_start_range"])
        if uniform_start_range is None:
            ur = split_manifest["subtrajectory_sampling"].get("uniform_start_range")
            uniform_start_range = tuple(ur) if ur else (0, n_frames - subtraj_len)

        self.partition = partition
        self.split = split
        self.cache_dir = cache_root / partition
        self.subtraj_len = int(subtraj_len)
        self.return_pressure = bool(return_pressure)
        self.return_forces = bool(return_forces)
        self.impact_aware_fraction = float(impact_aware_fraction)
        self.impact_aware_start_range = (int(impact_aware_start_range[0]), int(impact_aware_start_range[1]))
        self.uniform_start_range = (int(uniform_start_range[0]), int(uniform_start_range[1]))
        self.n_frames = n_frames
        self._seed = seed

        self.samples: list[tuple[str, int]] = []
        for case_id, case in split_manifest["cases"].items():
            if split == "train" and case["split"] == "train":
                for k in case["train_encounter_indices"]:
                    self.samples.append((case_id, int(k)))
            elif split == "test_a" and case["split"] == "train":
                for k in case["test_a_encounter_indices"]:
                    self.samples.append((case_id, int(k)))
            elif split in ("test_b", "test_c") and case["split"] == split:
                for k in range(int(case["n_encounters_full"])):
                    self.samples.append((case_id, k))

        if not self.samples:
            raise RuntimeError(f"No samples found for partition={partition} split={split}")

    def __len__(self) -> int:
        return len(self.samples)

    def _make_rng(self, idx: int) -> np.random.Generator:
        if self._seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self._seed * 100003 + idx)

    def _sample_start(self, rng: np.random.Generator) -> int:
        if rng.random() < self.impact_aware_fraction:
            lo, hi = self.impact_aware_start_range
        else:
            lo, hi = self.uniform_start_range
        return int(rng.integers(lo, hi + 1))

    def __getitem__(self, idx: int) -> dict:
        case_id, k = self.samples[idx]
        rng = self._make_rng(idx)
        start = self._sample_start(rng)
        end = start + self.subtraj_len

        sample: dict = {
            "case_id": case_id,
            "encounter_index": k,
            "frame_start": start,
        }
        enc_path = self.cache_dir / case_id / f"encounter_{k:02d}.h5"
        with h5py.File(enc_path, "r") as g:
            sample["omega_z"] = torch.from_numpy(g["omega_z"][start:end].astype(np.float32))
            sample["G"] = float(g.attrs["G"])
            sample["D"] = float(g.attrs["D"])
            sample["Y"] = float(g.attrs["Y"])
            sample["source_group"] = str(g.attrs["source_group"])
            if self.return_pressure:
                sample["p_wall"] = torch.from_numpy(g["p_wall"][start:end].astype(np.float32))
            if self.return_forces:
                sample["C_L"] = torch.from_numpy(g["C_L"][start:end].astype(np.float32))
                sample["C_D"] = torch.from_numpy(g["C_D"][start:end].astype(np.float32))
        return sample
