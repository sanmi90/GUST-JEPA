"""Per-encounter, sub-trajectory dataset for JEPA training.

Reads the partition v1 cache (omega_z, p_wall, C_L, C_D per encounter) and
yields fixed-length sub-trajectories. The start frame is drawn from a two-branch
mixture: with probability `impact_aware_fraction` (default 0.7) the start is
sampled uniformly from `impact_overlap_start_range` (default [8, 40]; any start
in this range yields a sub-trajectory [start, start + L) whose intersection with
the impact window [25, 55] contains at least 7 frames). Otherwise the start is
uniform over the full episode (`uniform_start_range`, default [0, n_frames - L]).

Split semantics (default split file: ``configs/splits/split_v2.json``; override via
``split_manifest_path``):
    train   -> for each case with split == 'train', enumerate train_encounter_indices
    test_a  -> for each case with split == 'train', enumerate val_encounter_indices
               (falls back to test_a_encounter_indices for v1 manifests; the
               in-code split name is still 'test_a' for backwards compatibility,
               and is treated as the validation holdout used for model selection)
    test_b  -> for each case with split == 'test_b',  enumerate all encounters
    test_c  -> for each case with split == 'test_c',  enumerate all encounters

Sample dict (returned from __getitem__):
    omega_z         torch.float32 (L, nx, ny)
    p_wall          torch.float32 (L, n_surface_points)    if return_pressure
    C_L, C_D        torch.float32 (L,)                     if return_forces
    cl_future       torch.float32 (L, len(cl_future_deltas))  if emit_cl_future
    case_id         str
    encounter_index int
    frame_start     int
    G, D, Y         float
    source_group    str

The cl_future tensor is the lift coefficient at frame
`frame_start + i + delta` for each delta in `cl_future_deltas` and each
i in [0, L). When `frame_start + i + delta` exceeds the encounter's
last valid frame index (`n_frames - 1`), the value is clamped to the
last valid C_L. This is the Session 6 design call recorded in D36:
the post-impact relaxation regime is approximately stationary, so
clamping introduces small bias rather than spurious signal.
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
        impact_overlap_start_range: tuple[int, int] | None = None,
        uniform_start_range: tuple[int, int] | None = None,
        return_pressure: bool = True,
        return_forces: bool = True,
        emit_cl_future: bool = False,
        cl_future_deltas: tuple[int, ...] = (8, 16, 24),
        emit_wake_observable: bool = False,
        wake_observable_type: str = "patch_signed_spectrum",
        wake_observables_root: str | Path | None = None,
        wake_observable_standardize: bool = True,
        omega_pipeline_manifest: str | Path | None = None,
        split_manifest_path: str | Path | None = None,
        seed: int | None = None,
    ) -> None:
        if split not in _VALID_SPLITS:
            raise ValueError(f"split must be one of {_VALID_SPLITS}, got {split!r}")

        repo = Path(__file__).resolve().parents[2]
        with open(repo / "configs" / "preprocessing.yaml") as f:
            config = yaml.safe_load(f)
        if split_manifest_path is None:
            split_manifest_path = repo / "configs" / "splits" / "split_v2.json"
        else:
            p = Path(split_manifest_path)
            if not p.is_absolute():
                p = repo / p
            split_manifest_path = p
        with open(split_manifest_path) as f:
            split_manifest = json.load(f)
        self.split_manifest_path = split_manifest_path

        prevent_root = Path(prevent_root or os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
        if cache_root is None:
            cache_root = os.environ.get("VORTEX_JEPA_CACHE")
        if cache_root is None:
            cache_root = prevent_root / config["cache"]["root_default"]
        cache_root = Path(cache_root)

        n_frames = int(config["encounter"]["frames_per_encounter"])
        max_valid_start = n_frames - int(subtraj_len)
        if max_valid_start < 0:
            raise ValueError(
                f"subtraj_len={subtraj_len} exceeds frames_per_encounter={n_frames}"
            )
        if impact_overlap_start_range is None:
            ior = tuple(split_manifest["subtrajectory_sampling"]["impact_overlap_start_range"])
            impact_overlap_start_range = (ior[0], min(int(ior[1]), max_valid_start))
        if uniform_start_range is None:
            ur = split_manifest["subtrajectory_sampling"].get("uniform_start_range")
            if ur:
                uniform_start_range = (int(ur[0]), min(int(ur[1]), max_valid_start))
            else:
                uniform_start_range = (0, max_valid_start)

        self.partition = partition
        self.split = split
        self.cache_dir = cache_root / partition
        self.subtraj_len = int(subtraj_len)
        self.return_pressure = bool(return_pressure)
        self.return_forces = bool(return_forces)
        self.emit_cl_future = bool(emit_cl_future)
        self.cl_future_deltas = tuple(int(d) for d in cl_future_deltas)
        if any(d < 0 for d in self.cl_future_deltas):
            raise ValueError(
                f"cl_future_deltas must be non-negative ints; got {self.cl_future_deltas}"
            )

        self.emit_wake_observable = bool(emit_wake_observable)
        self.wake_observable_type = str(wake_observable_type)
        self.wake_observable_standardize = bool(wake_observable_standardize)
        self._wake_stats = None  # lazily loaded on first __getitem__
        if self.emit_wake_observable:
            if wake_observables_root is None:
                wake_observables_root = cache_root / partition / "wake_observables"
            self.wake_observables_root = Path(wake_observables_root)
            if not self.wake_observables_root.exists():
                raise FileNotFoundError(
                    f"wake_observables_root not found: {self.wake_observables_root}. "
                    f"Run scripts/session11_precompute_wake_observables.py first."
                )
        else:
            self.wake_observables_root = None
        # Omega pipeline (mask + per-encounter clip + 3-sigma scale). Applying
        # it inside ``__getitem__`` instead of in the training collate keeps
        # the batch dict trivially worker-picklable, so ``num_workers > 0``
        # works (D85 fix; removes the historical num_workers=0 lock).
        self.omega_pipeline_manifest_path: Path | None = None
        if omega_pipeline_manifest is not None:
            p = Path(omega_pipeline_manifest)
            if not p.is_absolute():
                p = repo / p
            self.omega_pipeline_manifest_path = p
        self._omega_pipeline = None  # lazily loaded on first __getitem__

        self.impact_aware_fraction = float(impact_aware_fraction)
        self.impact_overlap_start_range = (int(impact_overlap_start_range[0]), int(impact_overlap_start_range[1]))
        self.uniform_start_range = (int(uniform_start_range[0]), int(uniform_start_range[1]))
        self.n_frames = n_frames
        self._seed = seed

        self.samples: list[tuple[str, int]] = []
        for case_id, case in split_manifest["cases"].items():
            if split == "train" and case["split"] == "train":
                for k in case["train_encounter_indices"]:
                    self.samples.append((case_id, int(k)))
            elif split == "test_a" and case["split"] == "train":
                # v2 manifests rename test_a_encounter_indices -> val_encounter_indices;
                # in-code "test_a" still names the within-train-case validation holdout.
                ks = case.get("val_encounter_indices")
                if ks is None:
                    ks = case["test_a_encounter_indices"]
                for k in ks:
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
            lo, hi = self.impact_overlap_start_range
        else:
            lo, hi = self.uniform_start_range
        return int(rng.integers(lo, hi + 1))

    def _load_omega_pipeline(self):
        """Lazy-load the OmegaPipeline (per worker, so fork is fine)."""
        if self._omega_pipeline is not None:
            return self._omega_pipeline
        if self.omega_pipeline_manifest_path is None:
            return None
        from src.data.omega_pipeline import OmegaPipeline
        self._omega_pipeline = OmegaPipeline.from_manifest(self.omega_pipeline_manifest_path)
        return self._omega_pipeline

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
            omega_arr = g["omega_z"][start:end].astype(np.float32)
            pipe = self._load_omega_pipeline()
            if pipe is not None:
                # Preprocess (mask + per-encounter clip) and normalize entirely
                # inside the worker -- output omega is in 3-sigma normalized
                # space, ready for the encoder directly.
                omega_arr = pipe.preprocess_raw(omega_arr, case_id, int(k))
                omega_t = torch.from_numpy(omega_arr)
                sample["omega_z"] = pipe.normalize(omega_t)
            else:
                sample["omega_z"] = torch.from_numpy(omega_arr)
            sample["G"] = float(g.attrs["G"])
            sample["D"] = float(g.attrs["D"])
            sample["Y"] = float(g.attrs["Y"])
            sample["source_group"] = str(g.attrs["source_group"])
            if self.return_pressure:
                sample["p_wall"] = torch.from_numpy(g["p_wall"][start:end].astype(np.float32))
            if self.return_forces:
                sample["C_L"] = torch.from_numpy(g["C_L"][start:end].astype(np.float32))
                sample["C_D"] = torch.from_numpy(g["C_D"][start:end].astype(np.float32))
            if self.emit_cl_future:
                max_delta = max(self.cl_future_deltas)
                cl_end = min(end + max_delta, self.n_frames)
                cl_full = g["C_L"][start:cl_end].astype(np.float32)
                last_valid = cl_full[-1]
                T = self.subtraj_len
                n_deltas = len(self.cl_future_deltas)
                cl_future = np.empty((T, n_deltas), dtype=np.float32)
                for j, d in enumerate(self.cl_future_deltas):
                    for i in range(T):
                        src = i + d
                        cl_future[i, j] = cl_full[src] if src < cl_full.shape[0] else last_valid
                sample["cl_future"] = torch.from_numpy(cl_future)
        if self.emit_wake_observable:
            sample["wake_target"] = self._load_wake_target(case_id, k, start, end)
        return sample

    def _load_wake_stats(self):
        """Lazily load and cache the train-pool wake observable stats."""
        if self._wake_stats is not None:
            return self._wake_stats
        if self.wake_observables_root is None:
            return None
        from src.data.wake_observables import WakeObservableStats
        stats_path = self.wake_observables_root / "_train_stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"wake observable stats not found at {stats_path}; "
                f"run the precompute script first."
            )
        with open(stats_path) as f:
            payload = json.load(f)
        if self.wake_observable_type not in payload:
            raise KeyError(
                f"wake_observable_type {self.wake_observable_type!r} not in stats "
                f"({list(payload)})"
            )
        self._wake_stats = WakeObservableStats.from_dict(payload[self.wake_observable_type])
        return self._wake_stats

    def _load_wake_target(self, case_id: str, k: int, start: int, end: int) -> torch.Tensor:
        """Read precomputed wake target ``(T, out_dim)`` and optionally standardize."""
        path = self.wake_observables_root / case_id / f"encounter_{k:02d}.h5"
        if not path.exists():
            raise FileNotFoundError(
                f"wake observable cache file missing: {path}. "
                f"Re-run the precompute script."
            )
        with h5py.File(path, "r") as g:
            if self.wake_observable_type not in g:
                raise KeyError(
                    f"wake_observable_type {self.wake_observable_type!r} not in "
                    f"{path}; precompute included {list(g.keys())}"
                )
            arr = g[self.wake_observable_type][start:end].astype(np.float32)
        t = torch.from_numpy(arr)
        if self.wake_observable_standardize:
            stats = self._load_wake_stats()
            if stats is not None:
                t = stats.standardize(t)
        return t
