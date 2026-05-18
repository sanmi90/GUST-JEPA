"""Dataset-level tests for the partition v1 cache and the Session 6
``emit_cl_future`` extension to ``src.data.episode_dataset.EpisodeDataset``.

The tests require the preprocessed cache. They skip cleanly when
``PREVENT_ROOT`` is unset or the cache directory is missing, so CI on a
clean checkout passes without DNS data.

Session 6 motivation (SESSION6_FACTORIAL_DIAGNOSTIC.md Step 1):
    The F-OBS run trains an auxiliary head that maps z_t -> CL at
    (t + 8, t + 16, t + 24) frames. The data loader must emit a
    ``(L, 3)`` tensor of future CL values per sample, with
    end-of-encounter clamping to the last valid frame.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import yaml

from src.data.episode_dataset import EpisodeDataset


def _cache_available() -> bool:
    cache = os.environ.get("VORTEX_JEPA_CACHE")
    if cache is None:
        prevent = os.environ.get("PREVENT_ROOT")
        if not prevent:
            return False
        cache = str(Path(prevent) / "data" / "processed" / "vortex-jepa")
    return (Path(cache) / "v1" / "Baseline" / "encounter_00.h5").exists()


pytestmark = pytest.mark.skipif(
    not _cache_available(),
    reason="partition v1 cache not present (need PREVENT_ROOT and preprocessed encounters)",
)


def _smoke_5_case_ids() -> list[str]:
    repo = Path(__file__).resolve().parents[1]
    with open(repo / "configs" / "cases" / "smoke_5cases.yaml") as f:
        return list(yaml.safe_load(f)["cases"])


def test_cl_future_shape_contract() -> None:
    """With ``emit_cl_future=True``, each sample carries a ``cl_future``
    tensor of shape ``(L, n_deltas)`` and dtype ``float32``.
    """
    ds = EpisodeDataset(
        partition="v1",
        split="train",
        subtraj_len=32,
        emit_cl_future=True,
        cl_future_deltas=(8, 16, 24),
    )
    sample = ds[0]
    assert "cl_future" in sample, "emit_cl_future=True did not produce cl_future"
    cl = sample["cl_future"]
    assert isinstance(cl, torch.Tensor)
    assert cl.shape == (32, 3), f"expected (32, 3); got {tuple(cl.shape)}"
    assert cl.dtype == torch.float32, f"expected float32; got {cl.dtype}"

    # default: emit_cl_future=False keeps the legacy contract (backwards-compatible)
    ds_legacy = EpisodeDataset(partition="v1", split="train", subtraj_len=32)
    s_legacy = ds_legacy[0]
    assert "cl_future" not in s_legacy, "legacy default must not emit cl_future"


def test_cl_future_no_nans_on_smoke_subset() -> None:
    """No NaN or Inf across the 5-case smoke subset. End-of-encounter
    clamping makes the tensor finite even for the last sub-trajectory.
    """
    smoke_ids = set(_smoke_5_case_ids())
    ds = EpisodeDataset(
        partition="v1",
        split="train",
        subtraj_len=32,
        emit_cl_future=True,
        cl_future_deltas=(8, 16, 24),
        seed=0,
    )
    ds.samples = [s for s in ds.samples if s[0] in smoke_ids]
    assert ds.samples, "no smoke samples found in train split"

    cl_min = float("inf")
    cl_max = float("-inf")
    for i in range(len(ds)):
        cl = ds[i]["cl_future"]
        assert torch.isfinite(cl).all(), f"sample {i}: non-finite cl_future"
        cl_min = min(cl_min, float(cl.min()))
        cl_max = max(cl_max, float(cl.max()))
    # Smoke subset includes G=+/-3 cases whose impact spikes reach |CL| ~ 8;
    # the soft envelope (-10, 10) is a sanity bound, not a tight prior.
    assert -10.0 < cl_min < 10.0 and -10.0 < cl_max < 10.0, (
        f"cl_future range [{cl_min:.3f}, {cl_max:.3f}] outside plausible CL envelope"
    )


def test_cl_future_clamps_at_encounter_end() -> None:
    """Force a near-end start so frame_start + L + max_delta > n_frames.
    The final entries of cl_future must equal the last valid C_L of the
    encounter (clamping behaviour, per D36 design call).
    """
    # near-end uniform start by zeroing the impact-aware branch and
    # restricting uniform_start_range to the largest valid start
    ds = EpisodeDataset(
        partition="v1",
        split="train",
        subtraj_len=32,
        emit_cl_future=True,
        cl_future_deltas=(8, 16, 24),
        impact_aware_fraction=0.0,
        uniform_start_range=(88, 88),  # last valid start for L=32 in a 120-frame encounter
        seed=0,
    )
    sample = ds[0]
    cl = sample["cl_future"]
    # legacy C_L of the same window for the last-frame reference
    C_L = sample["C_L"]
    # last frame of the sub-trajectory is at frame_start + L - 1 = 88 + 31 = 119
    # for delta=8/16/24 every entry beyond i s.t. 88 + i + delta >= 120 must clamp.
    # i + delta >= 32  ->  i >= 32 - delta. So delta=8: i >= 24, delta=16: i >= 16, delta=24: i >= 8.
    # We test the most-clamped column (delta=24, all i >= 8 should equal cl_full's last entry).
    # cl_full's last entry under this start equals C_L[31] (the last frame of the sub-trajectory).
    last_valid = float(C_L[-1])
    # Every cl_future[i, 2] for i >= 8 must equal last_valid (frame 88 + i + 24 >= 120 -> clamp)
    clamped = cl[8:, 2]
    assert torch.allclose(clamped, torch.full_like(clamped, last_valid), atol=1e-5), (
        f"clamped column (delta=24, i>=8) is not constant at last_valid={last_valid}; got {clamped[:5].tolist()}"
    )
