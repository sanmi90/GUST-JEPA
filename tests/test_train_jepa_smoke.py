"""Slow integration test for ``src.training.train_jepa.main``.

Marked ``slow`` so CI does not run it; invoke explicitly with::

    pytest tests/test_train_jepa_smoke.py -v -m slow

The full 200-iter smoke run is exercised manually from the command line
(Session 4 deliverable). This test runs a much smaller version of the same
code path (B=2, max_iters=20, --wandb-mode disabled) to verify the wiring
catches obvious breakage if a future refactor lands.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

from src.utils.device import NoRTX6000Error, require_rtx6000


@pytest.mark.slow
def test_train_jepa_smoke_runs_to_completion(tmp_path: Path) -> None:
    """End-to-end smoke: 20 iters on the Baseline case with wandb disabled."""
    try:
        require_rtx6000()
    except NoRTX6000Error as e:
        pytest.skip(f"No RTX 6000 GPU: {e}")

    from src.training import train_jepa

    output_dir = tmp_path / "smoke"
    argv = [
        "train_jepa",
        "--partition", "v1",
        "--cases", "Baseline",
        "--max-iters", "20",
        "--seed", "0",
        "--B", "2",
        "--T", "8",
        "--H-roll", "2",
        "--num-workers", "0",
        "--log-every", "10",
        "--diagnostic-every", "10",
        "--checkpoint-every", "20",
        "--output-dir", str(output_dir),
        "--wandb-mode", "disabled",
    ]
    old_argv = sys.argv
    os.environ.setdefault("PREVENT_ROOT", os.path.expanduser("~/PREVENT"))
    os.environ.setdefault("WANDB_PROJECT", "vortex-jepa")
    try:
        sys.argv = argv
        train_jepa.main()
    finally:
        sys.argv = old_argv

    ckpts = sorted(output_dir.glob("checkpoint_iter*.pt"))
    assert ckpts, f"no checkpoint written under {output_dir}"
    blob = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    assert blob["iteration"] == 20
    assert "jepa_state_dict" in blob
    assert "run_config" in blob
    assert "RTX" in blob["run_config"]["gpu_name"] and "6000" in blob["run_config"]["gpu_name"]
