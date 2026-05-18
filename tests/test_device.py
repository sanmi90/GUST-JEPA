"""Tests for ``src.utils.device.require_rtx6000``."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from src.utils.device import (
    NoRTX6000Error,
    find_rtx6000_index,
    find_rtx6000_indices,
    require_rtx6000,
)


def test_require_rtx6000_on_workstation() -> None:
    """On the Carlos workstation (RTX 6000 visible) returns a usable device.

    Skipped if no CUDA is available so the suite stays portable.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this machine")
    try:
        device = require_rtx6000()
    except NoRTX6000Error as e:
        pytest.skip(f"No RTX 6000 visible to torch: {e}")
    assert device.type == "cuda"
    name = torch.cuda.get_device_name(device.index)
    assert "RTX" in name and "6000" in name


def test_no_rtx6000_error_message_lists_visible_gpus() -> None:
    """When no RTX 6000 is found, the error lists what torch did see.

    Patches ``torch.cuda.is_available`` / ``device_count`` / ``get_device_name``
    so the test runs the same on machines with or without CUDA.
    """
    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
        torch.cuda, "device_count", return_value=2
    ), patch.object(
        torch.cuda,
        "get_device_name",
        side_effect=lambda i: ["NVIDIA L40S", "NVIDIA L40S"][i],
    ):
        assert find_rtx6000_indices() == []
        assert find_rtx6000_index() is None
        with pytest.raises(NoRTX6000Error) as exc_info:
            require_rtx6000()
        msg = str(exc_info.value)
        assert "L40S" in msg
        assert "RTX 6000" in msg


def test_find_rtx6000_indices_returns_all_with_two_cards() -> None:
    """A workstation with two RTX 6000s + two L40S returns both RTX indices."""
    names = [
        "NVIDIA L40S",
        "NVIDIA L40S",
        "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
        "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
    ]
    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
        torch.cuda, "device_count", return_value=4
    ), patch.object(torch.cuda, "get_device_name", side_effect=lambda i: names[i]):
        assert find_rtx6000_indices() == [2, 3]
        # backward-compat alias still returns the first one
        assert find_rtx6000_index() == 2


def test_require_rtx6000_gpu_index_out_of_range_errors() -> None:
    """gpu_index >= number-of-RTX-6000s raises NoRTX6000Error with a clear message."""
    names = [
        "NVIDIA L40S",
        "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
    ]
    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
        torch.cuda, "device_count", return_value=2
    ), patch.object(torch.cuda, "get_device_name", side_effect=lambda i: names[i]):
        # gpu_index=0 would succeed on a real GPU (we don't have one mocked)
        # so we just test the out-of-range error path.
        with pytest.raises(NoRTX6000Error) as exc_info:
            require_rtx6000(gpu_index=1)
        msg = str(exc_info.value)
        assert "gpu=1" in msg or "gpu_index=1" in msg
        assert "1 RTX 6000" in msg


def test_require_rtx6000_gpu_index_negative_errors() -> None:
    """Negative gpu_index is rejected (must be in [0, n_rtx))."""
    names = ["NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"]
    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
        torch.cuda, "device_count", return_value=1
    ), patch.object(torch.cuda, "get_device_name", side_effect=lambda i: names[i]):
        with pytest.raises(NoRTX6000Error):
            require_rtx6000(gpu_index=-1)


def test_require_rtx6000_gpu_index_selects_correct_card_on_workstation() -> None:
    """Live workstation test: with multiple RTX 6000s, gpu_index=N picks the
    Nth one. Skipped when only one RTX 6000 is available.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this machine")
    indices = find_rtx6000_indices()
    if len(indices) < 2:
        pytest.skip(f"need >=2 RTX 6000s for this test; saw {len(indices)}")
    dev0 = require_rtx6000(gpu_index=0)
    dev1 = require_rtx6000(gpu_index=1)
    assert dev0.index == indices[0]
    assert dev1.index == indices[1]
    assert dev0.index != dev1.index
