"""Tests for ``src.utils.device.require_rtx6000``."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from src.utils.device import NoRTX6000Error, find_rtx6000_index, require_rtx6000


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
        assert find_rtx6000_index() is None
        with pytest.raises(NoRTX6000Error) as exc_info:
            require_rtx6000()
        msg = str(exc_info.value)
        assert "L40S" in msg
        assert "RTX 6000" in msg
