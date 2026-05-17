"""Device selection for vortex-jepa runs.

Project rule (CLAUDE.md "Hardware"): training, smoke-test, and benchmark
entrypoints must run on the RTX 6000 Blackwell GPU. The workstation also
exposes two L40S cards (sm_89) which torch may pick by default; those
must NOT be used for vortex-jepa runs so the paper compute is on a single,
named accelerator class.

Unit tests stay CPU-friendly and may import this module to skip GPU-only
paths cleanly (``pytest.skip(NoRTX6000Error(...))``).
"""

from __future__ import annotations

import os

import torch


class NoRTX6000Error(RuntimeError):
    """Raised when ``require_rtx6000`` cannot find an RTX 6000 device."""


def find_rtx6000_index() -> int | None:
    """Return the torch CUDA index of the first RTX 6000 device, or ``None``.

    Honors the current ``CUDA_VISIBLE_DEVICES`` mapping: returned indices
    refer to torch's view of CUDA devices, not nvidia-smi's PCI ordering.
    """
    if not torch.cuda.is_available():
        return None
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        if "RTX" in name and "6000" in name:
            return i
    return None


def require_rtx6000() -> torch.device:
    """Return the ``torch.device`` of a usable RTX 6000 GPU, or raise.

    Call at the top of any training, smoke-test, or benchmark entrypoint
    before constructing model / data tensors. Failing fast surfaces the
    common breakage modes: PyTorch built without sm_120, CUDA driver
    mismatch, or ``CUDA_VISIBLE_DEVICES`` hiding the card.

    After finding a candidate device by name, runs a tiny probe kernel
    (``zeros + 1``) to confirm the installed PyTorch wheel actually has
    kernels compiled for the device's compute capability. The
    ``no kernel image is available for execution on the device`` error
    that otherwise surfaces deep inside the model forward becomes a
    clean ``NoRTX6000Error`` here instead.

    Returns:
        ``torch.device("cuda:<idx>")`` where ``<idx>`` is the torch index
        of the first device whose name contains both ``RTX`` and ``6000``
        AND on which the probe kernel succeeds.

    Raises:
        NoRTX6000Error: If no RTX 6000 device is visible, or if a visible
            RTX 6000 cannot execute the probe kernel.
    """
    idx = find_rtx6000_index()
    if idx is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        if torch.cuda.is_available():
            seen = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        else:
            seen = []
        raise NoRTX6000Error(
            "vortex-jepa requires an RTX 6000 GPU (project rule, CLAUDE.md "
            '"Hardware"). '
            f"CUDA_VISIBLE_DEVICES={visible!r}, torch.cuda.is_available()="
            f"{torch.cuda.is_available()}, torch sees devices {seen!r}. "
            "Verify: (1) the NVIDIA driver is loaded (nvidia-smi); "
            "(2) the installed PyTorch wheel supports the device's compute "
            "capability (sm_120 for RTX 6000 Blackwell); "
            "(3) CUDA_VISIBLE_DEVICES does not filter the RTX 6000 out."
        )

    device = torch.device(f"cuda:{idx}")
    try:
        probe = torch.zeros(4, device=device)
        probe = probe + 1.0
        torch.cuda.synchronize(device)
    except RuntimeError as e:
        name = torch.cuda.get_device_name(idx)
        major, minor = torch.cuda.get_device_capability(idx)
        raise NoRTX6000Error(
            f"Found RTX 6000 at cuda:{idx} ({name}, sm_{major}{minor}) but a "
            f"probe kernel failed: {e}. The installed PyTorch wheel does not "
            f"appear to ship kernels for this device's compute capability. "
            "Reinstall with: pip install --upgrade --index-url "
            "https://download.pytorch.org/whl/cu128 torch torchvision torchaudio"
        ) from e
    return device
