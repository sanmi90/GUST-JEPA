"""Device selection for vortex-jepa runs.

Project rule (CLAUDE.md "Hardware"): training, smoke-test, and benchmark
entrypoints must run on an RTX 6000 Blackwell GPU. The workstation exposes
two RTX 6000 Blackwell cards (sm_120) and two L40S cards (sm_89). The
L40S cards must NOT be used so paper compute is on a single, named
accelerator class.

Two-card pattern (D40): ``require_rtx6000(gpu_index=...)`` selects between
the available RTX 6000 cards. Training entrypoints expose this as ``--gpu``
so two independent runs (e.g. Session 7 R1 vs R2) can run in parallel
without shell-level ``CUDA_VISIBLE_DEVICES`` tricks.

Unit tests stay CPU-friendly and may import this module to skip GPU-only
paths cleanly (``pytest.skip(NoRTX6000Error(...))``).
"""

from __future__ import annotations

import os

import torch


class NoRTX6000Error(RuntimeError):
    """Raised when ``require_rtx6000`` cannot find a usable RTX 6000 device."""


def find_rtx6000_indices() -> list[int]:
    """Return torch CUDA indices of every RTX 6000 device, in torch's order.

    Honors the current ``CUDA_VISIBLE_DEVICES`` mapping: returned indices
    refer to torch's view of CUDA devices, not nvidia-smi's PCI ordering.
    Returns ``[]`` when CUDA is unavailable or no RTX 6000 is visible.
    """
    if not torch.cuda.is_available():
        return []
    return [
        i
        for i in range(torch.cuda.device_count())
        if "RTX" in torch.cuda.get_device_name(i)
        and "6000" in torch.cuda.get_device_name(i)
    ]


def find_rtx6000_index() -> int | None:
    """Backward-compatible alias for ``find_rtx6000_indices()[0]``."""
    indices = find_rtx6000_indices()
    return indices[0] if indices else None


def require_rtx6000(gpu_index: int | None = None) -> torch.device:
    """Return the ``torch.device`` of a usable RTX 6000 GPU, or raise.

    Call at the top of any training, smoke-test, or benchmark entrypoint
    before constructing model / data tensors. Failing fast surfaces the
    common breakage modes: PyTorch built without sm_120, CUDA driver
    mismatch, or ``CUDA_VISIBLE_DEVICES`` hiding the card.

    After finding the candidate device by name, runs a tiny probe kernel
    (``zeros + 1``) to confirm the installed PyTorch wheel actually has
    kernels compiled for the device's compute capability. The
    ``no kernel image is available for execution on the device`` error
    that otherwise surfaces deep inside the model forward becomes a
    clean ``NoRTX6000Error`` here instead.

    Args:
        gpu_index: 0-indexed selector into the *RTX 6000 subset*
            (NOT torch's full CUDA enumeration). With two RTX 6000 cards
            visible, ``gpu_index=0`` picks the first one and ``gpu_index=1``
            picks the second one. ``None`` (default) picks the first one,
            preserving the pre-D40 single-card behaviour.

    Returns:
        ``torch.device("cuda:<idx>")`` where ``<idx>`` is the torch index
        of the selected RTX 6000 device.

    Raises:
        NoRTX6000Error: If no RTX 6000 device is visible, if ``gpu_index``
            exceeds the number of RTX 6000 cards, or if the selected
            device fails the probe kernel.
    """
    # One-off opt-in bypass for the L40S cards. When
    # ``VORTEX_JEPA_ALLOW_NON_RTX6000=1`` (or =idx,idx,...), the helper
    # ALSO accepts non-RTX-6000 devices; gpu_index then indexes into the
    # full CUDA enumeration. This is a temporary hatch for the split_v2
    # rerun where the user explicitly opened the L40S cards for parallel
    # workloads; production paper-grade runs still default to RTX-only.
    allow_non = os.environ.get("VORTEX_JEPA_ALLOW_NON_RTX6000", "").strip()
    if allow_non and torch.cuda.is_available():
        if gpu_index is None:
            gpu_index = 0
        n = torch.cuda.device_count()
        if gpu_index < 0 or gpu_index >= n:
            raise NoRTX6000Error(
                f"--gpu={gpu_index} requested with NON_RTX6000 bypass, but only "
                f"{n} CUDA device(s) visible."
            )
        device = torch.device(f"cuda:{gpu_index}")
        name = torch.cuda.get_device_name(gpu_index)
        print(f"[device] WARNING: VORTEX_JEPA_ALLOW_NON_RTX6000 bypass active; "
              f"using cuda:{gpu_index} ({name})", flush=True)
        try:
            probe = torch.zeros(4, device=device) + 1.0
            torch.cuda.synchronize(device)
        except RuntimeError as e:
            raise NoRTX6000Error(
                f"NON_RTX6000 bypass: probe kernel failed on cuda:{gpu_index} "
                f"({name}): {e}"
            ) from e
        return device

    indices = find_rtx6000_indices()
    if not indices:
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

    if gpu_index is None:
        idx = indices[0]
    else:
        if gpu_index < 0 or gpu_index >= len(indices):
            raise NoRTX6000Error(
                f"--gpu={gpu_index} requested but only {len(indices)} RTX 6000 "
                f"card(s) are visible (valid gpu_index in [0, {len(indices) - 1}]). "
                f"Torch sees devices {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}."
            )
        idx = indices[gpu_index]

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
