"""CPU-only default inference path. No CUDA requirement."""

from __future__ import annotations

import os

DEFAULT_DEVICE = "cpu"
DEVICE_ENV = "WHISPER_DEVICE"


def default_device() -> str:
    """Return the default PyTorch device (CPU unless ``WHISPER_DEVICE`` is set).

    Never auto-selects CUDA from ``torch.cuda.is_available()``.
    """
    override = os.environ.get(DEVICE_ENV, "").strip()
    if override:
        return override
    return DEFAULT_DEVICE
