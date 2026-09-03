"""Default inference device is CPU.

Callers may still pass ``device="cuda"`` or set ``WHISPER_DEVICE``.
The default never auto-selects CUDA from ``torch.cuda.is_available()``.
"""

from __future__ import annotations

import os

DEFAULT_DEVICE = "cpu"
DEVICE_ENV = "WHISPER_DEVICE"


def default_device() -> str:
    """Return the default PyTorch device string (CPU unless overridden)."""
    override = os.environ.get(DEVICE_ENV, "").strip()
    if override:
        return override
    return DEFAULT_DEVICE
