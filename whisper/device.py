"""Default inference device is CPU.

GPU is opt-in via an explicit ``device=`` argument or ``WHISPER_DEVICE``.
"""

from __future__ import annotations

import os


def default_device() -> str:
    override = os.environ.get("WHISPER_DEVICE")
    if override is not None and override.strip():
        return override.strip()
    return "cpu"
