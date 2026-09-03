"""Inference device policy. CPU is the default; CUDA is opt-in."""

from __future__ import annotations

import os

CPU_DEVICE = "cpu"


def default_device() -> str:
    """Return the default PyTorch device.

    CPU unless ``WHISPER_DEVICE`` is set (e.g. ``cuda``). Callers can still
    pass an explicit ``device=`` to ``load_model`` or ``--device`` on the CLI.
    """
    override = os.getenv("WHISPER_DEVICE")
    if override is None:
        return CPU_DEVICE
    value = override.strip()
    return value if value else CPU_DEVICE
