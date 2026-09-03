"""Inference device defaults.

Whisper defaults to CPU. Callers that have a GPU must pass ``device="cuda"``
(or another torch device) explicitly. Availability of CUDA does not change
the default.
"""

from __future__ import annotations

from typing import Any, Optional

DEFAULT_DEVICE = "cpu"


def resolve_device(device: Optional[Any] = None) -> Any:
    """Return ``device`` if given, otherwise the CPU default."""
    if device is None:
        return DEFAULT_DEVICE
    return device
