"""Inference device defaults."""

from typing import Optional, Union

DEFAULT_DEVICE = "cpu"


def default_device() -> str:
    """Return the default inference device.

    Always ``cpu``. Callers that have a GPU must pass ``device="cuda"``
    (or a ``torch.device``) explicitly.
    """
    return DEFAULT_DEVICE


def resolve_device(device: Optional[Union[str, object]] = None):
    """Use ``device`` when given, otherwise ``cpu``.

    Does not inspect CUDA availability.
    """
    if device is None:
        return default_device()
    return device
