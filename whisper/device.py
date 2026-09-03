"""CPU is the default inference device."""

import os
from typing import Optional, Union

import torch

DEFAULT_DEVICE = "cpu"
DEVICE_ENV = "WHISPER_DEVICE"


def default_device() -> str:
    """Return the default PyTorch device.

    Always CPU unless ``WHISPER_DEVICE`` is set. CUDA availability does not
    change the default; pass ``device="cuda"`` (or ``--device cuda``) to opt in.
    """
    override = os.environ.get(DEVICE_ENV)
    if override:
        return override
    return DEFAULT_DEVICE


def resolve_device(
    device: Optional[Union[str, torch.device]] = None,
) -> Union[str, torch.device]:
    if device is None:
        return default_device()
    return device
