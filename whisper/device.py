"""Inference device defaults.

CPU is the default. Pass ``device="cuda"`` (or ``--device cuda``) to use a GPU.
"""

DEFAULT_DEVICE = "cpu"


def default_device() -> str:
    return DEFAULT_DEVICE
