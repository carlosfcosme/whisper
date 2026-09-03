"""Default inference device.

Always CPU. Callers that want a GPU must pass ``device="cuda"``
explicitly. This module does not import torch, load weights, or
talk to Hugging Face Hub.
"""


def default_device() -> str:
    """Return the default inference device (always ``cpu``)."""
    return "cpu"
