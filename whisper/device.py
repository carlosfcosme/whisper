"""Inference device defaults."""

DEFAULT_DEVICE = "cpu"


def default_device() -> str:
    """Return the default inference device.

    Always ``cpu``. Callers that have a GPU must pass ``device="cuda"``
    (or a ``torch.device``) explicitly.
    """
    return DEFAULT_DEVICE
