"""CPU-only default inference device. Pass device='cuda' to use a GPU."""

DEFAULT_DEVICE = "cpu"


def default_device() -> str:
    return DEFAULT_DEVICE
