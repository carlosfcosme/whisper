import os
import random as rand

import numpy
import pytest

# Cloud Agent / CI path: CPU-only default and no weight auto-download.
# setdefault so a caller can override (e.g. WHISPER_ALLOW_WEIGHT_DOWNLOAD=1).
os.environ.setdefault("WHISPER_CPU_ONLY", "1")
os.environ.setdefault("WHISPER_NO_WEIGHT_DOWNLOAD", "1")


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")


@pytest.fixture(autouse=True)
def _no_hf_hub_pull(monkeypatch):
    """Unit tests must not pull weights from the Hugging Face Hub."""

    def _blocked(*args, **kwargs):
        raise RuntimeError("Hugging Face Hub pull is not allowed in unit tests")

    try:
        import huggingface_hub
    except ImportError:
        return
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _blocked, raising=False)
    if hasattr(huggingface_hub, "snapshot_download"):
        monkeypatch.setattr(
            huggingface_hub, "snapshot_download", _blocked, raising=False
        )


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
