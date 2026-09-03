import os
import random as rand

import numpy
import pytest

# Ticket 4: CPU-only, offline/no-store. No GPU, no weight downloads.
os.environ.setdefault("WHISPER_CPU_ONLY", "1")
os.environ.setdefault("WHISPER_NO_STORE", "1")
os.environ.setdefault("WHISPER_OFFLINE", "1")
os.environ.setdefault("WHISPER_NO_WEIGHT_DOWNLOAD", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")


@pytest.fixture(autouse=True)
def _bind_127_0_0_1_only(monkeypatch):
    """No 0.0.0.0 in unit tests."""
    import socket

    original_bind = socket.socket.bind

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) and address else address
        if host in ("", "0.0.0.0", "::", "::0"):
            raise OSError("unit tests must bind 127.0.0.1, not a wildcard")
        return original_bind(self, address)

    monkeypatch.setattr(socket.socket, "bind", guarded)


@pytest.fixture(autouse=True)
def _no_hf_hub_pull(monkeypatch):
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
