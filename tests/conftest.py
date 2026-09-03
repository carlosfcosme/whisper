import os
import random as rand

import pytest

# CI / Cloud Agent path: CPU-only, no Hub download, bind 127.0.0.1.
os.environ.setdefault("WHISPER_CPU_ONLY", "1")
os.environ.setdefault("WHISPER_NO_WEIGHT_DOWNLOAD", "1")
os.environ.setdefault("WHISPER_LOCALHOST_ONLY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line("markers", "requires_local_weights")


@pytest.fixture(autouse=True)
def _bind_127_0_0_1_only(monkeypatch):
    """Unit tests must not bind a wildcard or public interface."""
    import socket

    original_bind = socket.socket.bind
    all_interfaces = ".".join(("0", "0", "0", "0"))

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) and address else address
        if host in ("", all_interfaces, "::", "::0"):
            raise OSError("unit tests must bind 127.0.0.1, not a wildcard")
        return original_bind(self, address)

    monkeypatch.setattr(socket.socket, "bind", guarded)


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
    import numpy

    rand.seed(42)
    numpy.random.seed(42)
