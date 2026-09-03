import os
import random as rand
from urllib.parse import urlparse

import numpy
import pytest

from whisper.runtime import (
    CPU_ONLY_ENV,
    LOCALHOST_ONLY_ENV,
    NO_WEIGHT_DOWNLOAD_ENV,
)

# CI / unit-test path: CPU default, no Hub, bind 127.0.0.1.
os.environ.setdefault(CPU_ONLY_ENV, "1")
os.environ.setdefault(NO_WEIGHT_DOWNLOAD_ENV, "1")
os.environ.setdefault(LOCALHOST_ONLY_ENV, "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")


@pytest.fixture(autouse=True)
def _bind_127_0_0_1_only(monkeypatch):
    """Unit tests must not bind a wildcard or public interface."""
    import socket

    original_bind = socket.socket.bind

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) and address else address
        if host in ("", "0.0.0.0", "::", "::0"):
            raise OSError("unit tests must bind 127.0.0.1, not a wildcard")
        return original_bind(self, address)

    monkeypatch.setattr(socket.socket, "bind", guarded)


@pytest.fixture(autouse=True)
def _no_remote_urlopen(monkeypatch):
    """Unit tests may talk to 127.0.0.1 only. No Hub, no WAN."""
    import socket
    import urllib.request

    real = urllib.request.urlopen

    def guarded(url, *args, **kwargs):
        target = url if isinstance(url, str) else getattr(url, "full_url", str(url))
        host = (urlparse(target).hostname or "").lower()
        if host in {"127.0.0.1", "localhost", "::1"}:
            return real(url, *args, **kwargs)
        raise RuntimeError("unit tests must not open remote URL: {}".format(target))

    monkeypatch.setattr(urllib.request, "urlopen", guarded)

    real_cc = socket.create_connection

    def guarded_cc(address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if str(host).lower() not in {"127.0.0.1", "localhost", "::1"}:
            raise OSError(
                "unit tests must connect to 127.0.0.1, not {!r}".format(address)
            )
        return real_cc(address, *args, **kwargs)

    monkeypatch.setattr(socket, "create_connection", guarded_cc)


@pytest.fixture(autouse=True)
def _no_hf_hub_client(monkeypatch):
    """If huggingface_hub is installed, its download APIs must not run."""

    def blocked(*args, **kwargs):
        raise RuntimeError("Hugging Face Hub pull is not allowed in unit tests")

    try:
        import huggingface_hub
    except ImportError:
        return
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", blocked, raising=False)
    if hasattr(huggingface_hub, "snapshot_download"):
        monkeypatch.setattr(
            huggingface_hub, "snapshot_download", blocked, raising=False
        )


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
