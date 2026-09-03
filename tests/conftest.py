import os
import random as rand

import numpy
import pytest

from tests.netguard import (
    NetworkBlocked,
    hostname_from_address,
    hostname_from_urlopen_target,
    is_loopback_host,
)

# CPU-only, no-store, no-fetch. No GPU, no weight downloads.
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
    """Unit tests may bind loopback only (not 0.0.0.0 / public)."""
    import socket

    original_bind = socket.socket.bind

    def guarded(self, address):
        host = hostname_from_address(address)
        if host in ("", "0.0.0.0", "::", "::0") or (
            isinstance(host, str) and host and not is_loopback_host(host)
        ):
            raise OSError("unit tests must bind 127.0.0.1, not a wildcard")
        return original_bind(self, address)

    monkeypatch.setattr(socket.socket, "bind", guarded)


@pytest.fixture(autouse=True)
def _block_non_loopback_network(monkeypatch):
    """No-fetch: block WAN urlopen/connect. Loopback is allowed for bind checks."""
    import socket
    import urllib.request

    original_urlopen = urllib.request.urlopen
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def guarded_urlopen(url, *args, **kwargs):
        host = hostname_from_urlopen_target(url)
        if not is_loopback_host(host):
            raise NetworkBlocked(f"network blocked: urlopen to {host!r}")
        return original_urlopen(url, *args, **kwargs)

    def guarded_connect(self, address):
        host = hostname_from_address(address)
        if isinstance(host, str) and not is_loopback_host(host):
            raise NetworkBlocked(f"network blocked: connect to {host!r}")
        return original_connect(self, address)

    def guarded_connect_ex(self, address):
        host = hostname_from_address(address)
        if isinstance(host, str) and not is_loopback_host(host):
            raise NetworkBlocked(f"network blocked: connect_ex to {host!r}")
        return original_connect_ex(self, address)

    monkeypatch.setattr(urllib.request, "urlopen", guarded_urlopen)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)


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
