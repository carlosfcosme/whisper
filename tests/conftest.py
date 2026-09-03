import os
import random as rand
import socket
import urllib.request
from urllib.parse import urlparse

import numpy
import pytest

# Ticket 4: CPU-only, offline, no-store. Tests must not download weights.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("WHISPER_OFFLINE", "1")
os.environ.setdefault("WHISPER_NO_STORE", "1")

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_real_urlopen = urllib.request.urlopen


def _url_target(url):
    if isinstance(url, str):
        return url
    return getattr(url, "full_url", None) or str(url)


def _is_loopback(url: str) -> bool:
    host = urlparse(url).hostname or ""
    return host in _LOOPBACK_HOSTS


def _offline_urlopen(url, *args, **kwargs):
    target = _url_target(url)
    if _is_loopback(target):
        return _real_urlopen(url, *args, **kwargs)
    raise RuntimeError(f"tests must not Hub or download weights: {target}")


urllib.request.urlopen = _offline_urlopen


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line("markers", "requires_local_weights")
    try:
        import huggingface_hub

        def _no_hub(*_args, **_kwargs):
            raise RuntimeError("tests must not use Hugging Face Hub")

        huggingface_hub.hf_hub_download = _no_hub
        huggingface_hub.snapshot_download = _no_hub
    except ImportError:
        pass


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)


@pytest.fixture(autouse=True)
def _bind_loopback_only(monkeypatch):
    """Runtime tests must bind 127.0.0.1 / loopback, not a wildcard."""
    original_bind = socket.socket.bind
    wildcard = frozenset(("", "::", "[::]"))
    loopback = frozenset(("127.0.0.1", "::1", "localhost"))
    unspecified = ".".join(["0"] * 4)

    def guarded(self, address):
        if getattr(self, "family", None) not in (socket.AF_INET, socket.AF_INET6):
            return original_bind(self, address)
        host = address
        if isinstance(address, (tuple, list)) and address:
            host = address[0]
        if isinstance(host, bytes):
            host = host.decode("utf-8", "replace")
        if isinstance(host, str) and (host in wildcard or host == unspecified):
            raise OSError("runtime tests must bind 127.0.0.1, not a wildcard")
        if isinstance(host, str) and host not in loopback:
            raise OSError("runtime tests must bind 127.0.0.1, not %r" % (host,))
        return original_bind(self, address)

    monkeypatch.setattr(socket.socket, "bind", guarded)
