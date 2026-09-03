import random as rand
import socket
import urllib.request

import numpy
import pytest

_ALL_INTERFACES = frozenset(("", "0.0.0.0", "::", "[::]"))
_LOOPBACK = frozenset(("127.0.0.1", "::1", "localhost"))
_WEIGHT_HOST_MARKERS = (
    "openaipublic.azureedge.net",
    "whisper/models",
)
_WEIGHT_SUFFIXES = (".pt", ".pth")


def _offline_requested():
    import os

    return os.getenv("WHISPER_OFFLINE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _url_text(url):
    if hasattr(url, "full_url"):
        return url.full_url
    return str(url)


def _is_weight_fetch(url):
    lowered = _url_text(url).lower()
    if any(lowered.endswith(suffix) for suffix in _WEIGHT_SUFFIXES):
        return True
    return any(marker in lowered for marker in _WEIGHT_HOST_MARKERS)


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line(
        "markers", "requires_weights: test may fetch named Whisper checkpoints"
    )


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)


@pytest.fixture(autouse=True)
def refuse_weight_downloads(request, monkeypatch):
    """Fail if a test tries to WAN-pull model weights.

    ``requires_weights`` tests may fetch unless WHISPER_OFFLINE is set (CI).
    """
    allowed = request.node.get_closest_marker("requires_weights") is not None
    if allowed and not _offline_requested():
        return

    original = urllib.request.urlopen

    def guarded_urlopen(url, *args, **kwargs):
        if _is_weight_fetch(url):
            raise RuntimeError(
                "WAN model weight download is blocked in tests/offline: %s"
                % (_url_text(url),)
            )
        return original(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", guarded_urlopen)


@pytest.fixture(autouse=True)
def refuse_non_localhost_binds(monkeypatch):
    """Services in tests must bind 127.0.0.1 / loopback only."""
    original_bind = socket.socket.bind

    def guarded_bind(self, address):
        if getattr(self, "family", None) not in (socket.AF_INET, socket.AF_INET6):
            return original_bind(self, address)
        host = address
        if isinstance(address, (tuple, list)) and address:
            host = address[0]
        if isinstance(host, bytes):
            host = host.decode("utf-8", "replace")
        if isinstance(host, str) and host in _ALL_INTERFACES:
            raise RuntimeError(
                "services must bind 127.0.0.1 only; refused %r" % (address,)
            )
        if isinstance(host, str) and host not in _LOOPBACK:
            raise RuntimeError(
                "services must bind 127.0.0.1 only; refused %r" % (address,)
            )
        return original_bind(self, address)

    monkeypatch.setattr(socket.socket, "bind", guarded_bind)
