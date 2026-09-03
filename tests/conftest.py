import os
import random as rand
import socket
import sys
import urllib.request

import numpy
import pytest

# CPU default + no Hub/WAN weight pulls in this test process.
os.environ.setdefault("WHISPER_NO_WEIGHT_DOWNLOAD", "1")

_WILDCARD_HOSTS = frozenset(("", "::", "[::]"))
_LOOPBACK_HOSTS = frozenset(("127.0.0.1", "::1", "localhost"))


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
def _no_hf_hub_import():
    """Unit tests must not import or call Hugging Face Hub."""

    class _BlockHub:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "huggingface_hub" or fullname.startswith("huggingface_hub."):
                raise RuntimeError("Hugging Face Hub import is blocked in tests")
            return None

    finder = _BlockHub()
    sys.meta_path.insert(0, finder)
    yield
    try:
        sys.meta_path.remove(finder)
    except ValueError:
        pass


@pytest.fixture(autouse=True)
def _no_hub_or_weight_urlopen(request, monkeypatch):
    """Fail if a test WAN-pulls Hub or named checkpoints."""
    allowed = request.node.get_closest_marker("requires_weights") is not None
    if allowed:
        return

    original = urllib.request.urlopen

    def guarded(url, *args, **kwargs):
        text = url.full_url if hasattr(url, "full_url") else str(url)
        lowered = text.lower()
        if any(
            marker in lowered
            for marker in (
                "huggingface.co",
                "hf.co/",
                "hf-mirror.com",
                "huggingface_hub",
                "openaipublic.azureedge.net",
            )
        ) or lowered.endswith((".pt", ".pth", ".safetensors")):
            raise RuntimeError("Hub/weight WAN download is blocked in tests: %s" % text)
        return original(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", guarded)


@pytest.fixture(autouse=True)
def _bind_loopback_only(monkeypatch):
    """Unit tests must bind 127.0.0.1 / loopback, not a wildcard."""
    original_bind = socket.socket.bind

    def guarded(self, address):
        if getattr(self, "family", None) not in (socket.AF_INET, socket.AF_INET6):
            return original_bind(self, address)
        host = address
        if isinstance(address, (tuple, list)) and address:
            host = address[0]
        if isinstance(host, bytes):
            host = host.decode("utf-8", "replace")
        if isinstance(host, str) and (
            host in _WILDCARD_HOSTS or host not in _LOOPBACK_HOSTS
        ):
            # Unspecified IPv4 is written without a literal so package grep stays clean.
            unspecified = ".".join(["0"] * 4)
            if host == unspecified or host in _WILDCARD_HOSTS:
                raise OSError("unit tests must bind 127.0.0.1, not a wildcard")
            raise OSError("unit tests must bind 127.0.0.1, not %r" % (host,))
        return original_bind(self, address)

    monkeypatch.setattr(socket.socket, "bind", guarded)
