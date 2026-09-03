import os
import random as rand

import numpy
import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")


@pytest.fixture(autouse=True)
def block_wan_urlopen(monkeypatch):
    """Fail any non-loopback model/network fetch during tests."""
    import urllib.request

    try:
        from whisper.offline import is_loopback_url
    except ImportError:
        return

    real = urllib.request.urlopen

    def guarded(url, *args, **kwargs):
        target = url if isinstance(url, str) else getattr(url, "full_url", str(url))
        if is_loopback_url(str(target)):
            return real(url, *args, **kwargs)
        raise RuntimeError("WAN/Hub fetch blocked in tests: {0}".format(target))

    monkeypatch.setattr(urllib.request, "urlopen", guarded)


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
