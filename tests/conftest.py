import os
import random as rand
from urllib.parse import urlparse

import numpy
import pytest

# Tests stay offline: no Hub, no WAN weight pull. Loopback HTTP is allowed.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")


@pytest.fixture(autouse=True)
def _no_remote_urlopen(monkeypatch):
    """Unit tests must not open remote URLs (including any Hub)."""
    import urllib.request

    real_urlopen = urllib.request.urlopen

    def guarded(url, *args, **kwargs):
        target = url if isinstance(url, str) else getattr(url, "full_url", str(url))
        host = (urlparse(target).hostname or "").lower()
        if host in {"127.0.0.1", "localhost"}:
            return real_urlopen(url, *args, **kwargs)
        raise RuntimeError(
            "unit tests must not open remote URLs (host={!r})".format(host)
        )

    monkeypatch.setattr(urllib.request, "urlopen", guarded)


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
