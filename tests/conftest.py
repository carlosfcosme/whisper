import os
import random as rand
import urllib.request
from urllib.parse import urlparse

import pytest

# Default test path is offline: no Hub / weight telemetry, no WAN fetch.
os.environ.setdefault("WHISPER_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line(
        "markers",
        "requires_weights: needs a local checkpoint (default path does not fetch)",
    )
    config.addinivalue_line(
        "markers",
        "localhost_only: localhost-only checks (no model-weight download, no WAN)",
    )


def _request_url(url) -> str:
    if isinstance(url, str):
        return url
    return getattr(url, "full_url", None) or str(url)


@pytest.fixture(autouse=True)
def _localhost_only_urlopen(monkeypatch):
    """Refuse WAN weight fetches. Localhost and file: URLs are allowed."""
    real_urlopen = urllib.request.urlopen

    def guarded(url, *args, **kwargs):
        raw = _request_url(url)
        parsed = urlparse(raw)
        host = (parsed.hostname or "").lower()
        if parsed.scheme == "file" or host in _LOOPBACK_HOSTS:
            return real_urlopen(url, *args, **kwargs)
        raise RuntimeError(
            f"tests are localhost-only and must not fetch weights from {raw!r}"
        )

    monkeypatch.setattr(urllib.request, "urlopen", guarded)


@pytest.fixture
def random():
    import numpy

    rand.seed(42)
    numpy.random.seed(42)
