import os
import random as rand
import urllib.request
from urllib.parse import urlparse

import pytest

try:
    import numpy
except ImportError:  # offline-bind-guards CI installs pytest only
    numpy = None

OFFLINE_ENV_DEFAULTS = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "WHISPER_OFFLINE": "1",
    "WHISPER_NO_WEIGHT_DOWNLOAD": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line("markers", "requires_weights")
    for key, value in OFFLINE_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)


@pytest.fixture
def random():
    rand.seed(42)
    if numpy is not None:
        numpy.random.seed(42)


@pytest.fixture(autouse=True)
def _block_wan_downloads(monkeypatch):
    """Fail if a test opens a non-loopback URL (no WAN weight downloads)."""
    real_urlopen = urllib.request.urlopen

    def urlopen(url, *args, **kwargs):
        target = url if isinstance(url, str) else getattr(url, "full_url", str(url))
        host = (urlparse(str(target)).hostname or "").lower()
        if host in {"127.0.0.1", "localhost", "::1"}:
            return real_urlopen(url, *args, **kwargs)
        raise AssertionError(f"WAN urlopen blocked in tests: {target!r}")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
