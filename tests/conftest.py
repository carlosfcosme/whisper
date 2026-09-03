import os
import random as rand
from urllib.parse import urlparse

import numpy
import pytest

# Hugging Face Hub and the official Whisper weight CDN. Tests never fetch these.
_BLOCKED_HOST_SUFFIXES = (
    "huggingface.co",
    "hf.co",
    "openaipublic.azureedge.net",
)


def _url_as_str(url) -> str:
    if isinstance(url, str):
        return url
    if hasattr(url, "full_url"):
        return url.full_url
    if hasattr(url, "get_full_url"):
        return url.get_full_url()
    return str(url)


def _host_is_blocked(url) -> bool:
    host = (urlparse(_url_as_str(url)).hostname or "").lower()
    return any(
        host == suffix or host.endswith("." + suffix)
        for suffix in _BLOCKED_HOST_SUFFIXES
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line(
        "markers",
        "requires_weights: loads official checkpoints (skipped offline)",
    )
    os.environ.setdefault("WHISPER_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("WHISPER_BIND_HOST", "127.0.0.1")


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(
        reason="offline: official checkpoints are not fetched (no Hub / no weights)"
    )
    for item in items:
        if item.get_closest_marker("requires_weights"):
            item.add_marker(skip)


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)


@pytest.fixture(autouse=True)
def _block_hub_and_weight_fetches(monkeypatch):
    import urllib.request

    real_urlopen = urllib.request.urlopen

    def guarded_urlopen(url, *args, **kwargs):
        if _host_is_blocked(url):
            raise RuntimeError(
                "offline tests: blocked Hub/weight fetch: " + _url_as_str(url)
            )
        return real_urlopen(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", guarded_urlopen)
