import ipaddress
import os
import random as rand
import urllib.request
from urllib.parse import urlparse

import numpy
import pytest

# Tests never talk to the Hugging Face Hub or Azure weight CDN.
os.environ["WHISPER_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.pop("WHISPER_ALLOW_DOWNLOADS", None)

_REAL_URLOPEN = urllib.request.urlopen


class DownloadAttempted(RuntimeError):
    """Raised when a test tries to fetch weights or Hub artifacts."""


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line(
        "markers",
        "requires_local_weights: needs a cached Whisper checkpoint on disk",
    )


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)


def _request_url(url):
    if isinstance(url, str):
        return url
    return getattr(url, "full_url", None) or str(url)


def _is_loopback_url(url):
    """True only when the *parsed hostname* is loopback.

    Prefix matching is not enough: ``http://localhost.evil.example/`` and
    ``http://localhost@evil.example/`` must stay blocked.
    """
    target = _request_url(url)
    try:
        parsed = urlparse(target)
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    host = parsed.hostname
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def _forbid_hub_and_remote_downloads(monkeypatch):
    """Fail the test if a Hub / WAN download is attempted. Loopback is ok."""

    def _blocked(url, *args, **kwargs):
        if _is_loopback_url(url):
            return _REAL_URLOPEN(url, *args, **kwargs)
        raise DownloadAttempted(
            "Network / Hub downloads are forbidden in tests (offline). "
            "Requested: {}".format(_request_url(url))
        )

    monkeypatch.setattr(urllib.request, "urlopen", _blocked)

    try:
        import huggingface_hub
    except ImportError:
        return

    for name in (
        "hf_hub_download",
        "snapshot_download",
        "hf_hub_url",
        "cached_download",
    ):
        if hasattr(huggingface_hub, name):
            monkeypatch.setattr(huggingface_hub, name, _blocked, raising=False)
