import os
import random as rand
import sys
import urllib.request
from urllib.parse import urlparse

import numpy
import pytest

# Tests and CI must not fetch Hub artifacts or model weights.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("WHISPER_NO_WEIGHT_DOWNLOAD", "1")

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_original_urlopen = urllib.request.urlopen


def _request_url(url) -> str:
    if hasattr(url, "full_url"):
        return url.full_url
    if isinstance(url, bytes):
        return url.decode("utf-8", "replace")
    return str(url)


def _is_allowed_url(raw: str) -> bool:
    parsed = urlparse(raw)
    if parsed.scheme in {"", "file"}:
        return True
    host = (parsed.hostname or "").lower().rstrip(".")
    return host in _LOOPBACK_HOSTS


def urlopen_without_network(url, *args, **kwargs):
    raw = _request_url(url)
    if _is_allowed_url(raw):
        return _original_urlopen(url, *args, **kwargs)
    host = urlparse(raw).hostname or raw
    raise RuntimeError("tests must not make network calls (%s)" % host)


urllib.request.urlopen = urlopen_without_network


def _block_huggingface_hub() -> None:
    try:
        import huggingface_hub
    except ImportError:
        return

    def _forbidden(*_args, **_kwargs):
        raise RuntimeError("huggingface_hub is forbidden in tests")

    huggingface_hub.hf_hub_download = _forbidden
    if hasattr(huggingface_hub, "snapshot_download"):
        huggingface_hub.snapshot_download = _forbidden


_block_huggingface_hub()


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line("markers", "requires_weights")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("WHISPER_ALLOW_WEIGHT_DOWNLOAD") == "1":
        return
    skip = pytest.mark.skip(reason="tests must not fetch model weights")
    for item in items:
        if "requires_weights" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)


@pytest.fixture
def forbid_network_calls(monkeypatch):
    """Fail the test if a non-loopback network call is attempted."""

    def _fail(url, *args, **kwargs):
        raw = _request_url(url)
        if _is_allowed_url(raw):
            return _original_urlopen(url, *args, **kwargs)
        raise RuntimeError("network-call monkeypatch: weight fetch forbidden")

    monkeypatch.setattr(urllib.request, "urlopen", _fail)
    if "whisper" in sys.modules:
        monkeypatch.setattr(sys.modules["whisper"].urllib.request, "urlopen", _fail)
    return _fail
