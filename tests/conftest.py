import os
import random as rand
import socket
import urllib.request

try:
    import numpy
except ImportError:  # bind-and-weights CI installs pytest only
    numpy = None

import pytest

# Tests must not hit Hugging Face Hub or download official weights.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("WHISPER_NO_WEIGHT_DOWNLOAD", "1")

_REAL_URLOPEN = urllib.request.urlopen
_REAL_CREATE_CONNECTION = socket.create_connection
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line(
        "markers",
        "requires_local_weights: needs a cached Whisper checkpoint on disk",
    )


@pytest.fixture
def random():
    rand.seed(42)
    if numpy is not None:
        numpy.random.seed(42)


def _request_url(url):
    if isinstance(url, str):
        return url
    return getattr(url, "full_url", None) or str(url)


def _is_loopback_url(url):
    target = _request_url(url)
    return target.startswith("http://127.0.0.1")


def _is_loopback_host(host):
    if host is None:
        return False
    raw = host.decode("ascii", "replace") if isinstance(host, bytes) else str(host)
    return raw.strip("[]").lower().rstrip(".") in _LOOPBACK_HOSTS


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Empty XDG cache so installer/model paths cannot reuse downloaded weights."""
    cache = tmp_path / "xdg-cache"
    cache.mkdir()
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    monkeypatch.setenv("WHISPER_NO_WEIGHT_DOWNLOAD", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    return cache


@pytest.fixture
def loopback_bind():
    """Serve/listen tests must bind this host only."""
    return "127.0.0.1"


@pytest.fixture(autouse=True)
def _forbid_hub_and_remote_downloads(monkeypatch):
    """Block Hub / remote weight downloads; 127.0.0.1 stays allowed."""

    def _blocked_urlopen(url, *args, **kwargs):
        if _is_loopback_url(url):
            return _REAL_URLOPEN(url, *args, **kwargs)
        raise RuntimeError(
            "Network / Hub downloads are forbidden in tests. "
            "Requested: {}".format(_request_url(url))
        )

    def _blocked_connect(address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if _is_loopback_host(host):
            return _REAL_CREATE_CONNECTION(address, *args, **kwargs)
        raise RuntimeError(
            "Network / Hub downloads are forbidden in tests. "
            "Requested: {}".format(address)
        )

    monkeypatch.setattr(urllib.request, "urlopen", _blocked_urlopen)
    monkeypatch.setattr(socket, "create_connection", _blocked_connect)

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
            monkeypatch.setattr(huggingface_hub, name, _blocked_urlopen, raising=False)
