import os

# Must run before torch / whisper imports so CI stays CPU and offline.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("WHISPER_OFFLINE", "1")
os.environ.setdefault("WHISPER_NO_STORE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import random as rand  # noqa: E402
import socket  # noqa: E402
import urllib.request  # noqa: E402

import numpy  # noqa: E402
import pytest  # noqa: E402

from whisper.netguard import is_loopback_connect, refuse_non_loopback  # noqa: E402

_REAL_URLOPEN = urllib.request.urlopen
_REAL_CONNECT = socket.socket.connect
_REAL_CONNECT_EX = socket.socket.connect_ex
_REAL_CREATE_CONNECTION = socket.create_connection


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
    target = _request_url(url)
    return target.startswith("http://127.0.0.1") or target.startswith(
        "http://localhost"
    )


@pytest.fixture(autouse=True)
def _forbid_hub_and_remote_downloads(monkeypatch):
    """Block WAN / Hub; loopback (serve) stays allowed."""

    def _blocked_urlopen(url, *args, **kwargs):
        if _is_loopback_url(url):
            return _REAL_URLOPEN(url, *args, **kwargs)
        raise RuntimeError(
            "Network / Hub downloads are forbidden in tests (offline). "
            "Requested: {}".format(_request_url(url))
        )

    def _blocked_connect(self, address):
        refuse_non_loopback(address)
        return _REAL_CONNECT(self, address)

    def _blocked_connect_ex(self, address):
        if not is_loopback_connect(address):
            refuse_non_loopback(address)
        return _REAL_CONNECT_EX(self, address)

    def _blocked_create_connection(address, *args, **kwargs):
        refuse_non_loopback(address)
        return _REAL_CREATE_CONNECTION(address, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", _blocked_urlopen)
    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked_connect_ex)
    monkeypatch.setattr(socket, "create_connection", _blocked_create_connection)

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
