import ipaddress
import os
import random as rand
import socket
import urllib.request

import numpy
import pytest

# CPU-only, offline test default. Must be set before test modules import torch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["WHISPER_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

_REAL_URLOPEN = urllib.request.urlopen
_REAL_CONNECT = socket.socket.connect
_REAL_CONNECT_EX = socket.socket.connect_ex


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


def _is_loopback_host(host):
    raw = str(host or "").strip().lower()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    raw = raw.split("%", 1)[0]
    if raw in {"127.0.0.1", "::1", "localhost"}:
        return True
    try:
        return ipaddress.ip_address(raw).is_loopback
    except ValueError:
        return False


def _is_loopback_url(url):
    target = _request_url(url)
    return target.startswith("http://127.0.0.1") or target.startswith(
        "http://localhost"
    )


def _wan_forbidden(host):
    return RuntimeError(
        "WAN is forbidden in tests (offline). Requested: {}".format(host)
    )


def _guarded_connect(self, address):
    if getattr(self, "family", None) not in (socket.AF_INET, socket.AF_INET6):
        return _REAL_CONNECT(self, address)
    host = address[0] if isinstance(address, (tuple, list)) else address
    if _is_loopback_host(host):
        return _REAL_CONNECT(self, address)
    raise _wan_forbidden(host)


def _guarded_connect_ex(self, address):
    if getattr(self, "family", None) not in (socket.AF_INET, socket.AF_INET6):
        return _REAL_CONNECT_EX(self, address)
    host = address[0] if isinstance(address, (tuple, list)) else address
    if _is_loopback_host(host):
        return _REAL_CONNECT_EX(self, address)
    raise _wan_forbidden(host)


@pytest.fixture(autouse=True)
def _forbid_wan(monkeypatch):
    """Refuse WAN; loopback (serve health checks) stays allowed."""

    def _blocked_urlopen(url, *args, **kwargs):
        if _is_loopback_url(url):
            return _REAL_URLOPEN(url, *args, **kwargs)
        raise _wan_forbidden(_request_url(url))

    monkeypatch.setattr(urllib.request, "urlopen", _blocked_urlopen)
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _guarded_connect_ex)

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
