import ipaddress
import os
import random as rand
import socket
import urllib.request
from urllib.parse import urlparse

import numpy
import pytest

# Offline test defaults. setdefault so a caller can opt back in.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("WHISPER_OFFLINE", "1")

_HUB_NETLOCS = frozenset(
    {
        "huggingface.co",
        "www.huggingface.co",
        "hf.co",
        "www.hf.co",
    }
)
_WEIGHT_NETLOCS = frozenset({"openaipublic.azureedge.net"})

_original_urlopen = urllib.request.urlopen
_original_create_connection = socket.create_connection


def _is_loopback_host(host: str) -> bool:
    raw = (host or "").strip().lower()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if raw == "localhost":
        return True
    try:
        return ipaddress.ip_address(raw.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _request_host(url) -> str:
    raw = url.full_url if hasattr(url, "full_url") else url
    if not isinstance(raw, str):
        raw = str(raw)
    return raw, (urlparse(raw).hostname or "")


def _offline_error(raw: str, host: str) -> RuntimeError:
    lowered = host.lower()
    if (
        lowered in _HUB_NETLOCS
        or lowered.endswith(".huggingface.co")
        or lowered.endswith(".hf.co")
    ):
        return RuntimeError(
            f"unit tests must not contact the Hugging Face Hub ({host})"
        )
    if lowered in _WEIGHT_NETLOCS or "/whisper/models/" in raw:
        return RuntimeError(f"tests must not download model weights ({host})")
    return RuntimeError(f"tests must not open network connections ({host})")


def _urlopen_offline(url, *args, **kwargs):
    raw, host = _request_host(url)
    scheme = urlparse(raw).scheme.lower()
    if scheme in {"http", "https"} and not _is_loopback_host(host):
        raise _offline_error(raw, host)
    return _original_urlopen(url, *args, **kwargs)


def _create_connection_offline(address, *args, **kwargs):
    host = address[0] if isinstance(address, (tuple, list)) else address
    if isinstance(host, bytes):
        host = host.decode()
    host = str(host)
    if not _is_loopback_host(host):
        raise _offline_error(host, host)
    return _original_create_connection(address, *args, **kwargs)


urllib.request.urlopen = _urlopen_offline
socket.create_connection = _create_connection_offline


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
