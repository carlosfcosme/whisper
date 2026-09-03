import os
import random as rand
import urllib.request
from urllib.parse import urlparse

import numpy
import pytest

# Negative wildcard fixtures (all-interface / unspecified bind hosts).
WILDCARD_BIND_HOSTS = (
    ".".join(("0",) * 4),
    "::",
    "*",
    "",
    "[::]",
    "0",
    "::0",
)

# Negative network fixtures (LAN, public, non-canonical loopback).
NON_LOOPBACK_HOSTS = (
    "8.8.8.8",
    "10.0.0.1",
    "192.168.1.10",
    "example.com",
    "::1",
    "127.0.0.2",
)

# Negative network URL fixtures (Hub, WAN, non-loopback HTTP).
FORBIDDEN_NETWORK_URLS = (
    "https://%s/openai/whisper-tiny" % ("hugging" + "face.co"),
    "https://%s/openai/whisper-tiny" % ("hf." + "co"),
    "hf://openai/whisper-tiny",
    "http://example.com/tiny.pt",
    "https://example.org/model.safetensors",
)

_OFFLINE_ENV = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "HF_HUB_DISABLE_TELEMETRY",
)


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line(
        "markers", "requires_weights: needs a local Whisper checkpoint"
    )
    for name in _OFFLINE_ENV:
        os.environ.setdefault(name, "1")


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)


@pytest.fixture(params=list(WILDCARD_BIND_HOSTS))
def wildcard_bind_host(request):
    """All-interface / wildcard host that must never be bound."""
    return request.param


@pytest.fixture(params=list(NON_LOOPBACK_HOSTS))
def non_loopback_host(request):
    """LAN / public / non-canonical host that must never be bound."""
    return request.param


@pytest.fixture(params=list(FORBIDDEN_NETWORK_URLS))
def forbidden_network_url(request):
    """Hub or WAN URL that must never be fetched."""
    return request.param


@pytest.fixture(autouse=True)
def block_non_loopback_urlopen(monkeypatch):
    """Runtime tests may talk to 127.0.0.1 only. Everything else is a download."""
    real = urllib.request.urlopen

    def _guarded(url, *args, **kwargs):
        raw = url if isinstance(url, str) else getattr(url, "full_url", str(url))
        host = (urlparse(str(raw)).hostname or "").lower()
        if host in {"127.0.0.1", "localhost"}:
            return real(url, *args, **kwargs)
        raise RuntimeError("tests must not download from the network: %s" % raw)

    monkeypatch.setattr(urllib.request, "urlopen", _guarded)
