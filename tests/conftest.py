import os
import random as rand
import urllib.request

import numpy
import pytest

# Deterministic, offline-friendly defaults. conftest is imported before test
# modules (and thus before most Hub clients), so these apply to the session.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("WHISPER_CPU_ONLY", "1")
os.environ.setdefault("WHISPER_NO_WEIGHT_DOWNLOAD", "1")
os.environ.setdefault("WHISPER_LOCALHOST_ONLY", "1")
# Tests must not hit the Hugging Face Hub.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

_REAL_URLOPEN = urllib.request.urlopen


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
def _bind_127_0_0_1_only(monkeypatch):
    """Unit tests must not bind a wildcard or public interface."""
    import socket

    original_bind = socket.socket.bind

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) and address else address
        if host in ("", "0.0.0.0", "::", "::0"):
            raise OSError("unit tests must bind 127.0.0.1, not a wildcard")
        return original_bind(self, address)

    monkeypatch.setattr(socket.socket, "bind", guarded)


@pytest.fixture(autouse=True)
def _forbid_hub_and_remote_downloads(monkeypatch):
    """Block Hub / remote weight downloads; loopback stays allowed."""

    def _blocked(url, *args, **kwargs):
        if _is_loopback_url(url):
            return _REAL_URLOPEN(url, *args, **kwargs)
        raise RuntimeError(
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
