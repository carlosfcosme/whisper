import os
import random as rand
from pathlib import Path

import pytest

# CI / Cloud Agent path: CPU-only, no Hub download, bind 127.0.0.1.
os.environ.setdefault("WHISPER_CPU_ONLY", "1")
os.environ.setdefault("WHISPER_NO_WEIGHT_DOWNLOAD", "1")
os.environ.setdefault("WHISPER_LOCALHOST_ONLY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line("markers", "requires_local_weights")


@pytest.fixture(autouse=True)
def _bind_127_0_0_1_only(monkeypatch):
    """Unit tests must not bind a wildcard or public interface."""
    import socket

    original_bind = socket.socket.bind
    all_interfaces = ".".join(("0", "0", "0", "0"))

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) and address else address
        if host in ("", all_interfaces, "::", "::0"):
            raise OSError("unit tests must bind 127.0.0.1, not a wildcard")
        return original_bind(self, address)

    monkeypatch.setattr(socket.socket, "bind", guarded)


@pytest.fixture(autouse=True)
def _no_hf_hub_pull(monkeypatch):
    """Unit tests must not pull weights from the Hugging Face Hub."""

    def _blocked(*args, **kwargs):
        raise RuntimeError("Hugging Face Hub pull is not allowed in unit tests")

    try:
        import huggingface_hub
    except ImportError:
        return
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _blocked, raising=False)
    if hasattr(huggingface_hub, "snapshot_download"):
        monkeypatch.setattr(
            huggingface_hub, "snapshot_download", _blocked, raising=False
        )


@pytest.fixture(autouse=True)
def _fail_any_wan_model_fetch(monkeypatch):
    """Fail any non-loopback http(s) fetch (models, Hub, CDN, fixtures)."""
    import urllib.request
    from urllib.parse import urlparse

    original = urllib.request.urlopen

    def _target(url):
        if isinstance(url, str):
            return url
        full_url = getattr(url, "full_url", None)
        if full_url:
            return str(full_url)
        get_full_url = getattr(url, "get_full_url", None)
        if callable(get_full_url):
            return str(get_full_url())
        return str(url)

    def guarded(url, *args, **kwargs):
        target = _target(url)
        parsed = urlparse(target)
        host = (parsed.hostname or "").lower().rstrip(".")
        if parsed.scheme in {"http", "https", "ftp"} and host not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise RuntimeError(
                "tests must not fetch over WAN (local fixtures + 127.0.0.1 only): "
                + target
            )
        return original(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", guarded)


@pytest.fixture
def sample_audio_path():
    """Absolute path to tests/jfk.flac (in-repo). Never a network URL."""
    from whisper.fixtures import require_local_fixture

    return str(require_local_fixture(Path(__file__).resolve().parent / "jfk.flac"))


@pytest.fixture
def tiny_audio_path():
    """Absolute path to the tiny committed/generated WAV. Local only."""
    from whisper.fixtures import require_local_fixture, tiny_fixture_path

    return str(require_local_fixture(tiny_fixture_path(generate=False)))


@pytest.fixture
def random():
    import numpy

    rand.seed(42)
    numpy.random.seed(42)
