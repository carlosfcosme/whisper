import random as rand
import socket
import sys
import urllib.request

import numpy
import pytest

from whisper.fixtures import (
    IN_REPO_SAMPLE_AUDIO,
    require_local_fixture,
    tiny_wav_bytes,
    write_tiny_wav,
)

_ALL_INTERFACES = frozenset(("", "0.0.0.0", "::", "[::]"))
_LOOPBACK = frozenset(("127.0.0.1", "::1", "localhost"))
_HUB_MARKERS = (
    "huggingface.co",
    "hf.co/",
    "hf-mirror.com",
    "huggingface_hub",
    "cas-bridge.xethub",
)
_WEIGHT_HOST_MARKERS = (
    "openaipublic.azureedge.net",
    "whisper/models",
)
_WEIGHT_SUFFIXES = (".pt", ".pth", ".safetensors")
_AUDIO_SUFFIXES = (".flac", ".wav", ".mp3", ".ogg", ".m4a")


def _offline_requested():
    import os

    return os.getenv("WHISPER_OFFLINE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _url_text(url):
    if hasattr(url, "full_url"):
        return url.full_url
    return str(url)


def _is_hub_fetch(url):
    lowered = _url_text(url).lower()
    return any(marker in lowered for marker in _HUB_MARKERS)


def _is_weight_fetch(url):
    lowered = _url_text(url).lower()
    if any(lowered.endswith(suffix) for suffix in _WEIGHT_SUFFIXES):
        return True
    return any(marker in lowered for marker in _WEIGHT_HOST_MARKERS)


def _is_remote_audio_fixture(url):
    lowered = _url_text(url).lower()
    if not lowered.startswith(("http://", "https://")):
        return False
    return any(lowered.split("?", 1)[0].endswith(suffix) for suffix in _AUDIO_SUFFIXES)


def _is_forbidden_fetch(url):
    return _is_hub_fetch(url) or _is_weight_fetch(url) or _is_remote_audio_fixture(url)


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line(
        "markers", "requires_weights: test may fetch named Whisper checkpoints"
    )


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)


@pytest.fixture
def sample_audio_path():
    """In-repo tests/jfk.flac. Never a network URL."""
    return require_local_fixture(IN_REPO_SAMPLE_AUDIO)


@pytest.fixture
def tiny_wav_path(tmp_path):
    """Temp 10 ms silence WAV. Never a network URL."""
    return write_tiny_wav(tmp_path / "tiny.wav")


@pytest.fixture
def tiny_audio_bytes():
    """Tiny WAV bytes generated in process. No Hub, no keys."""
    payload = tiny_wav_bytes()
    assert payload[:4] == b"RIFF"
    return payload


@pytest.fixture(autouse=True)
def refuse_weight_downloads(request, monkeypatch):
    """Fail if a test tries to WAN-pull model weights.

    ``requires_weights`` tests may fetch unless WHISPER_OFFLINE is set (CI).
    """
    allowed = request.node.get_closest_marker("requires_weights") is not None
    if allowed and not _offline_requested():
        return

    original = urllib.request.urlopen

    def guarded_urlopen(url, *args, **kwargs):
        if _is_forbidden_fetch(url):
            raise RuntimeError(
                "Hub/weight WAN download is blocked in tests/offline: %s"
                % (_url_text(url),)
            )
        return original(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", guarded_urlopen)


class _BlockHubImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "huggingface_hub" or fullname.startswith("huggingface_hub."):
            raise RuntimeError("Hugging Face Hub import/fetch is blocked in tests")
        return None


@pytest.fixture(autouse=True)
def refuse_hub_imports():
    finder = _BlockHubImport()
    sys.meta_path.insert(0, finder)
    yield
    try:
        sys.meta_path.remove(finder)
    except ValueError:
        pass


@pytest.fixture(autouse=True)
def refuse_non_localhost_binds(monkeypatch):
    """Services in tests must bind 127.0.0.1 / loopback only."""
    original_bind = socket.socket.bind

    def guarded_bind(self, address):
        if getattr(self, "family", None) not in (socket.AF_INET, socket.AF_INET6):
            return original_bind(self, address)
        host = address
        if isinstance(address, (tuple, list)) and address:
            host = address[0]
        if isinstance(host, bytes):
            host = host.decode("utf-8", "replace")
        if isinstance(host, str) and host in _ALL_INTERFACES:
            raise RuntimeError(
                "services must bind 127.0.0.1 only; refused %r" % (address,)
            )
        if isinstance(host, str) and host not in _LOOPBACK:
            raise RuntimeError(
                "services must bind 127.0.0.1 only; refused %r" % (address,)
            )
        return original_bind(self, address)

    monkeypatch.setattr(socket.socket, "bind", guarded_bind)
