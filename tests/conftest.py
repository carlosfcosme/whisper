import os
import random as rand
import urllib.request

import numpy
import pytest

# Sovereign path: CPU-only, no Hub/weight pull, bind 127.0.0.1.
os.environ.setdefault("WHISPER_CPU_ONLY", "1")
os.environ.setdefault("WHISPER_NO_WEIGHT_DOWNLOAD", "1")
os.environ.setdefault("WHISPER_LOCALHOST_ONLY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line(
        "markers", "requires_weights: test loads official checkpoints"
    )


@pytest.fixture(autouse=True)
def _bind_127_0_0_1_only(monkeypatch):
    """Unit tests fail if a wildcard or public interface is bound."""
    import socket

    original_bind = socket.socket.bind

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) and address else address
        if host in ("", "::", "::0") or host == "0.0.0.0":
            raise OSError("unit tests must bind 127.0.0.1, not a wildcard")
        return original_bind(self, address)

    monkeypatch.setattr(socket.socket, "bind", guarded)


@pytest.fixture(autouse=True)
def _no_hub_or_weight_pull(monkeypatch):
    """Unit tests fail if the Hub is contacted or weights are pulled."""
    import importlib

    from whisper.runtime import is_hf_hub_url

    def _blocked(*args, **kwargs):
        raise RuntimeError("Hugging Face Hub pull is not allowed in unit tests")

    try:
        hub = importlib.import_module("huggingface_hub")
    except ImportError:
        hub = None
    if hub is not None:
        monkeypatch.setattr(hub, "hf_hub_download", _blocked, raising=False)
        if hasattr(hub, "snapshot_download"):
            monkeypatch.setattr(hub, "snapshot_download", _blocked, raising=False)

    original_urlopen = urllib.request.urlopen

    def guarded_urlopen(url, *args, **kwargs):
        target = url if isinstance(url, str) else getattr(url, "full_url", str(url))
        text = str(target)
        if is_hf_hub_url(text):
            raise RuntimeError("Hugging Face Hub pull is not allowed in unit tests")
        host = ""
        try:
            from urllib.parse import urlparse

            host = (urlparse(text).hostname or "").lower()
        except Exception:
            host = text.lower()
        if host.endswith("azureedge.net") or host.endswith("huggingface.co"):
            raise RuntimeError("weight pull is not allowed in unit tests")
        return original_urlopen(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", guarded_urlopen)


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
