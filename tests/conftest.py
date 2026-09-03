import os
import random as rand
import urllib.request
from urllib.parse import urlparse

import numpy
import pytest

# Tests must not download from Hugging Face Hub or any remote weight host.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("WHISPER_OFFLINE", "1")

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_real_urlopen = urllib.request.urlopen


def _url_target(url):
    if isinstance(url, str):
        return url
    return getattr(url, "full_url", None) or str(url)


def _is_loopback(url: str) -> bool:
    host = urlparse(url).hostname or ""
    return host in _LOOPBACK_HOSTS


def _offline_urlopen(url, *args, **kwargs):
    target = _url_target(url)
    if _is_loopback(target):
        return _real_urlopen(url, *args, **kwargs)
    raise RuntimeError(f"tests must not Hub or download weights: {target}")


urllib.request.urlopen = _offline_urlopen


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line("markers", "requires_local_weights")
    try:
        import huggingface_hub

        def _no_hub(*_args, **_kwargs):
            raise RuntimeError("tests must not use Hugging Face Hub")

        huggingface_hub.hf_hub_download = _no_hub
        huggingface_hub.snapshot_download = _no_hub
    except ImportError:
        pass


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
