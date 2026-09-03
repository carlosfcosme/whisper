import os
import random as rand
import urllib.request

import numpy
import pytest

# Tests must not talk to the Hugging Face Hub.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

_REAL_URLOPEN = urllib.request.urlopen
_HUB_MARKERS = (
    "huggingface.co",
    "hf.co",
    "huggingface_hub",
)


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)


def _request_url(url):
    if isinstance(url, str):
        return url
    return getattr(url, "full_url", None) or str(url)


def _is_hub_url(url):
    target = _request_url(url).lower()
    return any(marker in target for marker in _HUB_MARKERS)


@pytest.fixture(autouse=True)
def _forbid_huggingface_hub(monkeypatch):
    """Block Hugging Face Hub downloads. Loopback and non-Hub URLs stay open."""

    def _urlopen(url, *args, **kwargs):
        if _is_hub_url(url):
            raise RuntimeError(
                "Hugging Face Hub downloads are forbidden in tests. "
                "Requested: {}".format(_request_url(url))
            )
        return _REAL_URLOPEN(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)

    try:
        import huggingface_hub
    except ImportError:
        return

    def _blocked(*args, **kwargs):
        raise RuntimeError("huggingface_hub is forbidden in tests")

    for name in (
        "hf_hub_download",
        "snapshot_download",
        "hf_hub_url",
        "cached_download",
    ):
        if hasattr(huggingface_hub, name):
            monkeypatch.setattr(huggingface_hub, name, _blocked, raising=False)
