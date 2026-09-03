import os
import random as rand
import urllib.request
from typing import Any

import pytest

# Offline fixtures only. Tests resolve sample audio from the checkout.
# No env var, no download URL, no Hub.
SAMPLE_AUDIO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jfk.flac")

_HUB_MARKERS = (
    "huggingface.co",
    "hf.co",
    "huggingface",
)
_WEIGHT_DOWNLOAD_MARKERS = (
    "openaipublic.azureedge.net",
    "openaipublic",
)


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line(
        "markers",
        "requires_local_weights: skip unless a named checkpoint is already on disk",
    )
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


@pytest.fixture
def sample_audio_path():
    """Absolute path to tests/jfk.flac (in-repo). Never a network URL."""
    return SAMPLE_AUDIO_PATH


@pytest.fixture
def random():
    import numpy

    rand.seed(42)
    numpy.random.seed(42)


def _request_target(url: Any) -> str:
    if isinstance(url, str):
        return url
    full_url = getattr(url, "full_url", None)
    if full_url:
        return str(full_url)
    get_full_url = getattr(url, "get_full_url", None)
    if callable(get_full_url):
        return str(get_full_url())
    return str(url)


@pytest.fixture(autouse=True)
def block_hub_and_weight_downloads(monkeypatch, request):
    allow_cached_weights = (
        request.node.get_closest_marker("requires_local_weights") is not None
    )
    original = urllib.request.urlopen

    def guarded(url, *args, **kwargs):
        target = _request_target(url).lower()
        if any(marker in target for marker in _HUB_MARKERS):
            raise RuntimeError("tests must not contact the Hub: {}".format(target))
        if not allow_cached_weights and any(
            marker in target for marker in _WEIGHT_DOWNLOAD_MARKERS
        ):
            raise RuntimeError("tests must not download weights: {}".format(target))
        return original(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", guarded)
