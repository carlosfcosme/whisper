import os
import random as rand
import urllib.request

import numpy
import pytest

# Tests must not talk to the Hugging Face Hub or pull checkpoints.
os.environ["WHISPER_OFFLINE"] = "1"
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
_WEIGHT_MARKERS = (
    "openaipublic.azureedge.net",
    ".pt",
    ".pth",
    ".safetensors",
    ".ckpt",
    ".gguf",
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


def _is_loopback_url(url):
    target = _request_url(url)
    return target.startswith("http://127.0.0.1") or target.startswith(
        "https://127.0.0.1"
    )


def _is_hub_url(url):
    target = _request_url(url).lower()
    return any(marker in target for marker in _HUB_MARKERS)


def _is_weight_url(url):
    target = _request_url(url).lower()
    return any(marker in target for marker in _WEIGHT_MARKERS)


@pytest.fixture(autouse=True)
def _forbid_hub_and_weight_pulls(monkeypatch):
    """Block Hub and weight downloads. Loopback (127.0.0.1) stays allowed."""

    def _urlopen(url, *args, **kwargs):
        target = _request_url(url)
        if _is_hub_url(url):
            raise RuntimeError(
                "Hugging Face Hub downloads are forbidden in tests. "
                "Requested: {}".format(target)
            )
        if _is_weight_url(url) or not _is_loopback_url(url):
            raise RuntimeError(
                "Remote weight pulls are forbidden in tests. "
                "Requested: {}".format(target)
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
