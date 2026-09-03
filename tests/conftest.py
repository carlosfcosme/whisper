import os
import random as rand
import urllib.request

import numpy
import pytest

# CPU-only default for the test process. CUDA_VISIBLE_DEVICES=0 still wins.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# Tests must not hit the Hugging Face Hub or pull checkpoints.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

_REAL_URLOPEN = urllib.request.urlopen


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
        "http://localhost"
    )


@pytest.fixture(autouse=True)
def _forbid_hub_and_weight_downloads(monkeypatch):
    """Block Hub and weight-CDN downloads. Loopback stays allowed."""

    def _blocked(url, *args, **kwargs):
        if _is_loopback_url(url):
            return _REAL_URLOPEN(url, *args, **kwargs)
        raise RuntimeError(
            "Hub / weight downloads are forbidden in tests. Requested: {}".format(
                _request_url(url)
            )
        )

    monkeypatch.setattr(urllib.request, "urlopen", _blocked)

    try:
        import huggingface_hub
    except ImportError:
        return

    def _hub_blocked(*args, **kwargs):
        raise RuntimeError("huggingface_hub downloads are forbidden in tests")

    for name in (
        "hf_hub_download",
        "snapshot_download",
        "hf_hub_url",
        "cached_download",
    ):
        if hasattr(huggingface_hub, name):
            monkeypatch.setattr(huggingface_hub, name, _hub_blocked, raising=False)
