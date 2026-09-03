import os
import random as rand

import numpy
import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("WHISPER_NO_DOWNLOAD", "1")

_BLOCKED_DOWNLOAD_MARKERS = (
    "huggingface.co",
    "hf.co",
    "azureedge.net",
    "openaipublic",
)


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")


@pytest.fixture(autouse=True)
def _block_hub_urlopen(monkeypatch):
    import urllib.request

    real_urlopen = urllib.request.urlopen

    def guarded(url, *args, **kwargs):
        target = url if isinstance(url, str) else getattr(url, "full_url", str(url))
        if any(marker in target for marker in _BLOCKED_DOWNLOAD_MARKERS):
            raise RuntimeError(f"Model downloads are disabled in tests: {target}")
        return real_urlopen(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", guarded)


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
