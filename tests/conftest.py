import os
import random as rand
from urllib.parse import urlparse

import numpy
import pytest

# CPU default and No Hub for the whole session (before test modules import torch).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("WHISPER_ALLOW_WEIGHT_FETCH", "0")


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line(
        "markers", "requires_weights: needs a local checkpoint (tests must not Hub)"
    )


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)


@pytest.fixture(autouse=True)
def _block_hub_and_wan(monkeypatch):
    """Tests must not contact Hugging Face Hub or other remote hosts."""
    import urllib.request

    real_urlopen = urllib.request.urlopen

    def guarded(url, *args, **kwargs):
        target = url if isinstance(url, str) else getattr(url, "full_url", str(url))
        parsed = urlparse(target)
        host = (parsed.hostname or "").lower().rstrip(".")
        if parsed.scheme == "file" or host in {"127.0.0.1", "localhost", "::1"}:
            return real_urlopen(url, *args, **kwargs)
        raise RuntimeError(f"tests must not Hub or WAN: {target}")

    monkeypatch.setattr(urllib.request, "urlopen", guarded)
