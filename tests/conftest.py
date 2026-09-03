import os
import random as rand
import urllib.request
from urllib.parse import urlparse

import numpy
import pytest

# Unit tests must not hit the Hugging Face Hub and must not pull weights.
# setdefault so a caller can opt back in for an explicit integration run.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("WHISPER_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Whisper downloads checkpoints with urllib.request.urlopen
# (whisper/__init__.py). Guard Hub hosts and the official weight CDN.
_HUB_NETLOCS = frozenset(
    {
        "huggingface.co",
        "www.huggingface.co",
        "hf.co",
        "www.hf.co",
    }
)
_WEIGHT_NETLOCS = frozenset({"openaipublic.azureedge.net"})

_original_urlopen = urllib.request.urlopen


def _urlopen_without_hub(url, *args, **kwargs):
    raw = url.full_url if hasattr(url, "full_url") else url
    if not isinstance(raw, str):
        raw = str(raw)
    host = urlparse(raw).netloc.lower().split("@")[-1].split(":")[0]
    if (
        host in _HUB_NETLOCS
        or host.endswith(".huggingface.co")
        or host.endswith(".hf.co")
    ):
        raise RuntimeError(f"unit tests must not contact the Hugging Face Hub ({host})")
    if host in _WEIGHT_NETLOCS or "/whisper/models/" in raw:
        raise RuntimeError(f"tests must not pull model weights ({host})")
    return _original_urlopen(url, *args, **kwargs)


urllib.request.urlopen = _urlopen_without_hub


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line("markers", "requires_weights")


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
