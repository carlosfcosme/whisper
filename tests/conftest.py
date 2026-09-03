import os
import random as rand
import urllib.request
from urllib.parse import urlparse

import numpy
import pytest

# Unit tests must not hit the Hugging Face Hub. setdefault so a caller can
# opt back in (e.g. HF_HUB_OFFLINE=0) for an explicit integration run.
# These flags are respected by huggingface_hub / transformers / datasets
# if those packages are present in the environment.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Whisper downloads checkpoints with urllib.request.urlopen
# (whisper/__init__.py). Guard Hub hosts only; Azure / other URLs stay open
# so test_transcribe can still load a cached or official checkpoint.
_HUB_NETLOCS = frozenset(
    {
        "huggingface.co",
        "www.huggingface.co",
        "hf.co",
        "www.hf.co",
    }
)

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
    return _original_urlopen(url, *args, **kwargs)


urllib.request.urlopen = _urlopen_without_hub


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
