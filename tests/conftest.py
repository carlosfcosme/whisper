import os
import random as rand
import sys
import urllib.request

import numpy
import pytest

from whisper.runtime import is_hub_url, is_weight_url

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("WHISPER_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line(
        "markers", "requires_weights: test may load named Whisper checkpoints"
    )


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)


class _BlockHubImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "huggingface_hub" or fullname.startswith("huggingface_hub."):
            raise RuntimeError("Hugging Face Hub import/fetch is blocked in tests")
        return None


@pytest.fixture(autouse=True)
def refuse_hub_imports():
    finder = _BlockHubImport()
    sys.meta_path.insert(0, finder)
    yield
    try:
        sys.meta_path.remove(finder)
    except ValueError:
        pass


@pytest.fixture(autouse=True)
def refuse_hub_and_weight_urlopen(monkeypatch):
    """Fail the test if it tries to fetch Hub or model-weight URLs."""
    original = urllib.request.urlopen

    def guarded_urlopen(url, *args, **kwargs):
        if is_hub_url(url) or is_weight_url(url):
            raise RuntimeError(
                "Hub/weight WAN download is blocked in tests: {}".format(url)
            )
        return original(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", guarded_urlopen)
