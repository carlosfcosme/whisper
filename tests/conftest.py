import importlib.util
import os
import random as rand
import sys
import urllib.request
from pathlib import Path

import numpy
import pytest

from whisper.runtime import is_hub_url, is_weight_url

os.environ.setdefault("WHISPER_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

_PATHS_FILE = Path(__file__).resolve().parent / "fixtures" / "paths.py"
_SPEC = importlib.util.spec_from_file_location(
    "whisper_test_fixture_paths", _PATHS_FILE
)
_PATHS = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_PATHS)


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line(
        "markers", "requires_weights: test may load named Whisper checkpoints"
    )


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)


@pytest.fixture
def sample_audio_path():
    """Tiny in-repo WAV under tests/fixtures/. Never a WAN URL."""
    return str(_PATHS.fixture_path("tiny.wav"))


@pytest.fixture
def jfk_audio_path():
    """Existing in-repo FLAC under tests/. Never a WAN URL."""
    return str(_PATHS.repo_audio_path("jfk.flac"))


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
def refuse_weight_downloads(monkeypatch):
    """Fail if a test tries to WAN-pull model weights."""
    original = urllib.request.urlopen

    def guarded_urlopen(url, *args, **kwargs):
        if is_hub_url(url) or is_weight_url(url):
            raise RuntimeError(
                "Hub/weight WAN download is blocked in tests: {}".format(url)
            )
        return original(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", guarded_urlopen)
