import os
import random as rand
import urllib.request

import numpy
import pytest

import whisper
from whisper.offline import banned_download, offline_enabled, require_local_path

# Bundled sample audio. Local checkout file — no Hub, no download URL.
SAMPLE_AUDIO_PATH = require_local_path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "jfk.flac")
)


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line("markers", "requires_download")


def pytest_collection_modifyitems(config, items):
    if not offline_enabled():
        return
    skip = pytest.mark.skip(reason="offline by default: no weight download")
    for item in items:
        if item.get_closest_marker("requires_download"):
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def fail_if_download_helpers_called(monkeypatch):
    """Tests fail if urlopen / Hub download helpers are invoked."""
    monkeypatch.setattr(urllib.request, "urlopen", banned_download)
    monkeypatch.setattr(urllib.request, "urlretrieve", banned_download)
    monkeypatch.setattr(whisper.urllib.request, "urlopen", banned_download)
    try:
        import huggingface_hub as hub
    except ImportError:
        return
    for name in ("hf_hub_download", "snapshot_download"):
        if hasattr(hub, name):
            monkeypatch.setattr(hub, name, banned_download)


@pytest.fixture
def sample_audio_path():
    """Absolute path to tests/jfk.flac (in-repo). Never a Hub or http(s) URL."""
    return SAMPLE_AUDIO_PATH


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
