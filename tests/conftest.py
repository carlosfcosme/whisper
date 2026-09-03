import os
import random as rand

import numpy
import pytest

from whisper.offline import require_local_path

# Bundled sample audio. Local checkout file — no Hub, no download URL.
SAMPLE_AUDIO_PATH = require_local_path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "jfk.flac")
)


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")


@pytest.fixture
def sample_audio_path():
    """Absolute path to tests/jfk.flac (in-repo). Never a Hub or http(s) URL."""
    return SAMPLE_AUDIO_PATH


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
