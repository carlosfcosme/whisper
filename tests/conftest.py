import os
import random as rand

import numpy
import pytest

# Bundled sample audio. Tests resolve this from the checkout — there is no
# env var, no download URL, and no WAN. See tests/README.md.
SAMPLE_AUDIO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jfk.flac")


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")


@pytest.fixture
def sample_audio_path():
    """Absolute path to tests/jfk.flac (in-repo). Never a network URL."""
    return SAMPLE_AUDIO_PATH


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
