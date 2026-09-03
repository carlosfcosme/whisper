import random as rand

import numpy
import pytest

from whisper.local_fixtures import resolve


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)


@pytest.fixture
def sample_audio_path():
    return str(resolve("jfk.flac"))


@pytest.fixture
def tiny_wav_path():
    return str(resolve("tone.wav"))


@pytest.fixture
def tiny_pcm_path():
    return str(resolve("pcm16le.raw"))
