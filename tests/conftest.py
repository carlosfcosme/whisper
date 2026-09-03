import importlib.util
import random as rand
from pathlib import Path

import numpy
import pytest

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
