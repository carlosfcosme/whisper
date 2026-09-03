import os
import random as rand
from pathlib import Path

import numpy
import pytest

from whisper.fixtures import assert_local_fixture, write_tiny_wav

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

# Tests never talk to the Hugging Face Hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("WHISPER_OFFLINE", "1")


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line(
        "markers", "requires_weights: tests that load official model checkpoints"
    )


def pytest_collection_modifyitems(config, items):
    flag = os.getenv("WHISPER_OFFLINE", "1").strip().lower()
    if flag not in {"1", "true", "yes", "on"}:
        return
    skip = pytest.mark.skip(
        reason="WHISPER_OFFLINE=1: skip tests that download weights"
    )
    for item in items:
        if "requires_weights" in item.keywords:
            item.add_marker(skip)


_PATH_FIXTURES = frozenset(
    {
        "sample_audio_path",
        "tiny_audio_path",
        "tiktoken_asset_path",
        "mel_filters_path",
    }
)


@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef, request):
    """Fail if a path fixture resolves to a remote URL."""
    outcome = yield
    if fixturedef.argname not in _PATH_FIXTURES:
        return
    try:
        value = outcome.get_result()
    except BaseException:
        return
    assert_local_fixture(value, must_exist=True)


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)


@pytest.fixture
def sample_audio_path():
    """In-repo JFK clip. Never an HTTP(S) or Hub URL."""
    return assert_local_fixture(TESTS_DIR / "jfk.flac")


@pytest.fixture
def tiny_audio_path(tmp_path):
    """Tiny silent WAV written to a tempfile. Never WAN."""
    return write_tiny_wav(tmp_path / "tiny.wav")


@pytest.fixture
def tiktoken_asset_path():
    return assert_local_fixture(
        REPO_ROOT / "whisper" / "assets" / "multilingual.tiktoken"
    )


@pytest.fixture
def mel_filters_path():
    return assert_local_fixture(REPO_ROOT / "whisper" / "assets" / "mel_filters.npz")
