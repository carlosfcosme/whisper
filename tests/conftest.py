import importlib.util
import os
import random as rand
import urllib.request
from typing import Any

import pytest

# Load whisper/fixtures.py without importing the whisper package (torch).
_FIXTURES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "whisper",
    "fixtures.py",
)
_spec = importlib.util.spec_from_file_location("whisper_fixtures", _FIXTURES_PATH)
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)
assert_local_fixture = _fixtures.assert_local_fixture
is_remote_fixture_url = _fixtures.is_remote_fixture_url
is_weight_download_url = _fixtures.is_weight_download_url
write_tiny_wav = _fixtures.write_tiny_wav

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("WHISPER_OFFLINE", "1")

# In-repo fixtures. Paths are local files, never a network URL.
SAMPLE_AUDIO_PATH = assert_local_fixture(os.path.join(TESTS_DIR, "jfk.flac"))
TINY_AUDIO_PATH = assert_local_fixture(os.path.join(TESTS_DIR, "tiny.wav"))

_PATH_FIXTURES = frozenset(
    {
        "sample_audio_path",
        "tiny_audio_path",
        "tiktoken_asset_path",
        "mel_filters_path",
    }
)


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line(
        "markers",
        "requires_local_weights: skip unless a named checkpoint is already on disk",
    )


def pytest_collection_modifyitems(config, items):
    flag = os.getenv("WHISPER_OFFLINE", "1").strip().lower()
    if flag not in {"1", "true", "yes", "on"}:
        return
    skip = pytest.mark.skip(
        reason="WHISPER_OFFLINE=1: skip tests that download weights"
    )
    for item in items:
        if "requires_local_weights" in item.keywords:
            item.add_marker(skip)


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
def sample_audio_path():
    """Absolute path to tests/jfk.flac (in-repo). Never a network URL."""
    return SAMPLE_AUDIO_PATH


@pytest.fixture
def tiny_audio_path():
    """Absolute path to tests/tiny.wav (in-repo). Never a network URL."""
    return TINY_AUDIO_PATH


@pytest.fixture
def tiktoken_asset_path():
    return assert_local_fixture(
        os.path.join(REPO_ROOT, "whisper", "assets", "multilingual.tiktoken")
    )


@pytest.fixture
def mel_filters_path():
    return assert_local_fixture(
        os.path.join(REPO_ROOT, "whisper", "assets", "mel_filters.npz")
    )


@pytest.fixture
def random():
    import numpy

    rand.seed(42)
    numpy.random.seed(42)


def _request_target(url: Any) -> str:
    if isinstance(url, str):
        return url
    full_url = getattr(url, "full_url", None)
    if full_url:
        return str(full_url)
    get_full_url = getattr(url, "get_full_url", None)
    if callable(get_full_url):
        return str(get_full_url())
    return str(url)


@pytest.fixture(autouse=True)
def block_hub_and_weight_downloads(monkeypatch, request):
    allow_cached_weights = (
        request.node.get_closest_marker("requires_local_weights") is not None
    )
    original = urllib.request.urlopen

    def guarded(url, *args, **kwargs):
        target = _request_target(url)
        if is_remote_fixture_url(target) and "127.0.0.1" not in target:
            raise RuntimeError("tests must not contact the Hub: {}".format(target))
        if not allow_cached_weights and is_weight_download_url(target):
            raise RuntimeError("tests must not download weights: {}".format(target))
        return original(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", guarded)
