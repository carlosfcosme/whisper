import random as rand

import numpy
import pytest

from tests.offline_guard import (
    install_offline_guard,
    isolated_cache_root,
    uninstall_offline_guard,
)


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda: tests that require a CUDA GPU")
    config.addinivalue_line(
        "markers",
        "requires_weights: tests that require local Whisper checkpoints (no download)",
    )
    cache_root = install_offline_guard()
    config._whisper_offline_cache = cache_root


def pytest_unconfigure(config):
    uninstall_offline_guard()


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)


@pytest.fixture
def offline_cache_root():
    root = isolated_cache_root()
    assert root is not None
    return root
