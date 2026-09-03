import importlib.util
import random as rand
from pathlib import Path

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")


def _load_bind_guard():
    path = Path(__file__).resolve().parents[1] / "whisper" / "bind.py"
    spec = importlib.util.spec_from_file_location("whisper_bind_guard", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def random():
    import numpy

    rand.seed(42)
    numpy.random.seed(42)


@pytest.fixture(autouse=True)
def _refuse_non_loopback_listen():
    """Fail the test if this process is left listening off 127.0.0.1."""
    yield
    _load_bind_guard().assert_only_loopback_listeners()
