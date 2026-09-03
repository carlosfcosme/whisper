import importlib.util
import random as rand
import sys
from pathlib import Path

import pytest

_WHISPER_DIR = Path(__file__).resolve().parents[1] / "whisper"


def _load_module(mod_name, filename):
    existing = sys.modules.get(mod_name)
    if existing is not None:
        return existing
    path = _WHISPER_DIR / filename
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def pytest_addoption(parser):
    parser.addoption(
        "--offline-bootstrap",
        action="store_true",
        default=False,
        help="Block WAN/model fetch and require 127.0.0.1 binds (offline bootstrap).",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line(
        "markers", "requires_network: fetches model weights or opens WAN"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--offline-bootstrap"):
        return
    skip = pytest.mark.skip(reason="offline bootstrap blocks network/model fetch")
    for item in items:
        if item.get_closest_marker("requires_network"):
            item.add_marker(skip)


def _load_bind_guard():
    return _load_module("whisper_bind_guard", "bind.py")


def _load_offline_guard():
    return _load_module("whisper_offline_guard", "offline.py")


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


@pytest.fixture(scope="session", autouse=True)
def _offline_bootstrap_session(request):
    if not request.config.getoption("--offline-bootstrap"):
        yield
        return
    offline = _load_offline_guard()
    before = offline.weight_files()
    with offline.offline_guards():
        yield
        offline.assert_only_loopback_listeners()
        offline.assert_no_new_weights(before)
