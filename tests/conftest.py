import os
import random as rand
import sys

import numpy
import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from offline_guard import (  # noqa: E402
    install_network_guard,
    uninstall_network_guard,
)


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda: test needs a CUDA GPU")
    config.addinivalue_line(
        "markers", "requires_weights: test needs a local model checkpoint"
    )
    config.addinivalue_line(
        "markers", "allows_network: test may use non-loopback network"
    )


def pytest_sessionstart(session):
    install_network_guard()


def pytest_sessionfinish(session, exitstatus):
    uninstall_network_guard()


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
