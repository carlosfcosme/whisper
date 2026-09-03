import random as rand

import numpy
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line(
        "markers",
        "localhost_only: localhost-only verify (no model-weight download, no WAN)",
    )


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
