import random as rand

try:
    import numpy
except ImportError:  # bind-guard CI installs pytest only
    numpy = None

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")


@pytest.fixture
def random():
    rand.seed(42)
    if numpy is not None:
        numpy.random.seed(42)
