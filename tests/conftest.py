import random as rand

import numpy
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda: test needs a CUDA GPU")
    config.addinivalue_line(
        "markers", "requires_weights: test needs a local model checkpoint"
    )


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
