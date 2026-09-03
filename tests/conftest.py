import random as rand

import numpy
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line(
        "markers",
        "localhost_only: bind 127.0.0.1 / no-Hub checks (no model-weight download)",
    )


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
