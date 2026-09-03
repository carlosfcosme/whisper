import random as rand

import numpy
import pytest
from hub_offline import install_hub_guards

install_hub_guards()


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
