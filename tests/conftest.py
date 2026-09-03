import os
import random as rand

import numpy
import pytest
from hub_offline import install_hub_guards

# Tests never need a GPU and must not contact the Hugging Face Hub.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
install_hub_guards()


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line(
        "markers", "requires_weights: needs a local Whisper checkpoint"
    )


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
