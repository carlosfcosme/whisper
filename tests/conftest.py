import os
import random as rand

import numpy
import pytest

# Unit tests stay offline: no Hugging Face Hub, CPU default.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("WHISPER_DEVICE", "cpu")


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
