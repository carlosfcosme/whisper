import os
import random as rand

import numpy
import pytest
from hub_offline import install_hub_guards

# Hide CUDA so the suite stays on CPU regardless of hardware. setdefault
# so a caller can opt in (e.g. CUDA_VISIBLE_DEVICES=0).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("WHISPER_OFFLINE", "1")
os.environ.setdefault("WHISPER_NO_STORE", "1")
install_hub_guards()


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
