import os
import random as rand
import sys
from pathlib import Path

import numpy
import pytest

# conftest loads before test modules (and before torch in those modules).
# CPU default: hide GPUs unless the caller already set CUDA_VISIBLE_DEVICES.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# CPU-only offline default: no WAN weight pull, no Hugging Face Hub.
os.environ.setdefault("WHISPER_DEVICE", "cpu")
os.environ.setdefault("WHISPER_ALLOW_WEIGHT_FETCH", "0")
os.environ.setdefault("WHISPER_BIND_HOST", "127.0.0.1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from hub_guard import (  # noqa: E402
    install_hub_import_block,
    install_offline_network_block,
)

install_hub_import_block()
install_offline_network_block()


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line(
        "markers",
        "environment: bind, no fetch, and gitignore contract",
    )


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
