import os
import random as rand

import numpy
import pytest

# Deterministic, offline-friendly test defaults. conftest is imported before any
# test module (and thus before torch), so setting these here applies to the whole
# session. All use setdefault so a caller can override them (e.g.
# CUDA_VISIBLE_DEVICES=0 to run on a GPU).
#
# CPU-only default: hide CUDA devices so tests run on CPU regardless of hardware.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# Unit tests must not hit the Hugging Face Hub: force HF libraries offline if present.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
