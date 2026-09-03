import os

import torch

# Verify the CPU-only / offline "no-store" defaults installed by conftest.py are
# actually in effect for the test session. These are set via os.environ.setdefault
# so they can be overridden (e.g. CUDA_VISIBLE_DEVICES=0), but by default the
# suite must be CPU-only and fully offline.


def test_cpu_only_default():
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    # conftest always sets a default value.
    assert cuda_visible is not None
    # With the default (empty) value, no CUDA device is exposed to torch.
    if cuda_visible == "":
        assert torch.cuda.is_available() is False


def test_offline_no_store_defaults():
    # Offline / no-store: HF libraries (if ever present) must not fetch from or
    # write to a remote hub during tests.
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert os.environ.get("HF_DATASETS_OFFLINE") == "1"
    assert os.environ.get("HF_HUB_DISABLE_TELEMETRY") == "1"
