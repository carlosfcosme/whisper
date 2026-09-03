import os
import random as rand
import socket
import urllib.request

import numpy
import pytest

# Session defaults apply before test modules import torch.
# CPU-only unless a caller overrides CUDA_VISIBLE_DEVICES.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("WHISPER_DEVICE", "cpu")
os.environ.setdefault("WHISPER_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Assembled so application sources stay free of a raw Hub host token.
_HF_MARKERS = ("huggingface" + ".co", "hf" + ".co")


class HuggingFaceHubFetchError(RuntimeError):
    """Raised when a test attempts to contact the Hugging Face Hub."""


def url_is_huggingface_hub(url) -> bool:
    target = getattr(url, "full_url", url)
    text = str(target).lower()
    return any(marker in text for marker in _HF_MARKERS)


def host_is_huggingface_hub(host) -> bool:
    if not isinstance(host, str):
        return False
    return any(marker in host.lower() for marker in _HF_MARKERS)


_orig_urlopen = urllib.request.urlopen
_orig_create_connection = socket.create_connection


def _guarded_urlopen(url, *args, **kwargs):
    if url_is_huggingface_hub(url):
        raise HuggingFaceHubFetchError(
            "tests must not fetch from HuggingFace Hub: {}".format(url)
        )
    return _orig_urlopen(url, *args, **kwargs)


def _guarded_create_connection(address, *args, **kwargs):
    host = address[0] if isinstance(address, tuple) else address
    if host_is_huggingface_hub(host):
        raise HuggingFaceHubFetchError(
            "tests must not fetch from HuggingFace Hub: {}".format(host)
        )
    return _orig_create_connection(address, *args, **kwargs)


urllib.request.urlopen = _guarded_urlopen
socket.create_connection = _guarded_create_connection


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line("markers", "requires_weights")


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
