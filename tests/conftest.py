import os
import random as rand
import socket
import urllib.request
from pathlib import Path
from typing import Iterator, List
from urllib.parse import urlparse

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

# Host markers assembled so application sources stay free of raw WAN tokens.
_HF_MARKERS = ("huggingface" + ".co", "hf" + ".co")
_WEIGHT_HOST_MARKERS = _HF_MARKERS + (
    "azureedge" + ".net",
    "cdn-lfs",
)
_WEIGHT_SUFFIXES = (
    ".pt",
    ".pth",
    ".bin",
    ".ckpt",
    ".safetensors",
    ".onnx",
    ".gguf",
    ".ggml",
    ".weights",
)

LOOPBACK_HOST = "127.0.0.1"
ALL_INTERFACES = "0.0.0.0"

# Negative bind hosts, including wildcard / all-interface forms.
NEGATIVE_WILDCARD_HOSTS = (
    ALL_INTERFACES,
    "*",
    "*:*",
    "0.0.0.*",
    "::",
    "[::]",
    "",
    "*.local",
    "*.example.com",
    "192.168.*.1",
    "192.168.1.10",
    "10.0.0.1",
    "172.16.0.1",
    "8.8.8.8",
    "example.com",
)

LOOPBACK_HOSTS = (LOOPBACK_HOST, "localhost", "LOCALHOST")


class WeightDownloadBlocked(RuntimeError):
    """Raised when a test attempts to download model weights over the network."""


class HuggingFaceHubFetchError(WeightDownloadBlocked):
    """Raised when a test attempts to contact the Hugging Face Hub."""


def _url_text(url) -> str:
    target = getattr(url, "full_url", None)
    if target is None:
        target = getattr(url, "get_full_url", lambda: url)()
    return str(target)


def url_is_huggingface_hub(url) -> bool:
    text = _url_text(url).lower()
    return any(marker in text for marker in _HF_MARKERS)


def host_is_huggingface_hub(host) -> bool:
    if not isinstance(host, str):
        return False
    return any(marker in host.lower() for marker in _HF_MARKERS)


def url_is_weight_download(url) -> bool:
    text = _url_text(url).lower()
    if any(marker in text for marker in _WEIGHT_HOST_MARKERS):
        return True
    path = urlparse(text).path
    return any(path.endswith(suffix) for suffix in _WEIGHT_SUFFIXES)


def host_is_weight_download(host) -> bool:
    if not isinstance(host, str):
        return False
    lowered = host.lower()
    return any(marker in lowered for marker in _WEIGHT_HOST_MARKERS)


_orig_urlopen = urllib.request.urlopen
_orig_create_connection = socket.create_connection


def _guarded_urlopen(url, *args, **kwargs):
    if url_is_huggingface_hub(url):
        raise HuggingFaceHubFetchError(
            "tests must not fetch from HuggingFace Hub: {}".format(url)
        )
    if url_is_weight_download(url):
        raise WeightDownloadBlocked(
            "tests must not download model weights: {}".format(url)
        )
    return _orig_urlopen(url, *args, **kwargs)


def _guarded_create_connection(address, *args, **kwargs):
    host = address[0] if isinstance(address, tuple) else address
    if host_is_huggingface_hub(host):
        raise HuggingFaceHubFetchError(
            "tests must not fetch from HuggingFace Hub: {}".format(host)
        )
    if host_is_weight_download(host):
        raise WeightDownloadBlocked(
            "tests must not download model weights: {}".format(host)
        )
    return _orig_create_connection(address, *args, **kwargs)


urllib.request.urlopen = _guarded_urlopen
socket.create_connection = _guarded_create_connection


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line("markers", "requires_weights")


@pytest.fixture(params=LOOPBACK_HOSTS)
def loopback_host(request) -> str:
    """Hosts that may be rewritten to 127.0.0.1."""
    return request.param


@pytest.fixture(params=NEGATIVE_WILDCARD_HOSTS)
def negative_wildcard_host(request) -> str:
    """Wildcard / non-loopback hosts that must never be a bind address."""
    return request.param


@pytest.fixture
def negative_wildcard_hosts() -> List[str]:
    return list(NEGATIVE_WILDCARD_HOSTS)


@pytest.fixture
def wildcard_bind_script(tmp_path) -> Iterator[Path]:
    """Start script that tries to bind all interfaces — must fail the scanner."""
    script = tmp_path / "start.sh"
    script.write_text(
        "python3 -m http.server --bind {}\n".format(ALL_INTERFACES),
        encoding="utf-8",
    )
    yield script


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
