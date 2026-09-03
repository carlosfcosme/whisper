import http.client
import importlib
import os
import random as rand
import socket
import urllib.request
from pathlib import Path

import numpy
import pytest
import torch

from whisper.model import ModelDimensions, Whisper
from whisper.offline import (
    is_all_interfaces_host,
    is_hub_host,
    is_loopback_host,
    is_weight_host,
)

# Tests must not download from the Hugging Face Hub (or the public internet).
os.environ["WHISPER_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
# Refuse credentials in the test process (never log values).
for _token_name in (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
):
    os.environ.pop(_token_name, None)

_REAL_URLOPEN = urllib.request.urlopen
_REAL_CONNECT = socket.socket.connect
_REAL_CREATE_CONNECTION = socket.create_connection
_REAL_GETADDRINFO = socket.getaddrinfo
_REAL_BIND = socket.socket.bind

TOY_DIMS = ModelDimensions(
    n_mels=80,
    n_audio_ctx=16,
    n_audio_state=32,
    n_audio_head=4,
    n_audio_layer=1,
    n_vocab=50,
    n_text_ctx=16,
    n_text_state=32,
    n_text_head=4,
    n_text_layer=1,
)


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line(
        "markers",
        "requires_weights: needs a cached Whisper checkpoint on disk",
    )


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)


@pytest.fixture
def toy_dims():
    """Tiny ModelDimensions for a local, generated checkpoint."""
    return TOY_DIMS


@pytest.fixture
def toy_checkpoint(tmp_path):
    """Write a tiny Whisper checkpoint under tmp_path (never committed)."""
    model = Whisper(TOY_DIMS)
    path = tmp_path / "toy.pt"
    torch.save(
        {"dims": TOY_DIMS.__dict__, "model_state_dict": model.state_dict()},
        path,
    )
    return path


@pytest.fixture
def jfk_audio_path():
    """Local JFK fixture already tracked in tests/ (not a weight file)."""
    path = Path(__file__).resolve().parent / "jfk.flac"
    assert path.is_file()
    return path


def _request_url(url):
    if isinstance(url, str):
        return url
    return getattr(url, "full_url", None) or str(url)


def _is_loopback_url(url):
    target = _request_url(url)
    return target.startswith("http://127.0.0.1") or target.startswith(
        "http://localhost"
    )


def _address_host(address):
    if isinstance(address, (tuple, list)) and address:
        return address[0]
    return address


def _refuse_non_loopback(kind, host):
    raise RuntimeError(
        "network intercept: {} to non-loopback host refused: {!r}".format(kind, host)
    )


@pytest.fixture(autouse=True)
def _intercept_network(monkeypatch):
    """Python-level socket intercept (not BPF): loopback only.

    * connect / create_connection: non-loopback refused
    * getaddrinfo: Hub and weight CDN hosts refused (no Hub DNS)
    * bind: all-interfaces / non-loopback refused
    * urlopen / Hub HTTPSConnection: same policy
    """

    def _blocked_urlopen(url, *args, **kwargs):
        if _is_loopback_url(url):
            return _REAL_URLOPEN(url, *args, **kwargs)
        raise RuntimeError(
            "Network / Hub downloads are forbidden in tests (offline). "
            "Requested: {}".format(_request_url(url))
        )

    monkeypatch.setattr(urllib.request, "urlopen", _blocked_urlopen)

    _real_https_init = http.client.HTTPSConnection.__init__

    def _https_init(self, host, *args, **kwargs):
        if is_hub_host(host) or is_weight_host(host):
            raise RuntimeError(
                "Network / Hub downloads are forbidden in tests (offline). "
                "Requested host: {}".format(host)
            )
        return _real_https_init(self, host, *args, **kwargs)

    monkeypatch.setattr(http.client.HTTPSConnection, "__init__", _https_init)

    def _connect(self, address):
        if getattr(self, "family", None) not in (socket.AF_INET, socket.AF_INET6):
            return _REAL_CONNECT(self, address)
        host = _address_host(address)
        if not is_loopback_host(host):
            _refuse_non_loopback("connect", host)
        return _REAL_CONNECT(self, address)

    def _create_connection(address, *args, **kwargs):
        host = _address_host(address)
        if not is_loopback_host(host):
            _refuse_non_loopback("create_connection", host)
        return _REAL_CREATE_CONNECTION(address, *args, **kwargs)

    def _getaddrinfo(host, port, *args, **kwargs):
        if is_hub_host(host) or is_weight_host(host):
            raise RuntimeError(
                "network intercept: Hub/weight DNS refused: {!r}".format(host)
            )
        return _REAL_GETADDRINFO(host, port, *args, **kwargs)

    def _bind(self, address):
        if getattr(self, "family", None) not in (socket.AF_INET, socket.AF_INET6):
            return _REAL_BIND(self, address)
        host = _address_host(address)
        if is_all_interfaces_host(host) or not is_loopback_host(host):
            raise RuntimeError(
                "services must bind 127.0.0.1 only; refused {!r}".format(address)
            )
        return _REAL_BIND(self, address)

    monkeypatch.setattr(socket.socket, "connect", _connect)
    monkeypatch.setattr(socket, "create_connection", _create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", _getaddrinfo)
    monkeypatch.setattr(socket.socket, "bind", _bind)

    try:
        huggingface_hub = importlib.import_module("huggingface_hub")
    except ImportError:
        return

    for name in (
        "hf_hub_download",
        "snapshot_download",
        "hf_hub_url",
        "cached_download",
    ):
        if hasattr(huggingface_hub, name):
            monkeypatch.setattr(huggingface_hub, name, _blocked_urlopen, raising=False)
