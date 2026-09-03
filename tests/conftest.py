import os
import random as rand
import socket

import numpy
import pytest

# Hugging Face Hub is disabled for the entire test process.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

_HUB_HOSTS = (
    "huggingface.co",
    "hf.co",
    "cdn-lfs.huggingface.co",
    "cas-bridge.xethub.hf.co",
)


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def _hub_blocked(host):
    if not isinstance(host, str):
        return False
    lowered = host.lower().rstrip(".")
    return lowered in _HUB_HOSTS or lowered.endswith(".huggingface.co")


@pytest.fixture(autouse=True)
def _block_huggingface_hub(monkeypatch):
    real_create_connection = socket.create_connection
    real_getaddrinfo = socket.getaddrinfo

    def guarded_create_connection(address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if _hub_blocked(host):
            raise RuntimeError(
                "Hugging Face Hub is disabled in tests: {0}".format(host)
            )
        return real_create_connection(address, *args, **kwargs)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if _hub_blocked(host):
            raise socket.gaierror(
                socket.EAI_NONAME, "Hugging Face Hub is disabled in tests"
            )
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
