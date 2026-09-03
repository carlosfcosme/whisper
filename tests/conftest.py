import ipaddress
import os
import random as rand
import socket

import numpy
import pytest
from hub_offline import install_hub_guards

# Hide CUDA so the suite stays on CPU regardless of hardware. setdefault
# so a caller can opt in (e.g. CUDA_VISIBLE_DEVICES=0).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("WHISPER_OFFLINE", "1")
os.environ.setdefault("WHISPER_NO_STORE", "1")
install_hub_guards()

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def _connect_host(address):
    if isinstance(address, (tuple, list)) and address:
        return str(address[0])
    return str(address)


def _allowed_connect_host(host):
    raw = host.split("%", 1)[0]
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if raw.lower() in {"127.0.0.1", "localhost", "::1"}:
        return True
    try:
        return ipaddress.ip_address(raw).is_loopback
    except ValueError:
        return False


def _guarded_connect(self, address):
    if getattr(self, "family", None) == socket.AF_UNIX:
        return _real_connect(self, address)
    host = _connect_host(address)
    if not _allowed_connect_host(host):
        raise RuntimeError(
            "unit tests must not contact non-loopback hosts ({})".format(host)
        )
    return _real_connect(self, address)


def _guarded_connect_ex(self, address):
    if getattr(self, "family", None) == socket.AF_UNIX:
        return _real_connect_ex(self, address)
    host = _connect_host(address)
    if not _allowed_connect_host(host):
        raise RuntimeError(
            "unit tests must not contact non-loopback hosts ({})".format(host)
        )
    return _real_connect_ex(self, address)


socket.socket.connect = _guarded_connect
socket.socket.connect_ex = _guarded_connect_ex


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
