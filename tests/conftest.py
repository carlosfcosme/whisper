import os
import random as rand
import socket

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

# Loopback-only network guard: unit tests must not reach the network (e.g. the
# Hugging Face Hub). We allow connections only to loopback addresses and block
# everything else so any external fetch fails loudly instead of silently
# downloading. Tests that genuinely need the network (e.g. downloading model
# weights) must be marked with @pytest.mark.requires_network.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "0.0.0.0", ""})
_real_socket_connect = socket.socket.connect
_real_socket_connect_ex = socket.socket.connect_ex


def _connect_host(address):
    if isinstance(address, tuple) and address:
        return address[0]
    return address  # AF_UNIX path (str) or similar; not a network address


def _is_loopback(address):
    host = _connect_host(address)
    if not isinstance(host, str):
        return True  # non-inet (e.g. AF_UNIX) is allowed
    return host in _LOOPBACK_HOSTS


def _blocked_network(address):
    host = _connect_host(address)
    raise RuntimeError(
        f"Blocked non-loopback network connect to {host!r} during tests; unit "
        "tests must not reach the network (e.g. the Hugging Face Hub). Mark tests "
        "that genuinely need downloads with @pytest.mark.requires_network."
    )


def _guarded_connect(self, address, *args, **kwargs):
    if _is_loopback(address):
        return _real_socket_connect(self, address, *args, **kwargs)
    _blocked_network(address)


def _guarded_connect_ex(self, address, *args, **kwargs):
    if _is_loopback(address):
        return _real_socket_connect_ex(self, address, *args, **kwargs)
    _blocked_network(address)


@pytest.fixture(autouse=True)
def block_non_loopback_network(request):
    if request.node.get_closest_marker("requires_network"):
        yield
        return
    socket.socket.connect = _guarded_connect
    socket.socket.connect_ex = _guarded_connect_ex
    try:
        yield
    finally:
        socket.socket.connect = _real_socket_connect
        socket.socket.connect_ex = _real_socket_connect_ex


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
