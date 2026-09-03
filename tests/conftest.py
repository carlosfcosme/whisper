import random as rand
import socket
import urllib.request

import numpy
import pytest


class NetworkDisabledError(RuntimeError):
    """Raised when a test attempts IP/WAN access."""


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda: tests that require a CUDA GPU")
    config.addinivalue_line(
        "markers",
        "requires_weights: tests that require local Whisper checkpoints (no download)",
    )


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)


@pytest.fixture(autouse=True)
def refuse_network(monkeypatch):
    """Fail any IPv4/IPv6 or urllib WAN attempt. Unix sockets stay available."""

    real_socket = socket.socket

    def denied(*args, **kwargs):
        raise NetworkDisabledError(
            "Network access is disabled in tests (offline CPU suite). "
            "Do not download weights or open WAN sockets."
        )

    class DeniedIPSocket(real_socket):
        def __init__(self, *args, **kwargs):
            family = args[0] if args else kwargs.get("family", socket.AF_INET)
            if family in (socket.AF_INET, socket.AF_INET6):
                denied()
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(socket, "socket", DeniedIPSocket)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(urllib.request, "urlopen", denied)
