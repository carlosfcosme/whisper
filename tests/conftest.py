import random as rand
import socket
import urllib.request

import numpy
import pytest
import torch

from whisper.offline import apply_offline_env


class NetworkDisabledError(RuntimeError):
    """Raised when a test attempts IP/WAN or Hub access."""


def pytest_configure(config):
    apply_offline_env()
    config.addinivalue_line("markers", "requires_cuda: tests that require a CUDA GPU")
    config.addinivalue_line(
        "markers",
        "requires_weights: tests that require local Whisper checkpoints (no download)",
    )


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)


def _denied(*args, **kwargs):
    raise NetworkDisabledError(
        "Network access is disabled in tests (offline CPU suite). "
        "Do not download weights, Hugging Face Hub files, or open WAN sockets."
    )


@pytest.fixture(autouse=True)
def refuse_network(monkeypatch):
    """Fail any IPv4/IPv6, urllib, Hugging Face Hub, or torch.hub fetch."""

    real_socket = socket.socket

    class DeniedIPSocket(real_socket):
        def __init__(self, *args, **kwargs):
            family = args[0] if args else kwargs.get("family", socket.AF_INET)
            if family in (socket.AF_INET, socket.AF_INET6):
                _denied()
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(socket, "socket", DeniedIPSocket)
    monkeypatch.setattr(socket, "create_connection", _denied)
    monkeypatch.setattr(urllib.request, "urlopen", _denied)

    try:
        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _denied, raising=False)
        monkeypatch.setattr(
            huggingface_hub, "snapshot_download", _denied, raising=False
        )
    except ImportError:
        pass

    if hasattr(torch, "hub"):
        monkeypatch.setattr(torch.hub, "load", _denied, raising=False)
        if hasattr(torch.hub, "load_state_dict_from_url"):
            monkeypatch.setattr(
                torch.hub, "load_state_dict_from_url", _denied, raising=False
            )
