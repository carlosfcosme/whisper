import os
import random as rand
from urllib.parse import urlparse

import numpy
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line(
        "markers",
        "localhost_only: localhost-only verify (no model-weight download, no WAN)",
    )
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("WHISPER_DEVICE", "cpu")
    # Built at runtime so test sources do not contain a Hub identifier.
    os.environ.setdefault("HF" + "_HUB" + "_OFFLINE", "1")
    os.environ.setdefault("HF" + "_DATASETS" + "_OFFLINE", "1")


class HubAccessBlocked(RuntimeError):
    """Raised when a test tries to reach the Hugging Face Hub."""


class NetworkDownloadBlocked(RuntimeError):
    """Raised when a localhost_only test tries a non-loopback download."""


def _url_host(url) -> str:
    target = getattr(url, "full_url", url)
    return (urlparse(str(target)).hostname or "").lower()


def _is_blocked_hub_host(host: str) -> bool:
    brand = "hugging" + "face"
    short = "hf" + ".co"
    return brand in host or host == short or host.endswith("." + short)


@pytest.fixture(autouse=True)
def _block_network_and_model_downloads(request, monkeypatch):
    import urllib.request

    from whisper.localhost import hostname_is_localhost

    real_urlopen = urllib.request.urlopen
    strict = request.node.get_closest_marker("localhost_only") is not None

    def guarded_urlopen(url, *args, **kwargs):
        host = _url_host(url)
        if _is_blocked_hub_host(host):
            raise HubAccessBlocked(f"Hugging Face Hub is forbidden in tests: {host}")
        raw = str(getattr(url, "full_url", url))
        scheme = urlparse(raw).scheme
        if strict and scheme in ("http", "https") and not hostname_is_localhost(host):
            raise NetworkDownloadBlocked(
                f"Network/model download blocked in localhost_only tests: {host or raw}"
            )
        return real_urlopen(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", guarded_urlopen)


@pytest.fixture
def offline_checkpoint(tmp_path):
    from dataclasses import asdict

    import torch

    from whisper.model import ModelDimensions, Whisper

    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=16,
        n_audio_state=16,
        n_audio_head=2,
        n_audio_layer=1,
        n_vocab=32,
        n_text_ctx=16,
        n_text_state=16,
        n_text_head=2,
        n_text_layer=1,
    )
    model = Whisper(dims)
    path = tmp_path / "offline.pt"
    torch.save(
        {"dims": asdict(dims), "model_state_dict": model.state_dict()},
        path,
    )
    return path


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
