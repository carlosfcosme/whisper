import http.client
import importlib
import os
import random as rand
import urllib.request

import numpy
import pytest

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


def _request_url(url):
    if isinstance(url, str):
        return url
    return getattr(url, "full_url", None) or str(url)


def _is_loopback_url(url):
    target = _request_url(url)
    return target.startswith("http://127.0.0.1") or target.startswith(
        "http://localhost"
    )


def _is_hub_host(host):
    host = (host or "").lower().split(":")[0]
    return (
        host == "huggingface.co"
        or host.endswith(".huggingface.co")
        or host == "hf.co"
        or host.endswith(".hf.co")
        or host == "hf-mirror.com"
        or host.endswith(".hf-mirror.com")
        or "xethub" in host
    )


@pytest.fixture(autouse=True)
def _forbid_hub_and_remote_downloads(monkeypatch):
    """Block Hub / remote downloads; loopback stays allowed."""

    def _blocked(url, *args, **kwargs):
        if _is_loopback_url(url):
            return _REAL_URLOPEN(url, *args, **kwargs)
        raise RuntimeError(
            "Network / Hub downloads are forbidden in tests (offline). "
            "Requested: {}".format(_request_url(url))
        )

    monkeypatch.setattr(urllib.request, "urlopen", _blocked)

    _real_https_init = http.client.HTTPSConnection.__init__

    def _https_init(self, host, *args, **kwargs):
        if _is_hub_host(host):
            raise RuntimeError(
                "Network / Hub downloads are forbidden in tests (offline). "
                "Requested host: {}".format(host)
            )
        return _real_https_init(self, host, *args, **kwargs)

    monkeypatch.setattr(http.client.HTTPSConnection, "__init__", _https_init)

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
            monkeypatch.setattr(huggingface_hub, name, _blocked, raising=False)
