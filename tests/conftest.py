import os
import random as rand
import sys
import urllib.request
from pathlib import Path

import numpy
import pytest

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

from network_intercept import NetworkIntercepted  # noqa: E402
from network_intercept import is_loopback_peer  # noqa: E402
from network_intercept import install as install_network_intercept  # noqa: E402

# Offline, CPU-only test defaults. setdefault keeps an explicit override.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("WHISPER_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

_REAL_URLOPEN = urllib.request.urlopen

install_network_intercept()


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line(
        "markers",
        "requires_local_weights: needs a cached Whisper checkpoint on disk",
    )
    install_network_intercept()


def pytest_sessionfinish(session, exitstatus):
    from network_intercept import uninstall

    uninstall()


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
    if target.startswith("http://127.0.0.1") or target.startswith("http://localhost"):
        return True
    if "://" in target:
        host = target.split("://", 1)[1].split("/", 1)[0].split(":", 1)[0]
        return is_loopback_peer(host)
    return False


@pytest.fixture(autouse=True)
def _forbid_hub_and_remote_downloads(monkeypatch):
    """Fail the test if Hub or any non-loopback download is attempted."""

    def _blocked(url, *args, **kwargs):
        if _is_loopback_url(url):
            return _REAL_URLOPEN(url, *args, **kwargs)
        raise NetworkIntercepted(
            "Network / Hub downloads are forbidden in tests (offline). "
            "Requested: {}".format(_request_url(url))
        )

    monkeypatch.setattr(urllib.request, "urlopen", _blocked)

    try:
        import huggingface_hub
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
