import os
import random as rand
import urllib.request

import numpy
import pytest

# Deterministic, offline-friendly test defaults. conftest is imported before
# any test module (and thus before torch), so these apply to the whole session.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("WHISPER_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.pop("WHISPER_ALLOW_WEIGHT_DOWNLOAD", None)

_REAL_URLOPEN = urllib.request.urlopen


class DownloadHelperInvoked(RuntimeError):
    """Raised when a test hits a weight/Hub download helper."""


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line(
        "markers",
        "requires_local_weights: needs a cached Whisper checkpoint on disk",
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


def _fail_download_helper(name, url=None, *args, **kwargs):
    target = _request_url(url) if url is not None else ""
    raise DownloadHelperInvoked("download helper invoked: {}({})".format(name, target))


@pytest.fixture(autouse=True)
def _fail_if_download_helpers_invoked(monkeypatch):
    """Tests fail if Hub / weight download helpers are invoked.

    Loopback (whisper serve health) is allowed. Named-model cache misses
    must raise in refuse_remote_download before urlopen.
    """

    def _blocked(url, *args, **kwargs):
        if _is_loopback_url(url):
            return _REAL_URLOPEN(url, *args, **kwargs)
        _fail_download_helper("urlopen", url, *args, **kwargs)

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
            monkeypatch.setattr(
                huggingface_hub,
                name,
                lambda *a, _name=name, **k: _fail_download_helper(_name, *a, **k),
                raising=False,
            )
