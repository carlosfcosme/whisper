import os
import random as rand
import urllib.request

import numpy
import pytest

# Deterministic, offline-friendly test defaults. conftest is imported before
# any test module (and thus before torch), so these apply to the whole session.
# setdefault keeps an explicit caller override (e.g. CUDA_VISIBLE_DEVICES=0).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("WHISPER_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

_REAL_URLOPEN = urllib.request.urlopen


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


def _is_hub_url(url):
    target = _request_url(url).lower()
    return any(
        marker in target
        for marker in (
            "huggingface.co",
            "hf.co",
            "huggingface.com",
            "cdn-lfs.huggingface.co",
        )
    )


@pytest.fixture(autouse=True)
def _forbid_hub_and_remote_downloads(monkeypatch):
    """Block Hub / remote downloads; loopback (serve) stays allowed.

    Any Hub fetch in tests must fail. Weight pulls over the network must
    also fail so CI never downloads checkpoints.
    """

    def _blocked(url, *args, **kwargs):
        if _is_loopback_url(url):
            return _REAL_URLOPEN(url, *args, **kwargs)
        kind = "Hub" if _is_hub_url(url) else "Network"
        raise RuntimeError(
            "{} / Hub downloads are forbidden in tests (offline). "
            "Requested: {}".format(kind, _request_url(url))
        )

    monkeypatch.setattr(urllib.request, "urlopen", _blocked)

    try:
        import huggingface_hub
    except ImportError:
        return

    def _hub_blocked(*args, **kwargs):
        raise RuntimeError(
            "Hub fetch is forbidden in tests (offline). "
            "args={!r} kwargs={!r}".format(args, kwargs)
        )

    for name in (
        "hf_hub_download",
        "snapshot_download",
        "hf_hub_url",
        "cached_download",
        "hf_hub_download_to_file",
    ):
        if hasattr(huggingface_hub, name):
            monkeypatch.setattr(huggingface_hub, name, _hub_blocked, raising=False)
