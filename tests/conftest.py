import os
import random as rand
from urllib.parse import urlparse

import numpy
import pytest

os.environ.setdefault("WHISPER_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line(
        "markers", "requires_weights: needs a local checkpoint (must not Hub)"
    )


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)


@pytest.fixture(autouse=True)
def _block_hub_and_wan(monkeypatch):
    """Fail the test if Hugging Face Hub or another remote host is contacted."""
    import urllib.request

    real_urlopen = urllib.request.urlopen

    def guarded(url, *args, **kwargs):
        target = url if isinstance(url, str) else getattr(url, "full_url", str(url))
        parsed = urlparse(target)
        host = (parsed.hostname or "").lower().rstrip(".")
        if parsed.scheme == "file" or host in {"127.0.0.1", "localhost", "::1"}:
            return real_urlopen(url, *args, **kwargs)
        raise RuntimeError(f"tests must not Hub or pull weights: {target}")

    monkeypatch.setattr(urllib.request, "urlopen", guarded)


@pytest.fixture(scope="session", autouse=True)
def _assert_download_unused_session():
    yield
    from whisper.offline import assert_download_unused

    assert_download_unused("pytest session")
