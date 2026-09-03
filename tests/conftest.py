import os
import random as rand
import urllib.request

import numpy
import pytest

# Offline Hub policy for the test suite. Whisper downloads checkpoints via
# urllib; Hugging Face clients honor these env vars if imported.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["WHISPER_NO_HUB"] = "1"
os.environ["WHISPER_OFFLINE"] = "1"

_HUB_HOSTS = (
    "huggingface.co",
    "hf.co",
    "hf-mirror.com",
    "openaipublic.azureedge.net",
    "cdn-lfs.huggingface.co",
)


def _url_is_hub(url) -> bool:
    if isinstance(url, str):
        target = url
    else:
        target = (
            getattr(url, "full_url", None) or getattr(url, "host", None) or str(url)
        )
    target = target.lower()
    return any(host in target for host in _HUB_HOSTS)


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line("markers", "requires_hub")
    config.addinivalue_line("markers", "commercial")
    config.addinivalue_line("markers", "requires_local_weights")


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)


@pytest.fixture(autouse=True)
def _block_hub_downloads(monkeypatch):
    real_urlopen = urllib.request.urlopen

    def guarded_urlopen(url, *args, **kwargs):
        if _url_is_hub(url):
            raise RuntimeError(f"Hub/CDN download blocked in tests: {url}")
        return real_urlopen(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", guarded_urlopen)
