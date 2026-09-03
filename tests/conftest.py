import random as rand
from urllib.parse import urlparse

import numpy
import pytest

from whisper.runtime import is_hf_hub_url


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)


@pytest.fixture(autouse=True)
def _block_huggingface_hub(monkeypatch):
    """Unit tests must not contact the Hugging Face Hub."""
    import urllib.request

    real_urlopen = urllib.request.urlopen

    def urlopen(url, *args, **kwargs):
        target = url if isinstance(url, str) else getattr(url, "full_url", str(url))
        if is_hf_hub_url(target):
            host = urlparse(target).hostname or "<unknown>"
            raise RuntimeError(
                f"tests must not contact Hugging Face Hub ({host}): {target}"
            )
        return real_urlopen(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    try:
        import huggingface_hub
    except ImportError:
        return

    def _blocked(*args, **kwargs):
        raise RuntimeError("tests must not call huggingface_hub")

    for name in ("hf_hub_download", "snapshot_download"):
        if hasattr(huggingface_hub, name):
            monkeypatch.setattr(huggingface_hub, name, _blocked)
