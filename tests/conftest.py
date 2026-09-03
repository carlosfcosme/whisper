import random as rand
import sys
import urllib.request

import numpy
import pytest

_HF_HOSTS = ("huggingface.co", "hf.co")


class _NoHubFinder:
    """Refuse huggingface_hub imports so tests never talk to the Hub."""

    def find_spec(self, fullname, path, target=None):
        if fullname == "huggingface_hub" or fullname.startswith("huggingface_hub."):
            raise ImportError("huggingface_hub is disabled in this test suite")
        return None


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    if not any(isinstance(f, _NoHubFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _NoHubFinder())

    real_urlopen = urllib.request.urlopen

    def _guarded_urlopen(url, *args, **kwargs):
        target = url if isinstance(url, str) else getattr(url, "full_url", str(url))
        lowered = str(target).lower()
        if any(host in lowered for host in _HF_HOSTS):
            raise RuntimeError("Hugging Face Hub is disabled in tests: %s" % target)
        return real_urlopen(url, *args, **kwargs)

    urllib.request.urlopen = _guarded_urlopen


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
