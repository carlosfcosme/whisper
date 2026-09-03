import random as rand
import sys
import urllib.request

import numpy
import pytest

_HF_HOSTS = ("huggingface.co", "hf.co")
_WEIGHT_HOSTS = ("openaipublic.azureedge.net",)
_ALLOW_WEIGHT_DOWNLOAD = False


class _NoHubFinder:
    """Refuse huggingface_hub imports so tests never talk to the Hub."""

    def find_spec(self, fullname, path, target=None):
        if fullname == "huggingface_hub" or fullname.startswith("huggingface_hub."):
            raise ImportError("huggingface_hub is disabled in this test suite")
        return None


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line("markers", "allows_weight_download")
    if not any(isinstance(f, _NoHubFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _NoHubFinder())

    real_urlopen = urllib.request.urlopen

    def _guarded_urlopen(url, *args, **kwargs):
        target = url if isinstance(url, str) else getattr(url, "full_url", str(url))
        lowered = str(target).lower()
        blocked = _HF_HOSTS if _ALLOW_WEIGHT_DOWNLOAD else _HF_HOSTS + _WEIGHT_HOSTS
        if any(host in lowered for host in blocked):
            raise RuntimeError("offline: weight/Hub download disabled: %s" % target)
        return real_urlopen(url, *args, **kwargs)

    urllib.request.urlopen = _guarded_urlopen


@pytest.fixture(autouse=True)
def _offline_weight_flag(request):
    global _ALLOW_WEIGHT_DOWNLOAD
    _ALLOW_WEIGHT_DOWNLOAD = (
        request.node.get_closest_marker("allows_weight_download") is not None
    )
    yield
    _ALLOW_WEIGHT_DOWNLOAD = False


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
