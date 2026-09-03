import ipaddress
import os
import random as rand
import urllib.parse
import urllib.request

import numpy
import pytest

# Tests must not hit Hugging Face Hub or pull checkpoints.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("WHISPER_OFFLINE", "1")

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _host_of(url) -> str:
    if not isinstance(url, str):
        url = getattr(url, "full_url", None) or (
            url.get_full_url() if hasattr(url, "get_full_url") else str(url)
        )
    return (urllib.parse.urlparse(str(url)).hostname or "").lower()


def _is_loopback_host(host: str) -> bool:
    if not host or host in _LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


_ORIG_URLOPEN = urllib.request.urlopen


def _guarded_urlopen(url, *args, **kwargs):
    host = _host_of(url)
    if host and not _is_loopback_host(host):
        raise RuntimeError(
            "tests must not download Hub or model weights ({}); "
            "network download prohibited".format(host)
        )
    return _ORIG_URLOPEN(url, *args, **kwargs)


urllib.request.urlopen = _guarded_urlopen

try:
    import huggingface_hub

    def _blocked_hub(*_args, **_kwargs):
        raise RuntimeError("tests must not use Hugging Face Hub")

    huggingface_hub.hf_hub_download = _blocked_hub
    if hasattr(huggingface_hub, "snapshot_download"):
        huggingface_hub.snapshot_download = _blocked_hub
except ImportError:
    pass


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
