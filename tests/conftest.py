import os
import random as rand
import urllib.request

import numpy
import pytest

OFFLINE_ENV_DEFAULTS = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "WHISPER_OFFLINE": "1",
    "WHISPER_NO_WEIGHT_DOWNLOAD": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line("markers", "requires_weights")
    config.addinivalue_line("markers", "allows_download")
    for key, value in OFFLINE_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)


@pytest.fixture(autouse=True)
def _fail_if_download_helpers_called(request, monkeypatch):
    """Fail the test if a network download helper is invoked.

    The default load path must stay offline. Tests marked ``allows_download``
    opt out of this guard (they still must not hit a real network).
    """
    if request.node.get_closest_marker("allows_download"):
        yield
        return

    calls = []

    def blocked(name):
        def inner(*args, **kwargs):
            target = args[0] if args else kwargs
            calls.append((name, target))
            raise AssertionError(f"download helper {name} was called: {target!r}")

        return inner

    monkeypatch.setattr(urllib.request, "urlopen", blocked("urllib.request.urlopen"))

    try:
        import huggingface_hub
    except ImportError:
        huggingface_hub = None
    if huggingface_hub is not None:
        for name in ("hf_hub_download", "snapshot_download", "hf_hub_url"):
            if hasattr(huggingface_hub, name):
                monkeypatch.setattr(huggingface_hub, name, blocked(name))

    yield
    if calls:
        pytest.fail(f"download helpers were called: {calls}")
