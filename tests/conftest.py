import random as rand

import pytest

from tests.wan_guard import (
    configure_offline_env,
    install_hub_client_guard,
    install_socket_guard,
    install_urlopen_guard,
)

configure_offline_env()


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line(
        "markers",
        "requires_weights: needs a local checkpoint; default path does not fetch",
    )


@pytest.fixture(autouse=True)
def _no_hub_or_wan(monkeypatch):
    """Normal tests must not contact model hubs or the WAN."""
    configure_offline_env()
    install_hub_client_guard(monkeypatch)
    install_urlopen_guard(monkeypatch)
    install_socket_guard(monkeypatch)


@pytest.fixture
def random():
    import numpy

    rand.seed(42)
    numpy.random.seed(42)
