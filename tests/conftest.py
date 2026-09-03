import random as rand

import pytest

from tests.import_stdlib import load_bind_standalone
from tests.wan_guard import (
    ATTEMPTS,
    configure_offline_env,
    install_http_client_guard,
    install_hub_client_guard,
    install_socket_guard,
    install_urlopen_guard,
    unexpected_network_attempts,
)

configure_offline_env()

_bind = load_bind_standalone()


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line(
        "markers",
        "requires_weights: needs a local checkpoint; default path does not fetch",
    )
    config.addinivalue_line(
        "markers",
        "probes_network: may attempt a refused hub/WAN/weight fetch on purpose",
    )
    config.addinivalue_line(
        "markers",
        "probes_bind: may attempt a refused non-loopback bind on purpose",
    )


def pytest_runtest_setup(item):
    item._whisper_attempt_mark = len(ATTEMPTS)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Fail the test if it attempted a network/weight download or bad bind."""
    outcome = yield
    report = outcome.get_result()
    if call.when != "call" or report.failed:
        return
    start = getattr(item, "_whisper_attempt_mark", 0)
    new = ATTEMPTS[start:]
    markers = [mark.name for mark in item.iter_markers()]
    unexpected = unexpected_network_attempts(markers, new)
    if unexpected:
        report.outcome = "failed"
        report.longrepr = (
            "test attempted network/model-weight download or non-loopback "
            "connect (fail-on-attempt):\n  - " + "\n  - ".join(unexpected)
        )


@pytest.fixture(autouse=True)
def _no_hub_or_wan(monkeypatch):
    """Normal tests must not contact model hubs or the WAN."""
    configure_offline_env()
    install_hub_client_guard(monkeypatch)
    install_urlopen_guard(monkeypatch)
    install_socket_guard(monkeypatch)
    install_http_client_guard(monkeypatch)
    _bind.install_bind_guard()
    try:
        yield
    finally:
        _bind.uninstall_bind_guard()


@pytest.fixture
def random():
    import numpy

    rand.seed(42)
    numpy.random.seed(42)
