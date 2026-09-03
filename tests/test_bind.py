import pytest

from whisper.bind import (
    LOOPBACK_HOST,
    BindError,
    is_loopback_host,
    require_loopback_host,
)

ALL_INTERFACES = ".".join(("0",) * 4)


@pytest.mark.parametrize("host", [None, "127.0.0.1", "localhost", "LOCALHOST"])
def test_require_loopback_host_allows_loopback(host):
    assert require_loopback_host(host) == LOOPBACK_HOST
    if host is not None:
        assert is_loopback_host(host)


@pytest.mark.parametrize(
    "host",
    [ALL_INTERFACES, "::", "*", "", "   ", "192.168.1.10", "example.com", "10.0.0.1"],
)
def test_require_loopback_host_refuses_non_loopback(host):
    with pytest.raises(BindError, match="127.0.0.1"):
        require_loopback_host(host)
    if host.strip():
        assert not is_loopback_host(host)


def test_all_interfaces_token_is_v4_unspecified():
    assert ALL_INTERFACES == ".".join(("0",) * 4)
    with pytest.raises(BindError):
        require_loopback_host(ALL_INTERFACES)
