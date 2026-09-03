import socket

import pytest

from whisper.bind import (
    LOOPBACK_HOST,
    UNSPECIFIED_V4,
    BindError,
    assert_own_listens_loopback_only,
    bind_tcp,
    is_loopback_host,
    non_loopback_listens,
    require_loopback_host,
)


@pytest.mark.parametrize("host", [None, "127.0.0.1", "localhost", "LOCALHOST"])
def test_require_loopback_host_allows_loopback(host):
    assert require_loopback_host(host) == LOOPBACK_HOST
    if host is not None:
        assert is_loopback_host(host)


@pytest.mark.parametrize(
    "host",
    [
        UNSPECIFIED_V4,
        "::",
        "*",
        "",
        "   ",
        "192.168.1.10",
        "example.com",
        "10.0.0.1",
        "8.8.8.8",
        "::1",
        "127.0.0.2",
    ],
)
def test_require_loopback_host_refuses_non_loopback(host):
    with pytest.raises(BindError, match="127.0.0.1"):
        require_loopback_host(host)
    if host.strip():
        assert not is_loopback_host(host)


def test_all_interfaces_token_is_v4_unspecified():
    assert UNSPECIFIED_V4 == ".".join(("0",) * 4)
    with pytest.raises(BindError):
        require_loopback_host(UNSPECIFIED_V4)


def test_bind_tcp_listens_on_127_0_0_1():
    sock = bind_tcp(host=LOOPBACK_HOST, port=0)
    try:
        host, port = sock.getsockname()[:2]
        assert host == LOOPBACK_HOST
        assert port > 0
        assert_own_listens_loopback_only()
        assert non_loopback_listens() == []
        probe = socket.create_connection((LOOPBACK_HOST, port), timeout=1)
        probe.close()
    finally:
        sock.close()


def test_bind_tcp_refuses_all_interfaces():
    with pytest.raises(BindError):
        bind_tcp(host=UNSPECIFIED_V4, port=0)
    assert non_loopback_listens() == []


def test_wildcard_fixture_never_binds(wildcard_bind_host):
    with pytest.raises(BindError, match="127.0.0.1"):
        require_loopback_host(wildcard_bind_host)
    with pytest.raises(BindError):
        bind_tcp(host=wildcard_bind_host, port=0)
    assert non_loopback_listens() == []


def test_network_host_fixture_never_binds(non_loopback_host):
    with pytest.raises(BindError, match="127.0.0.1"):
        require_loopback_host(non_loopback_host)
    assert not is_loopback_host(non_loopback_host)


def test_assert_own_listens_loopback_only_flags_unspecified():
    with pytest.raises(BindError, match="non-loopback listen"):
        assert_own_listens_loopback_only([(UNSPECIFIED_V4, 80)])
