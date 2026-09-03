import pytest

from whisper.loopback import (
    LOOPBACK_HOST,
    assert_bound_socket_is_loopback,
    assert_loopback_bind,
    bound_host,
    create_loopback_server,
    is_wildcard_host,
)


def test_assert_loopback_bind_rejects_0_0_0_0():
    with pytest.raises(ValueError, match="wildcard"):
        assert_loopback_bind("0.0.0.0")
    with pytest.raises(ValueError, match="wildcard"):
        assert_loopback_bind("")
    with pytest.raises(ValueError, match="127.0.0.1"):
        assert_loopback_bind("8.8.8.8")
    assert assert_loopback_bind("127.0.0.1") == LOOPBACK_HOST == "127.0.0.1"


def test_assert_bound_socket_is_loopback_rejects_wildcard():
    with pytest.raises(OSError, match="non-loopback"):
        assert_bound_socket_is_loopback("0.0.0.0")


def test_create_loopback_server_never_binds_0_0_0_0():
    with pytest.raises(ValueError):
        create_loopback_server(host="0.0.0.0", port=0)

    server = create_loopback_server(host="127.0.0.1", port=0)
    try:
        host = bound_host(server)
        advertised, _port = server.server_address
        assert host != "0.0.0.0"
        assert advertised != "0.0.0.0"
        assert host == "127.0.0.1"
        assert advertised == "127.0.0.1"
        assert not is_wildcard_host(host)
    finally:
        server.server_close()


def test_server_bind_fails_if_address_forced_to_0_0_0_0():
    server = create_loopback_server(host="127.0.0.1", port=0)
    try:
        server.server_address = ("0.0.0.0", 0)
        with pytest.raises((ValueError, OSError)):
            server.server_bind()
    finally:
        server.server_close()
