import os

import pytest

from whisper.loopback import (
    LOOPBACK_HOST,
    assert_bound_socket_is_loopback,
    assert_loopback_bind,
    is_wildcard_host,
)
from whisper.serve import ServeConfig, create_server


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
    assert assert_bound_socket_is_loopback("127.0.0.1") == "127.0.0.1"


def test_create_server_never_binds_0_0_0_0():
    with pytest.raises(ValueError):
        create_server(ServeConfig(host="0.0.0.0", port=0))

    server = create_server(host="127.0.0.1", port=0, device="cpu")
    try:
        host, _port = server.server_address
        sock_host = server.socket.getsockname()[0]
        assert host != "0.0.0.0"
        assert sock_host != "0.0.0.0"
        assert host == "127.0.0.1"
        assert sock_host == "127.0.0.1"
        assert not is_wildcard_host(host)
        assert not is_wildcard_host(sock_host)
    finally:
        server.server_close()


def test_server_bind_guard_fails_if_address_is_forced_to_wildcard():
    server = create_server(host="127.0.0.1", port=0, device="cpu")
    try:
        server.server_address = ("0.0.0.0", 0)
        with pytest.raises((ValueError, OSError)):
            server.server_bind()
    finally:
        server.server_close()


def test_whisper_package_has_no_wildcard_listen_address():
    root = os.path.join(os.path.dirname(__file__), "..", "whisper")
    offenders = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
            if 'HTTPServer(("0.0.0.0"' in text or "HTTPServer(('0.0.0.0'" in text:
                offenders.append(path)
            if '.bind(("0.0.0.0"' in text or ".bind(('0.0.0.0'" in text:
                offenders.append(path)
    assert offenders == []
