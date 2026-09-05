"""Runtime bind guard: listen on 127.0.0.1; fail if the host is 0.0.0.0."""

import socket

import pytest

from whisper.bind import ALL_INTERFACES, BindError, require_loopback_host
from whisper.serve import create_server


def test_bind_guard_defaults_to_loopback():
    assert require_loopback_host() == "127.0.0.1"
    assert require_loopback_host("127.0.0.1") == "127.0.0.1"


def test_bind_guard_fails_on_all_interfaces():
    with pytest.raises(BindError, match="all-interfaces"):
        require_loopback_host(ALL_INTERFACES)
    with pytest.raises(BindError):
        require_loopback_host("0.0.0.0")
    with pytest.raises(BindError):
        create_server(host="0.0.0.0", port=0)


def test_bind_guard_runtime_listens_on_127_0_0_1():
    httpd = create_server(host="127.0.0.1", port=0)
    try:
        host, port = httpd.server_address[:2]
        assert host == "127.0.0.1"
        with socket.create_connection((host, port), timeout=1):
            pass
    finally:
        httpd.server_close()
