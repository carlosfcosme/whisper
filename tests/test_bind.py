"""Helpers bind 127.0.0.1 only. Does not load weights."""

import socket

import pytest

from whisper.bind import BIND_HOST, BindError, bind_tcp, require_bind_host


def test_require_bind_host_defaults_to_loopback():
    assert BIND_HOST == "127.0.0.1"
    assert require_bind_host() == "127.0.0.1"
    assert require_bind_host(None) == "127.0.0.1"
    assert require_bind_host("127.0.0.1") == "127.0.0.1"


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "::", "[::]", "localhost", "::1", "192.168.1.1", "8.8.8.8", ""],
)
def test_require_bind_host_rejects_non_loopback(host):
    with pytest.raises(BindError, match="127.0.0.1"):
        require_bind_host(host)


def test_bind_tcp_listens_on_loopback():
    sock = bind_tcp(0)
    try:
        host, port = sock.getsockname()
        assert host == "127.0.0.1"
        assert port > 0
        sock.listen(1)
        client = socket.create_connection((host, port), timeout=1)
        try:
            accepted, addr = sock.accept()
            try:
                assert addr[0] == "127.0.0.1"
            finally:
                accepted.close()
        finally:
            client.close()
    finally:
        sock.close()


def test_bind_tcp_refuses_wildcard():
    with pytest.raises(BindError, match="127.0.0.1"):
        bind_tcp(0, host="0.0.0.0")
