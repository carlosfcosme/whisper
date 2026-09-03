import socket

import pytest

from whisper.bind import (
    LOOPBACK_HOST,
    BindError,
    bind_localhost,
    default_bind_host,
    require_bind_127_0_0_1,
)

ALL_INTERFACES = ".".join(("0", "0", "0", "0"))


def test_default_bind_host_is_127_0_0_1():
    assert LOOPBACK_HOST == "127.0.0.1"
    assert default_bind_host() == "127.0.0.1"
    assert require_bind_127_0_0_1(None) == "127.0.0.1"


@pytest.mark.parametrize(
    "host", [ALL_INTERFACES, "::", "", "8.8.8.8", "10.0.0.1", "example.com"]
)
def test_refuse_non_localhost_bind(host):
    with pytest.raises(BindError, match="127.0.0.1"):
        require_bind_127_0_0_1(host)


def test_bind_host_env_cannot_wildcard(monkeypatch):
    monkeypatch.setenv("WHISPER_BIND_HOST", ALL_INTERFACES)
    with pytest.raises(BindError):
        default_bind_host()


def test_bind_localhost_listens_on_127_0_0_1():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.settimeout(1)
        host, port = bind_localhost(server, 0)
        assert host == "127.0.0.1"
        assert port > 0
        server.listen(1)
        client = socket.create_connection(("127.0.0.1", port), timeout=1)
        try:
            conn, addr = server.accept()
            try:
                assert addr[0] == "127.0.0.1"
                client.sendall(b"ok")
                assert conn.recv(2) == b"ok"
            finally:
                conn.close()
        finally:
            client.close()
    finally:
        server.close()


def test_wildcard_socket_bind_is_blocked_in_unit_tests():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError, match="127.0.0.1"):
            sock.bind((ALL_INTERFACES, 0))
    finally:
        sock.close()
