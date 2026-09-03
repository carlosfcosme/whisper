import socket

import pytest

from whisper.localhost import (
    NonLocalhostBindError,
    bind_address,
    bind_localhost,
    is_loopback_host,
)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "127.0.0.2", "::1"])
def test_bind_address_accepts_loopback(host):
    bound = bind_address(host)
    assert is_loopback_host(bound)
    if host == "localhost":
        assert bound == "127.0.0.1"


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "::", "*", "", "8.8.8.8", "1.1.1.1", "example.com"],
)
def test_bind_address_refuses_non_localhost(host):
    with pytest.raises(NonLocalhostBindError, match="localhost"):
        bind_address(host)


def test_bind_address_reads_env(monkeypatch):
    monkeypatch.setenv("WHISPER_BIND_HOST", "127.0.0.1")
    assert bind_address() == "127.0.0.1"
    monkeypatch.setenv("WHISPER_BIND_HOST", "0.0.0.0")
    with pytest.raises(NonLocalhostBindError):
        bind_address()


def test_service_socket_binds_loopback_only():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        host, port = bind_localhost(server, 0)
        assert is_loopback_host(host)
        assert host == "127.0.0.1"
        assert port > 0
        server.listen(1)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.settimeout(1)
            client.connect(("127.0.0.1", port))
            assert client.getpeername()[0] == "127.0.0.1"
            conn, peer = server.accept()
            with conn:
                assert is_loopback_host(peer[0])
