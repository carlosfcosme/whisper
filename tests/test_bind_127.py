"""127.0.0.1 bind checks: loopback ok, wildcards and public hosts refused."""

import socket

import pytest

import whisper
from whisper.runtime import (
    DEFAULT_BIND_HOST,
    BindError,
    bind_localhost,
    default_bind_host,
    refuse_non_localhost_bind,
)
from whisper.serve import make_server


def test_default_bind_host_is_127_0_0_1():
    assert DEFAULT_BIND_HOST == "127.0.0.1"
    assert default_bind_host() == "127.0.0.1"
    assert whisper.default_bind_host() == "127.0.0.1"


@pytest.mark.parametrize(
    "host", ["0.0.0.0", "::", "", "8.8.8.8", "10.0.0.1", "example.com"]
)
def test_refuse_non_localhost_bind(host):
    with pytest.raises(BindError, match="127.0.0.1"):
        refuse_non_localhost_bind(host)


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


@pytest.mark.parametrize("host", ["0.0.0.0", "8.8.8.8"])
def test_socket_bind_non_loopback_is_blocked(host):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError, match="127.0.0.1"):
            sock.bind((host, 0))
    finally:
        sock.close()


def test_make_server_binds_127_0_0_1():
    httpd = make_server(port=0)
    try:
        host, port = httpd.server_address[:2]
        assert host == "127.0.0.1"
        assert port > 0
    finally:
        httpd.server_close()


def test_make_server_refuses_0_0_0_0():
    with pytest.raises(BindError, match="0.0.0.0"):
        make_server(host="0.0.0.0", port=0)
