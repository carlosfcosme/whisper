import json
import socket
import threading
import urllib.request

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


def test_bind_host_env_cannot_wildcard(monkeypatch):
    monkeypatch.setenv("WHISPER_BIND_HOST", "0.0.0.0")
    with pytest.raises(BindError, match="0.0.0.0"):
        default_bind_host()


def test_make_server_refuses_wildcard():
    with pytest.raises(BindError):
        make_server(host="0.0.0.0", port=0)


def test_serve_binds_127_0_0_1_and_answers():
    server = make_server(host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        assert host == "127.0.0.1"
        assert port > 0
        with urllib.request.urlopen("http://127.0.0.1:{}/".format(port)) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        assert data["ok"] is True
        assert data["device"] == "cpu"
        assert data["host"] == "127.0.0.1"
    finally:
        server.shutdown()
        server.server_close()


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
            sock.bind(("0.0.0.0", 0))
    finally:
        sock.close()
