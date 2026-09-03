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
from whisper.serve import create_server
from whisper.serve import main as serve_main


def test_default_bind_host_is_127_0_0_1():
    assert DEFAULT_BIND_HOST == "127.0.0.1"
    assert default_bind_host() == "127.0.0.1"
    assert whisper.default_bind_host() == "127.0.0.1"


@pytest.mark.parametrize(
    "host",
    ["", "8.8.8.8", "10.0.0.1", "example.com", ".".join(["0"] * 4)],
)
def test_refuse_non_localhost_bind(host):
    with pytest.raises(BindError, match="127.0.0.1"):
        refuse_non_localhost_bind(host)


def test_bind_host_env_cannot_wildcard(monkeypatch):
    monkeypatch.setenv("WHISPER_BIND_HOST", ".".join(["0"] * 4))
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


def test_serve_health_binds_127_0_0_1():
    httpd = create_server(port=0)
    host, port = httpd.server_address[:2]
    assert host == "127.0.0.1"
    assert port > 0
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=2
        ) as response:
            assert response.status == 200
            assert response.read() == b"ok\n"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_serve_cli_refuses_wildcard_before_bind():
    unspecified = ".".join(["0"] * 4)
    assert serve_main(["--host", unspecified, "--port", "0"]) == 2


def test_wildcard_socket_bind_is_blocked_in_unit_tests():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError, match="127.0.0.1"):
            sock.bind((".".join(["0"] * 4), 0))
    finally:
        sock.close()
