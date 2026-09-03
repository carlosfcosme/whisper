"""CI: require loopback-only binding."""

from __future__ import annotations

import json
import socket
import threading
from urllib.request import urlopen

import pytest

from whisper.bind import LOOPBACK_HOST, BindError, bind_loopback, require_loopback_host
from whisper.serve import create_server, main


@pytest.mark.parametrize(
    "host",
    [None, "127.0.0.1", "localhost"],
)
def test_require_loopback_host_accepts_canonical(host):
    assert require_loopback_host(host) == LOOPBACK_HOST


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "", "::", "*", "8.8.8.8", "192.168.1.1", "example.com", "127.0.0.2"],
)
def test_require_loopback_host_rejects_non_canonical(host):
    with pytest.raises(BindError):
        require_loopback_host(host)


def test_bind_loopback_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        host, port = bind_loopback(sock, 0)
        assert host == "127.0.0.1"
        assert port > 0
        assert sock.getsockname()[0] == "127.0.0.1"
    finally:
        sock.close()


def test_create_server_binds_127():
    httpd = create_server("127.0.0.1", 0)
    try:
        bound_host, bound_port = httpd.server_address[:2]
        assert bound_host == "127.0.0.1"
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        with urlopen(f"http://127.0.0.1:{bound_port}/health", timeout=2) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert payload["status"] == "ok"
        assert payload["bind"] == "127.0.0.1"
        assert payload["weights"] is False
        assert payload["network"] == "loopback"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_create_server_refuses_all_interfaces():
    with pytest.raises(BindError):
        create_server("0.0.0.0", 0)


def test_serve_cli_rejects_all_interfaces():
    assert main(["--host", "0.0.0.0", "--port", "0"]) == 2
