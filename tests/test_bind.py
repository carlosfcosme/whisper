import json
import socket
import threading
import urllib.request
from pathlib import Path

import pytest

from whisper.bind import (
    ALL_INTERFACES,
    LOOPBACK_HOST,
    BindError,
    require_loopback_host,
)
from whisper.serve import create_server
from whisper.serve import main as serve_main
from whisper.serve import serve

ROOT = Path(__file__).resolve().parents[1]


def test_require_loopback_host_defaults_to_ipv4_loopback():
    assert require_loopback_host() == "127.0.0.1"
    assert require_loopback_host("127.0.0.1") == "127.0.0.1"
    assert require_loopback_host("localhost") == "127.0.0.1"
    assert require_loopback_host("LOCALHOST") == "127.0.0.1"
    assert LOOPBACK_HOST == "127.0.0.1"


@pytest.mark.parametrize(
    "host",
    [
        ALL_INTERFACES,
        "::",
        "*",
        "",
        "192.168.1.1",
        "8.8.8.8",
        "example.com",
        "::1",
        "127.0.0.2",
    ],
)
def test_require_loopback_host_rejects_non_loopback(host):
    with pytest.raises(BindError):
        require_loopback_host(host)


@pytest.mark.parametrize(
    "host",
    [
        ALL_INTERFACES,
        "::",
        "*",
        "",
        "192.168.1.1",
        "10.0.0.1",
        "8.8.8.8",
        "example.com",
        "::1",
    ],
)
def test_create_server_rejects_non_loopback(host):
    with pytest.raises(BindError):
        create_server(host=host, port=0)


def test_listening_socket_getsockname_is_127_0_0_1():
    httpd = serve(port=0)
    try:
        sock_host, sock_port = httpd.socket.getsockname()[:2]
        assert sock_host == "127.0.0.1"
        assert sock_host != ALL_INTERFACES
        assert sock_port > 0
        server_host = httpd.server_address[0]
        assert server_host == "127.0.0.1"
    finally:
        httpd.server_close()


def test_create_server_default_binds_loopback_and_answers():
    httpd = create_server(port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = httpd.server_address[:2]
        assert host == LOOPBACK_HOST
        assert httpd.socket.getsockname()[0] == "127.0.0.1"
        assert port > 0
        with urllib.request.urlopen(
            "http://127.0.0.1:{}/health".format(port), timeout=2
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "ok"
        assert data["bind"] == "127.0.0.1"
        assert data["weights"] is False
        with socket.create_connection((host, port), timeout=1):
            pass
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.mark.parametrize("host", [ALL_INTERFACES, "*", "192.168.1.1", "example.com"])
def test_serve_cli_rejects_non_loopback(host):
    assert serve_main(["--host", host, "--port", "0"]) == 2


def test_application_sources_do_not_contain_all_interfaces_literal():
    token = ALL_INTERFACES
    for rel in ("whisper", ".cursor"):
        tree = ROOT / rel
        if not tree.exists():
            continue
        for path in tree.rglob("*"):
            if not path.is_file() or path.suffix in {".pyc"}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            assert token not in text, "{} must not contain {}".format(path, token)
