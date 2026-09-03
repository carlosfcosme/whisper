import socket
from pathlib import Path

import pytest

from whisper.bind import (
    ALL_INTERFACES,
    LOOPBACK_HOST,
    BindError,
    assert_loopback_listen,
    require_loopback_host,
)
from whisper.serve import create_server
from whisper.serve import main as serve_main

ROOT = Path(__file__).resolve().parents[1]


def test_require_loopback_host_defaults_to_ipv4_loopback():
    assert require_loopback_host() == "127.0.0.1"
    assert require_loopback_host("127.0.0.1") == "127.0.0.1"
    assert require_loopback_host("localhost") == "127.0.0.1"
    assert require_loopback_host("LOCALHOST") == "127.0.0.1"


@pytest.mark.parametrize(
    "host",
    [
        ALL_INTERFACES,
        "0.0.0.0",
        "::",
        "*",
        "",
        "192.168.1.1",
        "8.8.8.8",
        "example.com",
        "::1",
    ],
)
def test_require_loopback_host_rejects_non_loopback(host):
    with pytest.raises(BindError):
        require_loopback_host(host)


def test_create_server_rejects_all_interfaces():
    with pytest.raises(BindError):
        create_server(host="0.0.0.0", port=0)


def test_create_server_binds_loopback():
    httpd = create_server(host="127.0.0.1", port=0)
    try:
        host, port = httpd.server_address[:2]
        assert host == LOOPBACK_HOST
        assert port > 0
        assert assert_loopback_listen(httpd) == LOOPBACK_HOST
        with socket.create_connection((host, port), timeout=1):
            pass
    finally:
        httpd.server_close()


def test_assert_loopback_listen_fails_on_all_interfaces_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((ALL_INTERFACES, 0))
        sock.listen(1)
        assert sock.getsockname()[0] == ALL_INTERFACES
        with pytest.raises(BindError):
            assert_loopback_listen(sock)
    finally:
        sock.close()


def test_assert_loopback_listen_fails_on_lan_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.2", 0))
        except OSError:
            pytest.skip("127.0.0.2 is not bindable on this host")
        sock.listen(1)
        with pytest.raises(BindError):
            assert_loopback_listen(sock)
    finally:
        sock.close()


def test_ci_bind_guard_runs_live_listen_check():
    yml = (ROOT / ".github/workflows/test.yml").read_text(encoding="utf-8")
    assert "check_no_all_interfaces.sh" in yml
    assert "check_loopback_listen.py" in yml


def test_serve_cli_rejects_all_interfaces():
    assert serve_main(["--host", "0.0.0.0", "--port", "9"]) == 2


def test_application_sources_do_not_contain_all_interfaces_literal():
    token = "0.0.0.0"
    for rel in ("whisper", ".cursor"):
        tree = ROOT / rel
        if not tree.exists():
            continue
        for path in tree.rglob("*"):
            if not path.is_file() or path.suffix in {".pyc"}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            assert token not in text, "{} must not contain {}".format(path, token)
