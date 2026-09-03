import socket

import pytest

import whisper
from whisper.runtime import (
    BindError,
    bind_localhost,
    default_bind_host,
    refuse_non_localhost_bind,
)
from whisper.serve import create_server, main


def test_default_bind_host_is_127_0_0_1():
    assert default_bind_host() == "127.0.0.1"
    assert whisper.default_bind_host() == "127.0.0.1"


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "", "8.8.8.8", "example.com"])
def test_refuse_non_localhost_bind(host):
    with pytest.raises(BindError, match="127.0.0.1"):
        refuse_non_localhost_bind(host)


def test_bind_localhost_accepts_loopback():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.settimeout(1)
        host, port = bind_localhost(server, 0)
        assert host == "127.0.0.1"
        server.listen(1)
        client = socket.create_connection(("127.0.0.1", port), timeout=1)
        try:
            conn, addr = server.accept()
            try:
                assert addr[0] == "127.0.0.1"
            finally:
                conn.close()
        finally:
            client.close()
    finally:
        server.close()


def test_wildcard_socket_bind_blocked_in_tests():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError, match="127.0.0.1"):
            sock.bind(("0.0.0.0", 0))
    finally:
        sock.close()


def test_create_server_binds_127_0_0_1():
    httpd = create_server("127.0.0.1", 0)
    try:
        assert httpd.server_address[0] == "127.0.0.1"
    finally:
        httpd.server_close()


def test_cli_refuses_wildcard_host():
    assert main(["--host", "0.0.0.0", "--port", "0"]) == 2


def test_check_loopback_bind_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "check_loopback_bind.py"
    spec = importlib.util.spec_from_file_location("check_loopback_bind", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main() == 0
